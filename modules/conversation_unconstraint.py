from tqdm import tqdm
from modules.dataset_config import get_dataset_config
from modules.parsing import extract_answer
from modules.io_utils import save_results_to_file
from modules.metrics import calculate_edit_distance_metrics, run_kmeans_and_variances, compute_cluster_variances, CosineDriftMetric  
import torch
import numpy as np

def generate_response(text, tokenizer, model, temperature, truncation, max_length, max_new_tokens):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=truncation,
        max_length=max_length
    ).to(model.device)
    
    return model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=(temperature > 0),
        pad_token_id=tokenizer.eos_token_id 
    )

def process_scenarios_with_questions(
    scenarios,
    questions,
    metadata,
    model_name,
    dataset_name,
    temperature,
    templates,
    tokenizer,
    model,
    N=None,
    complement_fn=None,
    debug=False,
    prompt="Unconstraint",
    truncation=True,
    max_length=512,
    max_new_tokens=500,
    k=2,
):
    if questions and len(scenarios) != len(questions):
        raise ValueError("The number of scenarios and questions must match.")

    if N is not None:
        scenarios = scenarios[:N]
        if questions:
            questions = questions[:N]
        if metadata:
            metadata = metadata[:N]

    results = []
    drift_metric = CosineDriftMetric(dim=0)

    # store hidden states for clustering
    last_input_token_states = []
    first_gen_token_states = []
    final_token_states = []   

    with tqdm(total=N or len(scenarios), desc=f"Processing {dataset_name}") as pbar:
        for i in range(len(scenarios)):
            format_kwargs = {}
            scenario = scenarios[i]

            _, _, decode_word, format_kwargs, _ = get_dataset_config(
                dataset_name=dataset_name,
                scenario=scenario,
                questions=questions,
                index=i,
                prompt="Unconstraint"
            )

            try:
                user_prompt = templates[0]["user_prompt"].format(**format_kwargs)
            except Exception as e:
                print(f"Prompt formatting error: {e}")
                continue

            # === 1. First question ===
            chat = [
                {"role": "user", "content": user_prompt}, 
                {"role": "assistant", "content": "ANSWER:"}
            ]

            try:
                chat_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                print("DEBUG [first q]: type(chat_text) =", type(chat_text), "value=", chat_text[:200] if isinstance(chat_text, str) else chat_text)
                outputs = generate_response(chat_text, tokenizer, model, temperature, truncation, max_length, max_new_tokens)
                decoded_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer = extract_answer(decoded_response)
                chat[-1] = {"role": "assistant", "content": f"ANSWER: {answer}"}

                # === 2. Complement & revised scenario ===
                print("DEBUG: answer =", answer)
                print("DEBUG: complement_fn =", complement_fn)
                complement = complement_fn(answer)
                format_kwargs["complement"] = complement

                revised_user_prompt = templates[1]["revised_user_prompt"].format(**format_kwargs)
                chat.append({"role": "user", "content": revised_user_prompt})

                # Seed with decode_word so model knows what to generate after
                chat.append({"role": "assistant", "content": f"{decode_word}"})

                chat_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                print("DEBUG: chat_text (string) =", chat_text[:200])

                tokenized_inputs = tokenizer(
                    chat_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(model.device)

                print("DEBUG: tokenized_inputs['input_ids'].shape =", tokenized_inputs["input_ids"].shape)

                input_ids = tokenized_inputs["input_ids"]
                print("DEBUG: input_ids shape =", input_ids.shape, "type =", type(input_ids))

                try:
                    _ = input_ids[0, -1, :]
                except Exception as e:
                    print("DEBUG: input_ids[0, -1, :] failed with:", e)

                try:
                    _ = input_ids[-1]
                    print("DEBUG: input_ids[-1] works, shape =", _.shape)
                except Exception as e:
                    print("DEBUG: input_ids[-1] failed with:", e)

                # get prompt length safely
                prompt_len = input_ids.size(1)
                print("DEBUG: prompt_len =", prompt_len)

                # --- hidden states for input (last token before generation) ---
                with torch.no_grad():
                    outputs = model(**tokenized_inputs, output_hidden_states=True, return_dict=True)
                    hidden_last_layer = outputs.hidden_states[-1]
                    print("DEBUG: pre-squeeze hidden_last_layer shape =", hidden_last_layer.shape)
                    if hidden_last_layer.dim() == 3:   # [batch, seq_len, hidden_dim]
                        hidden_last_layer = hidden_last_layer.squeeze(0)
                    print("DEBUG: post-squeeze hidden_last_layer shape =", hidden_last_layer.shape)
                    try:
                        last_input_token_states.append(hidden_last_layer[-1].cpu().numpy())
                        print("DEBUG: appended last_input_token_states, vector shape =", hidden_last_layer[-1].shape)
                    except Exception as e:
                        print("ERROR while appending last_input_token_states:", e)
                        raise

                # --- generate revised problem ---
                gen_out = model.generate(
                    **tokenized_inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    pad_token_id=tokenizer.eos_token_id 
                )

                generated_ids = gen_out.sequences[0]  # shape [seq_len] or [1, seq_len]
                print("DEBUG: generated_ids shape before unsqueeze =", generated_ids.shape)

                # ensure batch dimension
                if generated_ids.dim() == 1:
                    generated_ids = generated_ids.unsqueeze(0)  # [1, seq_len]
                print("DEBUG: generated_ids shape after unsqueeze =", generated_ids.shape)

                # --- hidden states for whole sequence (prompt + generated) ---
                with torch.no_grad():
                    full_outputs = model(generated_ids, output_hidden_states=True, return_dict=True)
                    hidden_last_layer = full_outputs.hidden_states[-1]
                    print("DEBUG: pre-squeeze full hidden_last_layer shape =", hidden_last_layer.shape)
                    if hidden_last_layer.dim() == 3:
                        hidden_last_layer = hidden_last_layer.squeeze(0)  # [seq_len, hidden_dim]
                    print("DEBUG: post-squeeze full hidden_last_layer shape =", hidden_last_layer.shape)

                    # guard against no new tokens
                    try:
                        if prompt_len < hidden_last_layer.size(0):
                            first_vec = hidden_last_layer[prompt_len].cpu().numpy()
                            first_gen_token_states.append(first_vec)
                            print("DEBUG: appended first_gen_token_states, vector shape =", first_vec.shape)
                        else:
                            print(f"WARNING: prompt_len {prompt_len} >= seq_len {hidden_last_layer.size(0)}")
                            first_vec = hidden_last_layer[-1].cpu().numpy()
                            first_gen_token_states.append(first_vec)

                        final_vec = hidden_last_layer[-1].cpu().numpy()
                        final_token_states.append(final_vec)
                        print("DEBUG: appended final_token_states, vector shape =", final_vec.shape)
                    except Exception as e:
                        print("ERROR while appending gen token states:", e)
                        raise

                revised_scenario = tokenizer.decode(
                    generated_ids[0], skip_special_tokens=True
                ).split(f"{decode_word}")[-1].split("ANSWER:")[0].strip()

                # Save in kwargs and chat
                format_kwargs["revised_scenario"] = revised_scenario
                chat[-1] = {"role": "assistant", "content": f"{decode_word} {revised_scenario}"}

                # --- compute drift ---
                with torch.no_grad():
                    encoded = tokenizer(
                        [str(scenario), str(revised_scenario)],
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).to(model.device)

                    embeds = model.model.embed_tokens(encoded["input_ids"])
                    print("DEBUG: embeds shape =", embeds.shape)
                    sentence_embed = embeds.mean(dim=1)
                    print("DEBUG: sentence_embed shape =", sentence_embed.shape)
                    drift = drift_metric.compute(sentence_embed).mean().item()
                    print("DEBUG: computed drift =", drift)

                edit = calculate_edit_distance_metrics(scenario, revised_scenario)

                # Fallback if nothing was produced
                if not revised_scenario:
                    revised_scenario = "NO SCE GENERATED"
            
                # === 3. Third question with history ===
                user_prompt_with_history = templates[2]["user_prompt_with_history"].format(**format_kwargs)
                chat.append({"role": "user", "content": user_prompt_with_history})
                chat.append({"role": "assistant", "content": "ANSWER:"})
                chat_text_with_history = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                print("DEBUG [with history]: type(chat_text_with_history) =", type(chat_text_with_history), "value=", chat_text_with_history[:200] if isinstance(chat_text_with_history, str) else chat_text_with_history)
                outputs = generate_response(chat_text_with_history, tokenizer, model, temperature, truncation, max_length, max_new_tokens)
                revised_answer_with_history = extract_answer(tokenizer.decode(outputs[0], skip_special_tokens=True))
                chat[-1] = {"role": "assistant", "content": f"ANSWER: {revised_answer_with_history}"}

                # === 4. Third question without history ===
                third_question_chat = [
                    {"role": "user", "content": user_prompt_with_history},
                    {"role": "assistant", "content": "ANSWER:"}
                ]
                chat_text_without_history = tokenizer.apply_chat_template(third_question_chat, tokenize=False, add_generation_prompt=True)
                outputs = generate_response(chat_text_without_history, tokenizer, model, temperature, truncation, max_length, max_new_tokens)
                print("DEBUG [without history]: type(chat_text_without_history) =", type(chat_text_without_history), "value=", chat_text_without_history[:200] if isinstance(chat_text_without_history, str) else chat_text_without_history)
                revised_answer_without_history = extract_answer(tokenizer.decode(outputs[0], skip_special_tokens=True))
                third_question_chat[-1] = {"role": "assistant", "content": f"ANSWER: {revised_answer_without_history}"}

                # === 5. Save results ===     
                result = {
                    "Scenario": scenario,
                    "Original Answer": answer,
                    "Target": complement,
                    "SCEs": revised_scenario,
                    "Revised Answer With History": revised_answer_with_history,
                    "Revised Answer Without History": revised_answer_without_history,
                    "Cosine Drift (Original vs Revised Problem)": round(drift, 4),
                    **edit,
                }
                results.append(result)
                if debug:
                    print(result)

            except Exception as e:
                print(f"Error at example {i}: {e}")
            pbar.update(1)

    # --- KMeans clustering ---
    if len(results) >= 2:
        first_np = np.stack(first_gen_token_states)
        last_input_np = np.stack(last_input_token_states)
        final_np = np.stack(final_token_states)

        cluster_summary = {}

        for name, X in [
            ("First Token", first_np),
            ("Last Input Token", last_input_np),
            ("Final Token in Revised Scenario", final_np),
        ]:
            cluster_summary[name] = {}

            for metric in ["euclidean", "cosine"]:
                for norm_flag in [False, True]:
                    label = f"{metric.capitalize()}_{'Normalized' if norm_flag else 'Raw'}"

                    kmeans, within, between = run_kmeans_and_variances(X, k, metric=metric, normalize=norm_flag)

                    # assign per-sample cluster indices
                    for i, result in enumerate(results):
                        result[f"Cluster Index ({name}, {label})"] = int(kmeans.labels_[i])

                    # save summary stats
                    cluster_summary[name][label] = {
                        "Within Cluster Variance": float(round(within, 6)),
                        "Between Cluster Variance": float(round(between, 6)),
                        "Ratio (Between/Within)": float(round(between / (within + 1e-9), 6))
                    }

        results.append({"Cluster Variance Info": cluster_summary})
        if debug:
            print("Clustering results added (Euclidean & Cosine, raw & normalized)")

    save_results_to_file(results, model_name, f"temp_{temperature}", N or len(scenarios), dataset_name, prompt)
    return results