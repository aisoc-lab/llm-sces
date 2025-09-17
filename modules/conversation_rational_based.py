from tqdm import tqdm
from modules.dataset_config import get_dataset_config
from modules.parsing import extract_answer
from modules.io_utils import save_results_to_file
from modules.metrics import calculate_edit_distance_metrics 
from modules.conversation_unconstraint import generate_response

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
    prompt="Rational_based",
    truncation=True,
    max_length=512,
    max_new_tokens=500,
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

    with tqdm(total=N or len(scenarios), desc=f"Processing {dataset_name}") as pbar:
        for i in range(len(scenarios)):
            format_kwargs = {}
            scenario = scenarios[i]

            _, _, decode_word, format_kwargs, decode_word_r = get_dataset_config(
                dataset_name=dataset_name,
                scenario=scenario,
                questions=questions,
                index=i,
                prompt=prompt  
            )

            try:
                user_prompt = templates[0]["user_prompt"].format(**format_kwargs)
            except Exception as e:
                print(f"Prompt formatting error: {e}")
                continue

            chat = [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": "ANSWER:"}
            ]

            try:
                # --- Original answer ---
                chat_text = tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True
                )
                outputs = generate_response(
                    chat_text, tokenizer, model, temperature, truncation, max_length, max_new_tokens
                )
                decoded_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer = extract_answer(decoded_response)
                chat[-1]["content"] = f"ANSWER: {answer}"

                # --- Complement ---
                print("DEBUG: answer =", answer)
                print("DEBUG: complement_fn =", complement_fn)
                complement = complement_fn(answer)
                print("DEBUG: complement =", complement)
                format_kwargs["complement"] = complement
                format_kwargs["answer"] = answer 

                # --- Rationals ---
                rational_user_prompt = templates[1]["rational_user_prompt"].format(**format_kwargs)
                print("DEBUG: rational_user_prompt =", rational_user_prompt)

                chat.append({"role": "user", "content": rational_user_prompt})
                chat.append({"role": "assistant", "content": "RATIONALES:"})
                print("DEBUG: chat so far =", chat)

                chat_text = tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                print("DEBUG: chat_text[:500] =", repr(chat_text[:500]), "...")

                # Encode prompt
                inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
                input_len = inputs.input_ids.shape[1]
                print("DEBUG: input_len =", input_len)

                # Generate
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False if temperature == 0 else True,
                    temperature=temperature if temperature > 0 else None,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                print("DEBUG: outputs.sequences shape =", outputs.sequences.shape)

                # Slice only generated tokens (after prompt)
                gen_tokens = outputs.sequences[:, input_len:]
                print("DEBUG: generated token count =", gen_tokens.shape[1])

                decoded_response = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
                print("DEBUG: decoded_response =", repr(decoded_response))

                # Extract rationals
                if decode_word_r in decoded_response:
                    rationals = decoded_response.split(decode_word_r)[-1].strip()
                else:
                    rationals = decoded_response.strip()
                print("DEBUG: rationals =", repr(rationals))

                chat[-1]["content"] = f"RATIONALES: {rationals}"


                # --- Altered Scenario ---
                revised_user_prompt = templates[2]["revised_user_prompt"].format(**format_kwargs)
                chat.append({"role": "user", "content": revised_user_prompt})

                # Seed with decode_word so model knows what to generate after
                chat.append({"role": "assistant", "content": f"{decode_word} "})

                # Tokenize before generation
                chat_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                tokenized_inputs = tokenizer(chat_text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
                input_ids = tokenized_inputs["input_ids"]

                # Generate revised scenario
                gen_out = model.generate(
                    **tokenized_inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    return_dict_in_generate=True,
                    output_hidden_states=False,
                    pad_token_id=tokenizer.eos_token_id
                )

                generated_ids = gen_out.sequences[0]
                prompt_len = input_ids.shape[1]

                # Decode only the generation beyond the prompt
                decoded_response = tokenizer.decode(generated_ids[prompt_len:], skip_special_tokens=True)

                # Extract revised scenario cleanly
                revised_scenario = decoded_response.strip()
                for delimiter in ["ANSWER:", "?\n\n**Answer:**", "\n```\n\n**Answer:**"]:
                    if delimiter in revised_scenario:
                        revised_scenario = revised_scenario.split(delimiter)[0].strip()

                # Fallback if nothing was produced
                if not revised_scenario:
                    revised_scenario = "NO SCE GENERATED"

                # Save in kwargs and chat
                format_kwargs["revised_scenario"] = revised_scenario
                chat[-1] = {"role": "assistant", "content": f"{decode_word} {revised_scenario}"}

            except Exception as e:
                print(f"Altered scenario generation failed at example {i}: {e}")
                revised_scenario = ""

            # --- Third Question with history ---
            user_prompt_with_history = templates[3]["user_prompt_with_history"].format(**format_kwargs)
            chat.append({"role": "user", "content": user_prompt_with_history})
            chat.append({"role": "assistant", "content": "ANSWER:"})

            chat_text_with_history = tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            outputs = generate_response(
                chat_text_with_history, tokenizer, model, temperature, truncation, max_length, max_new_tokens
            )
            revised_answer_with_history = extract_answer(
                tokenizer.decode(outputs[0], skip_special_tokens=True)
            )
            chat[-1]["content"] = f"ANSWER: {revised_answer_with_history}"

            # --- Third question without history ---
            third_question_chat = [
                {"role": "user", "content": user_prompt_with_history},
                {"role": "assistant", "content": "ANSWER:"}
            ]
            chat_text_without_history = tokenizer.apply_chat_template(
                third_question_chat, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            outputs = generate_response(
                chat_text_without_history, tokenizer, model, temperature, truncation, max_length, max_new_tokens
            )
            revised_answer_without_history = extract_answer(
                tokenizer.decode(outputs[0], skip_special_tokens=True)
            )
            third_question_chat[-1]["content"] = f"ANSWER: {revised_answer_without_history}"

            # --- Log results ---
            edit_distance = calculate_edit_distance_metrics(scenario, revised_scenario)
            result = {
                "Scenario": scenario,
                "Original Answer": answer,
                "Target": complement,
                "Rationals": rationals,
                "SCEs": revised_scenario,
                "Revised Answer With History": revised_answer_with_history,
                "Revised Answer Without History": revised_answer_without_history,
                "Normalized Edit Distance Percentage": edit_distance,
            }

            results.append(result)
            if debug:
                print(result)

            pbar.update(1)

    save_results_to_file(results, model_name, f"temp_{temperature}", N or len(scenarios), dataset_name, prompt)
    return results
