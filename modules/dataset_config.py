def get_dataset_config(dataset_name, scenario, questions=None, index=None, prompt="Unconstraint"):
    """
    Returns dataset-specific components such as scenario, question, decode_word, and formatting kwargs.
    """
    format_kwargs, question = {}, None
    decode_word_r = "RATIONALES:"
    decode_word = None

    # Helper for prompt mapping
    def get_decode_word(prompt, revised, altered):
        if prompt in ("Unconstraint", "CoT"):
            return revised
        elif prompt == "Rational_based":
            return altered
        raise ValueError(f"Unknown prompt type: {prompt}")

    if dataset_name == "DiscrimEval":
        current_scenario = (
            scenario.get("filled_template")
            or scenario.get("scenario")
            or scenario
        )

        if questions is not None and index is not None:
            q = questions[index]
            question = q.get("text") if isinstance(q, dict) else q

        format_kwargs.update({"scenario": current_scenario, "question": question})
        decode_word = get_decode_word(prompt, "REVISED SCENARIO:", "ALTERED SCENARIO:")

    elif dataset_name == "Folktexts":
        current_scenario = scenario.get("description", "").strip()
        question = scenario.get("question", "").strip()

        format_kwargs.update({
            "description": current_scenario,
            "question": question,
            "choices": scenario.get("choices", [])
        })
        decode_word = get_decode_word(prompt, "REVISED DATA:", "ALTERED DATA:")

    elif dataset_name == "GSM8K":
        current_scenario = scenario.get("question", "").strip()
        format_kwargs["question"] = current_scenario
        decode_word = get_decode_word(prompt, "REVISED PROBLEM:", "ALTERED PROBLEM:")

    elif dataset_name == "SST2":
        current_scenario = scenario.get("sentence", "")
        format_kwargs["sentence"] = current_scenario
        decode_word = get_decode_word(prompt, "REVISED REVIEW:", "ALTERED REVIEW:")

    elif dataset_name == "Twitter":
        current_scenario = scenario.get("text", "").strip()
        format_kwargs["text"] = current_scenario
        decode_word = get_decode_word(prompt, "REVISED POST:", "ALTERED TWITTER POST:")

    elif dataset_name == "NLI":
        current_scenario = scenario.get("premise")
        question = scenario.get("hypothesis")

        format_kwargs.update({"premise": current_scenario, "hypothesis": question})
        decode_word = get_decode_word(prompt, "REVISED HYPOTHESIS:", "ALTERED HYPOTHESIS:")

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return current_scenario, question, decode_word, format_kwargs, decode_word_r
