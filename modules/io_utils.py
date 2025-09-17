import os
import json

# ==========================
# File loading / saving utils
# ==========================
def load_prompt_templates(prompt_file):
    """Load JSON prompt templates from file."""
    try:
        with open(prompt_file, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Error loading prompt file '{prompt_file}': {e}")

def save_results_to_file(results, model_name, temperature, N, dataset_name, prompt):
    """Save results to structured folder based on prompt, model, and dataset."""
    sanitized_model_name = model_name.replace("/", "_")
    sanitized_dataset_name = dataset_name.replace("/", "_")
    
    folder_name = f"{prompt}_{sanitized_model_name}_{sanitized_dataset_name}"
    os.makedirs(folder_name, exist_ok=True)
    
    file_name = os.path.join(
        folder_name, 
        f"results_{sanitized_model_name}_{sanitized_dataset_name}_temp_{temperature}_n_{N}.json"
    )
    with open(file_name, 'w') as file:
        json.dump(results, file, indent=4)
    
    print(f"Results saved to {file_name}")

def load_inputs(scenario_file, question_file=None, metadata_file=None):
    """Load scenarios, questions, and metadata JSON files if provided."""
    def _load_json(file_path):
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                raise RuntimeError(f"Error loading file '{file_path}': {e}")
        return None

    scenarios = _load_json(scenario_file)
    questions = _load_json(question_file)
    metadata = _load_json(metadata_file)

    return scenarios, questions, metadata


# ==========================
# Dataset ↔ File mapping
# ==========================

# Base paths
DATASET_FILES = {
    "DiscrimEval": ("Datasets/DiscrimEval/Scenarios.json", "Datasets/DiscrimEval/Questions.json"),
    "Folktexts": ("Datasets/Folktexts/Folktexts.json", None),
    "GSM8K": ("Datasets/GSM8K/GSM8K.json", None),
    "NLI": ("Datasets/NLI/NLI.json", None),
    "SST2": ("Datasets/SST2/SST2.json", None),
    "Twitter": ("Datasets/Twitter/Twitter.json", None),
}

PROMPT_FILES = {
    "Unconstraint": {
        "DiscrimEval": "Prompts/Unconstraint/DiscrimEval_prompts.json",
        "Folktexts": "Prompts/Unconstraint/Folktexts_prompts.json",
        "GSM8K": "Prompts/Unconstraint/GSM8K_prompts.json",
        "NLI": "Prompts/Unconstraint/NLI_prompts.json",
        "SST2": "Prompts/Unconstraint/SST2_prompts.json",
        "Twitter": "Prompts/Unconstraint/Twitter_prompts.json",
    },
    "CoT": {
        "DiscrimEval": "Prompts/CoT/DiscrimEval_prompts.json",
        "Folktexts": "Prompts/CoT/Folktexts_prompts.json",
        "GSM8K": "Prompts/CoT/GSM8K_prompts.json",
        "NLI": "Prompts/CoT/NLI_prompts.json",
        "SST2": "Prompts/CoT/SST2_prompts.json",
        "Twitter": "Prompts/CoT/Twitter_prompts.json",
    },
    "Rational_based": {
        "DiscrimEval": "Prompts/Rational_based/DiscrimEavl_prompts.json",  # kept typo to stay compatible
        "Folktexts": "Prompts/Rational_based/Folktexts_prompts.json",
        "GSM8K": "Prompts/Rational_based/GSM8K_prompts.json",
        "NLI": "Prompts/Rational_based/NLI_prompts.json",
        "SST2": "Prompts/Rational_based/SST2_prompts.json",
        "Twitter": "Prompts/Rational_based/Twitter_prompts.json",
    }
}


def infer_file_paths_from_dataset(dataset_name, prompt):
    """Infer scenario, question, and prompt file paths for a dataset/prompt combo."""
    if dataset_name not in DATASET_FILES:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    if prompt not in PROMPT_FILES:
        raise ValueError(f"Unsupported prompt type: {prompt}")

    scenario_file, question_file = DATASET_FILES[dataset_name]
    prompt_file = PROMPT_FILES[prompt][dataset_name]

    return {
        "scenario_file": scenario_file,
        "question_file": question_file,
        "prompt_file": prompt_file,
    }
