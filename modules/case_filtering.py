import re

# Precompiled regex patterns
_space_pattern = re.compile(r"\s+")
_word_pattern = re.compile(r"\b\w+\b")
_gsm8k_alpha_pattern = re.compile(r"^[A-Za-z\s]+$")

# Dataset-specific thresholds
DATASET_THRESHOLDS = {
    "DiscrimEval": 15,
    "Folktexts": 60,
    "Twitter": 3,
    "NLI": 2,
    "SST2": 1,
    "GSM8K": 5,
}

def count_words(text):
    """Count words in text after normalizing spaces and removing non-alphabetic characters."""
    if not isinstance(text, str) or not text.strip():
        return 0
    
    normalized_text = _space_pattern.sub(" ", text).strip()
    return len(_word_pattern.findall(normalized_text))

def is_short_case(dataset_name, result, decode_word):
    revised_scenario = result.get("SCEs", "").strip()
    word_count = count_words(revised_scenario)

    # Handle GSM8K separately (extra condition)
    if dataset_name == "GSM8K":
        return word_count < DATASET_THRESHOLDS["GSM8K"] and bool(_gsm8k_alpha_pattern.fullmatch(revised_scenario))
    
    # Default threshold check
    threshold = DATASET_THRESHOLDS.get(dataset_name)
    return word_count < threshold if threshold is not None else False
