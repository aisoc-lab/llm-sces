import re
import random
import string
from modules.utils import set_seed

set_seed(42)

def extract_answer_value(answer):
    """Extract the actual answer string if the model output is wrapped in a dict."""
    if isinstance(answer, dict):
        return answer.get("answer", "")
    return str(answer)

def get_complement_function(dataset_name: str):
    dataset_name = dataset_name.lower()

    def complement_discrim(answer):
        answer = extract_answer_value(answer)
        normalized = answer.strip("* ").strip("\"'").lower().rstrip(".!?")
        normalized = " ".join(normalized.split())

        words = normalized.split()
        if not words:
            return "Unknown"

        candidates = [words[0], words[-1]]  # check both first and last
        if "yes" in candidates:
            return "No"
        elif "no" in candidates:
            return "Yes"
        return "Unknown"

    def complement_folktexts(answer):
        answer = extract_answer_value(answer)
        normalized = answer.strip("* ").strip("\"'").lower().rstrip(".!?")
        normalized = " ".join(normalized.split())

        words = normalized.split()
        if not words:
            return "Unknown"

        candidates = [" ".join(words[:3]), " ".join(words[-3:])]  # capture multi-word labels
        if "below $50,000" in candidates:
            return "Above $50,000"
        elif "above $50,000" in candidates:
            return "Below $50,000"
        return "Unknown"

    def complement_gsm8k(answer):
        answer = extract_answer_value(answer)
        normalized = " ".join(answer.strip().split())

        # Find all integers in the normalized string
        matches = re.findall(r"\d+", normalized)
        if not matches:
            return "Unknown"

        # Always use the last number (final answer in CoT)
        base_value = int(matches[-1])
        return str(base_value + random.randint(1, 10))

    def complement_sst2(answer):
        answer = extract_answer_value(answer)
        normalized = answer.strip("* ").strip("\"'").lower().rstrip(".!?")
        normalized = " ".join(normalized.split())

        words = normalized.split()
        if not words:
            return "Unknown"

        candidates = [words[0], words[-1]]
        if "positive" in candidates:
            return "negative"
        elif "negative" in candidates:
            return "positive"
        return "Unknown"

    def complement_twitter(answer):
        answer = extract_answer_value(answer)
        normalized = answer.strip("* ").strip("\"'").lower().rstrip(".!?")
        normalized = " ".join(normalized.split())

        options = ["bearish", "bullish", "neutral"]
        words = normalized.split()
        if not words:
            return "Unknown"

        candidates = [words[0], words[-1]]
        for cand in candidates:
            if cand in options:
                others = [o for o in options if o != cand]
                return random.choice(others)

        return "Unknown"

    def complement_nli(answer):
        answer = extract_answer_value(answer).lower().strip()
        answer = answer.translate(str.maketrans("", "", string.punctuation))
        normalized = " ".join(answer.split())

        options = ["entail", "contradict", "neutral"]
        words = normalized.split()
        if not words:
            return "Unknown"

        candidates = [words[0], words[-1]]
        for cand in candidates:
            if cand in options:
                others = [o for o in options if o != cand]
                return random.choice(others)

        return "Unknown"

    if "discrim" in dataset_name:
        return complement_discrim
    elif "folk" in dataset_name:
        return complement_folktexts
    elif "gsm8k" in dataset_name:
        return complement_gsm8k
    elif "sst2" in dataset_name:
        return complement_sst2
    elif "twitter" in dataset_name:
        return complement_twitter
    elif "nli" in dataset_name:
        return complement_nli
    else:
        raise ValueError(f"No complement function defined for dataset: {dataset_name}")
