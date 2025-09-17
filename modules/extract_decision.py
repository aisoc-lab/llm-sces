import re

def get_extract_decision_function(dataset_name):
    """Return a dataset-specific decision extraction function."""

    # ---------------------------
    # Helpers
    # ---------------------------
    
    def _normalize_text(answer, *, lower=True, strip_punct=True):
        """Shared normalization for consistency with complement functions."""
        if answer is None:
            return ""
        # First remove generic whitespace
        text = str(answer).strip()
        # Then remove specific leading/trailing symbols
        text = text.strip("*").strip("\"'")
        if lower:
            text = text.lower()
        if strip_punct:
            text = text.rstrip(".!?")
        return " ".join(text.split())

    # ---------------------------
    # Dataset-specific functions
    # ---------------------------

    def discrimeval_extract_decision(answer):
        if not answer or not isinstance(answer, str):
            print(f"DEBUG: Unexpected answer format: {answer}")
            return ""
        cleaned = _normalize_text(answer)
        if "yes" in cleaned:
            return "Yes"
        elif "no" in cleaned:
            return "No"
        return ""

    def folktexts_extract_decision(answer):
        if not answer or not isinstance(answer, str):
            return ""
        cleaned = _normalize_text(answer)
        # Match with or without $ / comma
        if re.search(r"(above|over)\s*\$?\s*50,?000", cleaned):
            return "Above $50,000"
        elif re.search(r"(below|under)\s*\$?\s*50,?000", cleaned):
            return "Below $50,000"
        return ""

    def _format_number(num_value: float) -> str:
        """Normalize numeric answers."""
        return str(int(num_value)) if num_value.is_integer() else str(num_value)

    def gsm8k_extract_decision(answer):
        print(f"DEBUG: Processing answer: {repr(answer)} (Type: {type(answer)})")

        # Handle non-responses
        if answer is None or str(answer).strip() in {"'", "'.", " ' ", "", ".", ".."}:
            print("DEBUG: Answer classified as non_response")
            return "non_response"

        # Direct numeric passthrough
        if isinstance(answer, (int, float)):
            print(f"DEBUG: Directly returning numeric answer {answer}")
            return _format_number(float(answer))

        # Invalid type
        if not isinstance(answer, str):
            print(f"ERROR: Unexpected answer type: {type(answer)}")
            return ""

        # Normalize text safely
        try:
            cleaned = _normalize_text(answer)
            print(f"DEBUG: Cleaned answer: {repr(cleaned)}")
        except Exception as e:
            print(f"ERROR: Failed to clean answer {answer}: {e}")
            return ""

        # Non-response phrases
        if "only provide the final answer" in cleaned or cleaned in {"'", "'.", " ' ", "", ".", ".."}:
            print("DEBUG: Answer classified as non_response")
            return "non_response"

        # Extract first valid number
        number_match = re.search(r"-?\d+(\.\d+)?", cleaned)
        if number_match:
            raw_num = number_match.group()
            print(f"DEBUG: Extracted number: {raw_num}")
            try:
                return _format_number(float(raw_num))
            except ValueError:
                print(f"ERROR: Conversion failed for {raw_num}")
                return cleaned

        # Fallback → return normalized string
        print("DEBUG: No number found, returning cleaned answer")
        return cleaned

    def nli_extract_decision(answer):
        if not answer or not isinstance(answer, str):
            return ""
        cleaned = _normalize_text(answer)
        if "entail" in cleaned:
            return "Entail"
        elif "neutral" in cleaned:
            return "Neutral"
        elif "contradict" in cleaned:
            return "Contradict"
        return ""

    def twitter_extract_decision(answer):
        if not answer or not isinstance(answer, str):
            return ""
        cleaned = _normalize_text(answer)
        if "bearish" in cleaned:
            return "Bearish"
        elif "bullish" in cleaned:
            return "Bullish"
        elif "neutral" in cleaned:
            return "Neutral"
        return ""

    def sst2_extract_decision(answer):
        if not answer or not isinstance(answer, str):
            return ""
        cleaned = _normalize_text(answer, strip_punct=False)
        if "positive" in cleaned:
            return "Positive"
        elif "negative" in cleaned:
            return "Negative"
        return ""

    # ---------------------------
    # Dispatcher
    # ---------------------------
    if dataset_name == "DiscrimEval":
        return discrimeval_extract_decision
    elif dataset_name == "Folktexts":
        return folktexts_extract_decision
    elif dataset_name == "GSM8K":
        return gsm8k_extract_decision
    elif dataset_name == "SST2":
        return sst2_extract_decision
    elif dataset_name == "Twitter":
        return twitter_extract_decision
    elif dataset_name == "NLI":
        return nli_extract_decision
    else:
        raise ValueError(f"No extract decision function defined for dataset: {dataset_name}")

