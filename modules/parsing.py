import logging
import re

logger = logging.getLogger(__name__)

def extract_answer(decoded_response):
    """
    Extract the answer after the last occurrence of 'ANSWER:', 'assistant', or 'model'.
    Handles multi-line answers, trims trailing junk, and normalizes formatting.
    """
    try:
        if not decoded_response or not isinstance(decoded_response, str):
            return ""

        # Patterns that can mark the start of the answer
        start_markers = [r"ANSWER:"]

        # Find the *last* marker that appears
        last_pos = -1
        last_marker = None
        for marker in start_markers:
            match = list(re.finditer(marker, decoded_response, flags=re.IGNORECASE))
            if match:
                pos = match[-1].end()
                if pos > last_pos:
                    last_pos = pos
                    last_marker = marker

        if last_pos == -1:  # no marker found
            return ""

        # Extract text after the last marker
        answer = decoded_response[last_pos:].strip()

        # Cut off at any new marker that might follow
        stop_markers = [
            r"\?\n\n\*\*Answer:\*\*",
            r"\n```[\s\S]*?\*\*Answer:\*\*",
            r"ANSWER:"
        ]
        for stop in stop_markers:
            parts = re.split(stop, answer, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) > 1:
                answer = parts[0].strip()
                break

        # Normalize formatting
        answer = re.sub(r"\s+", " ", answer)  # collapse spaces/newlines
        answer = answer.strip("* \"'").lower().rstrip(".!?")

        return answer

    except Exception:
        logger.error("Error extracting answer", exc_info=True)
        return ""
