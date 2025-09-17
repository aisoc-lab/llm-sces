import pytest
from modules.case_classifier import classify_case


def make_empty_results():
    return {
        "With History": {
            "match_cases": [],
            "non_match_cases": [],
            "non_response_cases": [],
            "other_cases": []
        },
        "Without History": {
            "match_cases": [],
            "non_match_cases": [],
            "non_response_cases": [],
            "other_cases": []
        },
    }


class DummyExtract:
    """Dummy numeric/string extractor"""
    def __init__(self, mapping=None):
        self.mapping = mapping or {}

    def __call__(self, text):
        return self.mapping.get(text, text)


class DummyDecision:
    """Dummy decision extractor (returns text unchanged unless overridden)"""
    def __init__(self, mapping=None):
        self.mapping = mapping or {}

    def __call__(self, text):
        return self.mapping.get(text, text)


def test_discrimeval_match():
    results = make_empty_results()
    result = {
        "Original Answer": "Yes",
        "Revised Answer With History": "Yes",
        "Revised Answer Without History": "No"
    }
    out = classify_case(
        result, "DiscrimEval", results,
        extract_fn=DummyExtract(), extract_decision=DummyDecision()
    )
    assert result in out["With History"]["match_cases"]
    assert result in out["Without History"]["non_match_cases"]


def test_folktexts_match_and_nonmatch():
    results = make_empty_results()
    result = {
        "Original Answer": "Below $50,000",
        "Revised Answer With History": "Above $50,000",
        "Revised Answer Without History": "Below $50,000"
    }
    out = classify_case(
        result, "Folktexts", results,
        extract_fn=DummyExtract(), extract_decision=DummyDecision()
    )
    assert result in out["With History"]["non_match_cases"]
    assert result in out["Without History"]["match_cases"]


def test_gsm8k_cases():
    results = make_empty_results()
    extractor = DummyExtract(mapping={
        "42": "42", "43": "43", "dummy": "42"
    })
    result = {
        "Original Answer": "dummy",  # extractor → "42"
        "Revised Answer With History": "43",
        "Revised Answer Without History": "",
        "Target": "43"
    }
    out = classify_case(result, "GSM8K", results, extract_fn=extractor, extract_decision=DummyDecision())
    assert result in out["With History"]["non_match_cases"]
    assert result in out["Without History"]["non_response_cases"]


def test_sst2_binary():
    results = make_empty_results()
    decider = DummyDecision(mapping={"pos": "Positive", "neg": "Negative"})
    result = {
        "Original Answer": "pos",
        "Revised Answer With History": "neg",
        "Revised Answer Without History": "pos"
    }
    out = classify_case(result, "SST2", results, extract_fn=DummyExtract(), extract_decision=decider)
    assert result in out["With History"]["non_match_cases"]
    assert result in out["Without History"]["match_cases"]


def test_twitter_cases():
    results = make_empty_results()
    decider = DummyDecision(mapping={"b": "Bearish", "n": "Neutral"})
    result = {
        "Original Answer": "b",
        "Revised Answer With History": "n",
        "Revised Answer Without History": "x"
    }
    out = classify_case(result, "Twitter", results, extract_fn=DummyExtract(), extract_decision=decider)
    assert result in out["With History"]["non_match_cases"]
    assert result in out["Without History"]["other_cases"]


def test_nli_cases():
    results = make_empty_results()
    decider = DummyDecision()
    result = {
        "Original Answer": "Entail",
        "Revised Answer With History": "Contradict",
        "Revised Answer Without History": ""
    }
    out = classify_case(result, "NLI", results, extract_fn=DummyExtract(), extract_decision=decider)
    assert result in out["With History"]["non_match_cases"]
    assert result in out["Without History"]["non_response_cases"]
