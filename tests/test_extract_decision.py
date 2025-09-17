import pytest
import modules.extract_decision as ed

get_extract_decision = ed.get_extract_decision_function


# ----------------------------
# DiscrimEval
# ----------------------------
@pytest.mark.parametrize("answer,expected", [
    ("Yes, definitely", "Yes"),
    ("no way!", "No"),
    ("maybe", ""),   # should not match
    ("", ""),        # empty
    (None, ""),      # None
])
def test_discrimeval_extract(answer, expected):
    fn = get_extract_decision("DiscrimEval")
    assert fn(answer) == expected


# ----------------------------
# Folktexts
# ----------------------------
@pytest.mark.parametrize("answer,expected", [
    ("Above 50000", "Above $50,000"),
    ("above $50,000.", "Above $50,000"),
    ("below 50000", "Below $50,000"),
    ("salary is under $50,000!", "Below $50,000"),
    ("nothing related", ""),
    (None, ""),
])
def test_folktexts_extract(answer, expected):
    fn = get_extract_decision("Folktexts")
    assert fn(answer) == expected


# ----------------------------
# GSM8K
# ----------------------------
@pytest.mark.parametrize("answer,expected", [
    (42, "42"),                        # int passthrough
    (42.0, "42"),                      # float, integer value
    (3.14, "3.14"),                    # float with decimal
    ("The answer is 100.", "100"),     # extract number
    ("It should be -5", "-5"),         # negative number
    ("No number here", "no number here"),  # returns cleaned text
    (None, "non_response"),            # non-response
    ("", "non_response"),
    (".", "non_response"),
    ("only provide the final answer", "non_response"),
])
def test_gsm8k_extract(answer, expected):
    fn = get_extract_decision("GSM8K")
    assert fn(answer) == expected


def test_gsm8k_invalid_type():
    fn = get_extract_decision("GSM8K")
    assert fn(["not", "a", "string"]) == ""


# ----------------------------
# NLI
# ----------------------------
@pytest.mark.parametrize("answer,expected", [
    ("entailment", "Entail"),
    ("neutral case", "Neutral"),
    ("contradiction here", "Contradict"),
    ("unknown", ""),
    (None, ""),
])
def test_nli_extract(answer, expected):
    fn = get_extract_decision("NLI")
    assert fn(answer) == expected


# ----------------------------
# Twitter
# ----------------------------
@pytest.mark.parametrize("answer,expected", [
    ("bearish market", "Bearish"),
    ("bullish trend", "Bullish"),
    ("neutral sentiment", "Neutral"),
    ("something else", ""),
    (None, ""),
])
def test_twitter_extract(answer, expected):
    fn = get_extract_decision("Twitter")
    assert fn(answer) == expected


# ----------------------------
# SST2
# ----------------------------
@pytest.mark.parametrize("answer,expected", [
    ("positive review", "Positive"),
    ("Negative tone", "Negative"),
    ("totally neutral", ""),
    ("", ""),
    (None, ""),
])
def test_sst2_extract(answer, expected):
    fn = get_extract_decision("SST2")
    assert fn(answer) == expected


# ----------------------------
# Dispatcher errors
# ----------------------------
def test_invalid_dataset_name():
    with pytest.raises(ValueError):
        get_extract_decision("UnknownDataset")
