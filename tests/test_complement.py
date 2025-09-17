import re
import pytest
from modules import complement


# -----------------------
# Tests for extract_answer_value
# -----------------------

def test_extract_answer_value_from_dict():
    ans = {"answer": "Yes"}
    assert complement.extract_answer_value(ans) == "Yes"

def test_extract_answer_value_from_string():
    ans = "No"
    assert complement.extract_answer_value(ans) == "No"

def test_extract_answer_value_from_other_type():
    ans = 123
    assert complement.extract_answer_value(ans) == "123"


# -----------------------
# Tests for DiscrimEval
# -----------------------

def test_complement_discrim_yes_to_no():
    fn = complement.get_complement_function("DiscrimEval")
    assert fn("Yes") == "No"

def test_complement_discrim_no_to_yes():
    fn = complement.get_complement_function("discrim")  # lowercase also works
    assert fn("No") == "Yes"

def test_complement_discrim_unknown():
    fn = complement.get_complement_function("DiscrimEval")
    assert fn("Maybe") == "Unknown"


# -----------------------
# Tests for Folktexts
# -----------------------

def test_complement_folktexts_below_to_above():
    fn = complement.get_complement_function("Folktexts")
    assert fn("Below $50,000") == "Above $50,000"

def test_complement_folktexts_above_to_below():
    fn = complement.get_complement_function("Folktexts")
    assert fn("Above $50,000") == "Below $50,000"

def test_complement_folktexts_unknown():
    fn = complement.get_complement_function("Folktexts")
    assert fn("something else") == "Unknown"


# -----------------------
# Tests for GSM8K
# -----------------------

def test_complement_gsm8k_adds_number():
    fn = complement.get_complement_function("GSM8K")
    out = fn("Answer is 42")
    assert re.match(r"^\d+$", out)  # must be numeric
    assert int(out) > 42

def test_complement_gsm8k_no_number():
    fn = complement.get_complement_function("GSM8K")
    assert fn("No number here") == "Unknown"


# -----------------------
# Tests for SST2
# -----------------------

def test_complement_sst2_positive_to_negative():
    fn = complement.get_complement_function("SST2")
    assert fn("Positive") == "negative"

def test_complement_sst2_negative_to_positive():
    fn = complement.get_complement_function("SST2")
    assert fn("Negative") == "positive"

def test_complement_sst2_unknown():
    fn = complement.get_complement_function("SST2")
    assert fn("Neutral") == "Unknown"


# -----------------------
# Tests for Twitter
# -----------------------

def test_complement_twitter_changes_sentiment():
    fn = complement.get_complement_function("Twitter")
    out = fn("Bullish")
    assert out in ["bearish", "neutral"]  # anything except "bullish"

def test_complement_twitter_unknown():
    fn = complement.get_complement_function("Twitter")
    assert fn("nonsense") == "Unknown"


# -----------------------
# Tests for NLI
# -----------------------

def test_complement_nli_changes_label():
    fn = complement.get_complement_function("NLI")
    out = fn("Entail")
    assert out in ["contradict", "neutral"]

def test_complement_nli_unknown():
    fn = complement.get_complement_function("NLI")
    assert fn("maybe") == "Unknown"


# -----------------------
# Tests for ValueError
# -----------------------

def test_complement_unknown_dataset():
    with pytest.raises(ValueError):
        complement.get_complement_function("UnknownDataset")
