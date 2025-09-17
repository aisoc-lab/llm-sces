import pytest
from modules import case_filtering


# -----------------------
# Tests for count_words
# -----------------------

def test_count_words_basic():
    text = "This   is   a test"
    assert case_filtering.count_words(text) == 4

def test_count_words_with_punctuation():
    text = "Hello, world! 123"
    # word_pattern counts "Hello", "world", "123"
    assert case_filtering.count_words(text) == 3

def test_count_words_empty_and_nonstring():
    assert case_filtering.count_words("") == 0
    assert case_filtering.count_words("   ") == 0
    assert case_filtering.count_words(None) == 0
    assert case_filtering.count_words(123) == 0


# -----------------------
# Tests for is_short_case
# -----------------------

@pytest.mark.parametrize("dataset,threshold", [
    ("DiscrimEval", 15),
    ("Folktexts", 60),
    ("Twitter", 3),
    ("NLI", 2),
    ("SST2", 1),
])
def test_is_short_case_below_threshold(dataset, threshold):
    result = {"SCEs": "word " * (threshold - 1)}  # one less than threshold
    assert case_filtering.is_short_case(dataset, result, decode_word=None) is True

@pytest.mark.parametrize("dataset,threshold", [
    ("DiscrimEval", 15),
    ("Folktexts", 60),
    ("Twitter", 3),
    ("NLI", 2),
    ("SST2", 1),
])
def test_is_short_case_at_or_above_threshold(dataset, threshold):
    result = {"SCEs": "word " * threshold}  # exactly threshold
    assert case_filtering.is_short_case(dataset, result, decode_word=None) is False

def test_is_short_case_unknown_dataset():
    result = {"SCEs": "some text"}
    assert case_filtering.is_short_case("UnknownDS", result, decode_word=None) is False


# -----------------------
# GSM8K specific tests
# -----------------------

def test_is_short_case_gsm8k_below_threshold_alpha_only():
    result = {"SCEs": "one two three"}  # 3 words < 5
    assert case_filtering.is_short_case("GSM8K", result, decode_word=None) is True

def test_is_short_case_gsm8k_below_threshold_non_alpha():
    result = {"SCEs": "1 2 3"}  # not alphabetic
    assert case_filtering.is_short_case("GSM8K", result, decode_word=None) is False

def test_is_short_case_gsm8k_at_threshold():
    result = {"SCEs": "word " * 5}  # 5 words = threshold
    assert case_filtering.is_short_case("GSM8K", result, decode_word=None) is False
