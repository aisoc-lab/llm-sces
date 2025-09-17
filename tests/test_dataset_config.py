import pytest
from modules.dataset_config import get_dataset_config


@pytest.mark.parametrize("prompt,revised,altered", [
    ("Unconstraint", "REVISED SCENARIO:", "ALTERED SCENARIO:"),
    ("CoT", "REVISED SCENARIO:", "ALTERED SCENARIO:"),
    ("Rational_based", "ALTERED SCENARIO:", "ALTERED SCENARIO:"),
])
def test_discrimeval_prompts(prompt, revised, altered):
    scenario = {"scenario": "The sky is blue."}
    questions = [{"text": "What color is the sky?"}]
    current, question, decode_word, kwargs, decode_word_r = get_dataset_config(
        "DiscrimEval", scenario, questions=questions, index=0, prompt=prompt
    )
    assert current == "The sky is blue."
    assert question == "What color is the sky?"
    assert decode_word in (revised, altered)
    assert kwargs["scenario"] == "The sky is blue."
    assert kwargs["question"] == "What color is the sky?"
    assert decode_word_r == "RATIONALES:"


def test_folktexts_config():
    scenario = {"description": "Story text", "question": "What happens?", "choices": ["A", "B"]}
    current, question, decode_word, kwargs, decode_word_r = get_dataset_config("Folktexts", scenario, prompt="Unconstraint")
    assert current == "Story text"
    assert question == "What happens?"
    assert kwargs["description"] == "Story text"
    assert kwargs["choices"] == ["A", "B"]
    assert decode_word == "REVISED DATA:"


def test_gsm8k_config():
    scenario = {"question": "2+2=?"}
    current, question, decode_word, kwargs, decode_word_r = get_dataset_config("GSM8K", scenario, prompt="Unconstraint")
    assert current == "2+2=?"
    assert kwargs["question"] == "2+2=?"
    assert decode_word == "REVISED PROBLEM:"


def test_sst2_config():
    scenario = {"sentence": "This movie is great!"}
    current, question, decode_word, kwargs, decode_word_r = get_dataset_config("SST2", scenario, prompt="Unconstraint")
    assert current == "This movie is great!"
    assert kwargs["sentence"] == "This movie is great!"
    assert decode_word == "REVISED REVIEW:"


def test_twitter_config():
    scenario = {"text": "I love pizza!"}
    current, question, decode_word, kwargs, decode_word_r = get_dataset_config("Twitter", scenario, prompt="Unconstraint")
    assert current == "I love pizza!"
    assert kwargs["text"] == "I love pizza!"
    assert decode_word == "REVISED POST:"


def test_nli_config():
    scenario = {"premise": "All dogs bark", "hypothesis": "Dogs make sounds"}
    current, question, decode_word, kwargs, decode_word_r = get_dataset_config("NLI", scenario, prompt="CoT")
    assert current == "All dogs bark"
    assert question == "Dogs make sounds"
    assert kwargs["premise"] == "All dogs bark"
    assert kwargs["hypothesis"] == "Dogs make sounds"
    assert decode_word == "REVISED HYPOTHESIS:"


def test_discrimeval_with_filled_template():
    scenario = {"filled_template": "Template text"}
    current, question, decode_word, kwargs, decode_word_r = get_dataset_config("DiscrimEval", scenario, questions=[{"text": "Q"}], index=0)
    assert current == "Template text"
    assert kwargs["scenario"] == "Template text"


def test_discrimeval_with_plain_string_scenario():
    scenario = {"scenario": "Plain text scenario"}
    current, question, decode_word, kwargs, decode_word_r = get_dataset_config(
        "DiscrimEval", scenario, questions=["Q"], index=0, prompt="Unconstraint"
    )

    # Check current scenario
    assert current == "Plain text scenario"

    # Check question
    assert question == "Q"

    # Check decode words
    assert decode_word == "REVISED SCENARIO:"
    assert decode_word_r == "RATIONALES:"

    # Check format kwargs
    assert kwargs["scenario"] == "Plain text scenario"
    assert kwargs["question"] == "Q"


def test_invalid_dataset_raises():
    scenario = {"text": "invalid"}
    with pytest.raises(ValueError):
        get_dataset_config("UnknownDataset", scenario)


def test_invalid_prompt_type():
    scenario = {"scenario": "Something"}
    with pytest.raises(ValueError):
        get_dataset_config("DiscrimEval", scenario, prompt="InvalidPrompt")
