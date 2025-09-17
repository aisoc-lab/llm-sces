import pytest
import torch
import modules.conversation_rational_based as psq


# -------------------------
# Dummy components
# -------------------------

class DummyBatch(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add input_ids like a real HuggingFace batch
        if "input_ids" not in self:
            self["input_ids"] = torch.ones(1, 5, dtype=torch.long)
        self.input_ids = self["input_ids"]

    def to(self, device):
        return self


class DummyTokenizer:
    def __init__(self):
        self.eos_token_id = 0
        self.device = "cpu"

    def __call__(self, text, return_tensors="pt", truncation=True, max_length=512, **kwargs):
        return DummyBatch({"input_ids": torch.ones(1, 5, dtype=torch.long)})

    def apply_chat_template(self, chat, tokenize=False, add_generation_prompt=True, **kwargs):
        return " ".join([msg["content"] for msg in chat])

    def decode(self, ids, skip_special_tokens=True):
        # Always return something rationalizable
        return "RATIONALES: some rational text"


class DummyModel:
    def __init__(self):
        self.device = "cpu"
        self.model = self

    def generate(self, *args, **kwargs):
        return type("GenOut", (), {"sequences": torch.arange(5).unsqueeze(0)})


# -------------------------
# Fixtures
# -------------------------

@pytest.fixture
def dummy_env(monkeypatch):
    tokenizer = DummyTokenizer()
    model = DummyModel()

    # Mock dependencies
    monkeypatch.setattr(psq, "generate_response", lambda *a, **k: torch.ones(1, 10, dtype=torch.long))
    monkeypatch.setattr(psq, "extract_answer", lambda text: "answer")
    monkeypatch.setattr(psq, "save_results_to_file", lambda *a, **k: None)
    monkeypatch.setattr(psq, "calculate_edit_distance_metrics", lambda a, b: 42)
    monkeypatch.setattr(
        psq,
        "get_dataset_config",
        lambda dataset_name, scenario, questions, index, prompt:
            (None, None, "REVISED:", {"index": index, "revised_scenario": "DUMMY_SCE"}, "RATIONALES:")
    )

    return tokenizer, model


# -------------------------
# Helpers
# -------------------------

def _make_templates():
    return [
        {"user_prompt": "Prompt {index}"},
        {"rational_user_prompt": "Why? {answer}"},
        {"revised_user_prompt": "Revised {complement}"},
        {"user_prompt_with_history": "History {revised_scenario}"}
    ]


# -------------------------
# Tests
# -------------------------

def test_process_scenarios_happy_path(dummy_env):
    tokenizer, model = dummy_env
    scenarios = ["Scenario A", "Scenario B"]
    questions = ["Q1", "Q2"]
    metadata = [{} for _ in scenarios]
    complement_fn = lambda ans: "complemented"

    results = psq.process_scenarios_with_questions(
        scenarios, questions, metadata,
        model_name="DummyModel", dataset_name="DummyDataset",
        temperature=0.7, templates=_make_templates(),
        tokenizer=tokenizer, model=model,
        N=2, complement_fn=complement_fn, debug=True
    )

    assert isinstance(results, list)
    assert len(results) == 2
    for r in results:
        assert r["Scenario"] in scenarios
        assert r["Original Answer"] == "answer"
        assert r["Target"] == "complemented"
        assert isinstance(r["Rationals"], str)
        assert isinstance(r["SCEs"], str)
        assert isinstance(r["Revised Answer With History"], str)
        assert isinstance(r["Revised Answer Without History"], str)
        assert r["Normalized Edit Distance Percentage"] == 42


def test_process_scenarios_mismatched_lengths(dummy_env):
    tokenizer, model = dummy_env
    with pytest.raises(ValueError):
        psq.process_scenarios_with_questions(
            scenarios=["Scenario A"], questions=["Q1", "Q2"], metadata=[{}],
            model_name="Dummy", dataset_name="Dummy", temperature=0.5,
            templates=_make_templates(), tokenizer=tokenizer, model=model
        )


def test_process_scenarios_truncation(dummy_env):
    tokenizer, model = dummy_env
    scenarios = ["S1", "S2", "S3"]
    questions = ["Q1", "Q2", "Q3"]
    metadata = [{}, {}, {}]

    results = psq.process_scenarios_with_questions(
        scenarios, questions, metadata,
        model_name="Dummy", dataset_name="Dummy",
        temperature=0.7, templates=_make_templates(),
        tokenizer=tokenizer, model=model,
        N=2, complement_fn=lambda a: "C"
    )
    assert len(results) == 2
    assert results[0]["Scenario"] == "S1"
    assert results[-1]["Scenario"] == "S2"


def test_process_scenarios_empty_input(dummy_env):
    tokenizer, model = dummy_env
    results = psq.process_scenarios_with_questions(
        [], [], [], "Dummy", "Dummy", 0.7,
        _make_templates(), tokenizer, model,
        complement_fn=lambda a: "C"
    )
    assert results == []


def test_process_scenarios_complement_unknown(dummy_env):
    tokenizer, model = dummy_env
    scenarios = ["S1"]
    questions = ["Q1"]
    metadata = [{}]

    results = psq.process_scenarios_with_questions(
        scenarios, questions, metadata,
        "Dummy", "Dummy", 0.7,
        _make_templates(), tokenizer, model,
        complement_fn=lambda a: "Unknown"
    )
    assert results[0]["Target"] == "Unknown"


def test_process_scenarios_template_error(dummy_env):
    tokenizer, model = dummy_env
    scenarios = ["S1"]
    questions = ["Q1"]
    metadata = [{}]
    templates = [{"user_prompt": "This {bad_key} is missing"},
                 {"rational_user_prompt": "Why? {answer}"},
                 {"revised_user_prompt": "Revised {complement}"},
                 {"user_prompt_with_history": "History {revised_scenario}"}]

    results = psq.process_scenarios_with_questions(
        scenarios, questions, metadata,
        "Dummy", "Dummy", 0.7,
        templates, tokenizer, model,
        complement_fn=lambda a: "C"
    )
    # Should gracefully skip → no results
    assert results == []


def test_process_scenarios_revised_scenario_fallback(dummy_env, monkeypatch):
    tokenizer, model = dummy_env
    scenarios = ["S1"]
    questions = ["Q1"]
    metadata = [{}]

    # Force decode to return empty string → triggers fallback "NO SCE GENERATED"
    monkeypatch.setattr(tokenizer, "decode", lambda ids, skip_special_tokens=True: "")

    results = psq.process_scenarios_with_questions(
        scenarios, questions, metadata,
        "Dummy", "Dummy", 0.7,
        _make_templates(), tokenizer, model,
        complement_fn=lambda a: "C"
    )
    assert results[0]["SCEs"] == "NO SCE GENERATED"
