import pytest
import torch
import numpy as np
from modules import conversation_unconstraint as cu


# -----------------------
# Dummy objects
# -----------------------

class DummyBatch(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self["input_ids"] = torch.ones(1, 5, dtype=torch.long)
        self.input_ids = self["input_ids"]

    def to(self, device):
        return self


class DummyTokenizer:
    def __init__(self):
        self.eos_token_id = 0
        self.device = "cpu"

    def __call__(self, text, return_tensors="pt", truncation=True, max_length=512, **kwargs):
        return DummyBatch()

    def to(self, device):
        return self

    def apply_chat_template(self, chat, tokenize=False, add_generation_prompt=True):
        return " ".join([msg["content"] for msg in chat])

    def decode(self, ids, skip_special_tokens=True):
        # Always return something to extract
        return "ANSWER: decoded text"


class DummyModel:
    def __init__(self):
        self.device = "cpu"
        self.model = self

    def __call__(self, *args, **kwargs):
        hidden = torch.ones(1, 5, 4)  # [batch, seq_len, hidden_dim]
        return type("DummyOutput", (), {"hidden_states": [hidden]})

    def generate(self, *args, **kwargs):
        if kwargs.get("return_dict_in_generate"):
            return type("GenOut", (), {"sequences": torch.arange(5).unsqueeze(0)})
        return torch.ones(1, 5, dtype=torch.long)

    def embed_tokens(self, x):
        return torch.ones(x.shape[0], x.shape[1], 4)


# -----------------------
# Fixtures
# -----------------------

@pytest.fixture
def dummy_env(monkeypatch):
    tokenizer = DummyTokenizer()
    model = DummyModel()

    # Patch dependencies
    monkeypatch.setattr(cu, "extract_answer", lambda text: "answer")
    monkeypatch.setattr(cu, "save_results_to_file", lambda *a, **k: None)
    monkeypatch.setattr(cu, "calculate_edit_distance_metrics", lambda a, b: {"edit": 1})
    monkeypatch.setattr(
        cu, "run_kmeans_and_variances",
        lambda X, k, metric="euclidean", normalize=False:
            (type("KM", (), {"labels_": np.zeros(X.shape[0], dtype=int)}), 1.0, 2.0)
    )
    monkeypatch.setattr(cu, "compute_cluster_variances", lambda *a, **k: (1.0, 2.0))
    monkeypatch.setattr(cu, "CosineDriftMetric",
        lambda dim: type("CDM", (), {"compute": lambda self, x: torch.tensor([0.5, 0.5])})()
    )
    monkeypatch.setattr(cu, "get_dataset_config",
        lambda dataset_name, scenario, questions, index, prompt:
            (None, None, "REVISED:", {"index": index, "revised_scenario": "DUMMY_SCE"}, None)
    )

    return tokenizer, model


# -----------------------
# Tests
# -----------------------

def test_generate_response(dummy_env):
    tokenizer, model = dummy_env
    outputs = cu.generate_response("hi", tokenizer, model, 0.7, True, 10, 5)
    assert isinstance(outputs, torch.Tensor)
    assert outputs.shape[0] == 1


def test_process_scenarios_with_questions_flow(dummy_env):
    tokenizer, model = dummy_env
    scenarios = ["Scenario A", "Scenario B"]
    questions = ["Q1", "Q2"]
    metadata = [{} for _ in scenarios]
    templates = [
        {"user_prompt": "Prompt {index}"},
        {"revised_user_prompt": "Revised {complement}"},
        {"user_prompt_with_history": "History {revised_scenario}"}
    ]
    complement_fn = lambda ans: "complemented"

    results = cu.process_scenarios_with_questions(
        scenarios, questions, metadata,
        "DummyModel", "DummyDataset", 0.7,
        templates, tokenizer, model,
        N=2, complement_fn=complement_fn, debug=True,
        max_new_tokens=5, k=2,
    )

    assert isinstance(results, list)
    assert all(isinstance(r, dict) for r in results)
    assert results[0]["Scenario"] == "Scenario A"
    assert results[0]["Original Answer"] == "answer"
    assert results[0]["Target"] == "complemented"
    assert "Cluster Variance Info" in results[-1]


def test_process_scenarios_empty_input(dummy_env):
    tokenizer, model = dummy_env
    results = cu.process_scenarios_with_questions(
        [], [], [], "Dummy", "Dummy", 0.7,
        [{"user_prompt": "x"}, {"revised_user_prompt": "y"}, {"user_prompt_with_history": "z"}],
        tokenizer, model, complement_fn=lambda a: "C"
    )
    assert results == []


def test_process_scenarios_template_error(dummy_env):
    tokenizer, model = dummy_env
    results = cu.process_scenarios_with_questions(
        ["Scenario"], ["Q1"], [{}],
        "Dummy", "Dummy", 0.7,
        [{"user_prompt": "This {bad_key} is missing"},
         {"revised_user_prompt": "Revised {complement}"},
         {"user_prompt_with_history": "History {revised_scenario}"}],
        tokenizer, model, complement_fn=lambda a: "C"
    )
    # Expect graceful skip
    assert results == []


def test_process_scenarios_complement_function(dummy_env):
    tokenizer, model = dummy_env
    scenarios = ["S1"]
    questions = ["Q1"]
    metadata = [{}]
    templates = [
        {"user_prompt": "Prompt {index}"},
        {"revised_user_prompt": "Revised {complement}"},
        {"user_prompt_with_history": "History {revised_scenario}"}
    ]

    results = cu.process_scenarios_with_questions(
        scenarios, questions, metadata,
        "Dummy", "Dummy", 0.7,
        templates, tokenizer, model,
        complement_fn=lambda a: "SPECIAL",
    )
    assert results[0]["Target"] == "SPECIAL"


def test_process_scenarios_mismatched_lengths(dummy_env):
    tokenizer, model = dummy_env
    with pytest.raises(ValueError):
        cu.process_scenarios_with_questions(
            ["s1"], ["q1", "q2"], [{}],
            "Dummy", "Dummy", 0.5,
            [{"user_prompt": "{index}"},
             {"revised_user_prompt": "{complement}"},
             {"user_prompt_with_history": "{revised_scenario}"}],
            tokenizer, model
        )
