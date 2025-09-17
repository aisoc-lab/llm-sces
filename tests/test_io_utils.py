import os
import json
import pytest
from pathlib import Path

from modules import io_utils


# ----------------------
# Fixtures
# ----------------------

@pytest.fixture
def tmp_json_file(tmp_path):
    """Create a temporary JSON file with known content."""
    file_path = tmp_path / "test.json"
    content = {"key": "value"}
    with open(file_path, "w") as f:
        json.dump(content, f)
    return file_path, content


@pytest.fixture
def tmp_invalid_json_file(tmp_path):
    """Create a temporary file with invalid JSON content."""
    file_path = tmp_path / "invalid.json"
    with open(file_path, "w") as f:
        f.write("{ invalid json }")
    return file_path


# ----------------------
# load_prompt_templates
# ----------------------

def test_load_prompt_templates_valid(tmp_json_file):
    file_path, content = tmp_json_file
    loaded = io_utils.load_prompt_templates(file_path)
    assert loaded == content


def test_load_prompt_templates_invalid(tmp_invalid_json_file):
    with pytest.raises(RuntimeError) as excinfo:
        io_utils.load_prompt_templates(tmp_invalid_json_file)
    assert "Error loading prompt file" in str(excinfo.value)


def test_load_prompt_templates_missing():
    with pytest.raises(RuntimeError) as excinfo:
        io_utils.load_prompt_templates("nonexistent.json")
    assert "Error loading prompt file" in str(excinfo.value)


# ----------------------
# save_results_to_file
# ----------------------

def test_save_results_to_file_creates_file(tmp_path, monkeypatch):
    results = [{"Scenario": "test"}]
    model_name = "my/model"
    dataset_name = "TestDataset"
    temperature = "0.7"
    N = 2
    prompt = "Unconstraint"

    # Run function in tmp_path
    monkeypatch.chdir(tmp_path)
    io_utils.save_results_to_file(results, model_name, temperature, N, dataset_name, prompt)

    folder_name = f"{prompt}_my_model_TestDataset"
    file_path = tmp_path / folder_name / f"results_my_model_TestDataset_temp_{temperature}_n_{N}.json"

    assert file_path.exists()
    with open(file_path) as f:
        saved_data = json.load(f)
    assert saved_data == results


# ----------------------
# load_inputs
# ----------------------

def test_load_inputs_valid(tmp_json_file):
    file_path, content = tmp_json_file
    scenarios, questions, metadata = io_utils.load_inputs(file_path, file_path, file_path)
    assert scenarios == content
    assert questions == content
    assert metadata == content


def test_load_inputs_invalid(tmp_invalid_json_file):
    with pytest.raises(RuntimeError):
        io_utils.load_inputs(tmp_invalid_json_file)


def test_load_inputs_none_files():
    scenarios, questions, metadata = io_utils.load_inputs(None, None, None)
    assert scenarios is None
    assert questions is None
    assert metadata is None


# ----------------------
# infer_file_paths_from_dataset
# ----------------------

def test_infer_file_paths_valid():
    paths = io_utils.infer_file_paths_from_dataset("DiscrimEval", "Unconstraint")
    assert "scenario_file" in paths
    assert "question_file" in paths
    assert "prompt_file" in paths
    assert paths["scenario_file"].endswith("Datasets/DiscrimEval/Scenarios.json")
    assert paths["prompt_file"].endswith("Prompts/Unconstraint/DiscrimEval_prompts.json")


def test_infer_file_paths_invalid_dataset():
    with pytest.raises(ValueError):
        io_utils.infer_file_paths_from_dataset("UnknownDataset", "Unconstraint")


def test_infer_file_paths_invalid_prompt():
    with pytest.raises(ValueError):
        io_utils.infer_file_paths_from_dataset("DiscrimEval", "UnknownPrompt")
