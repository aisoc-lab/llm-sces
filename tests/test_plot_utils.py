import os
import json
import tempfile
import pytest
import matplotlib.pyplot as plt
import logging
from modules import plot_utils


# ----------------------------
# Fixtures
# ----------------------------
@pytest.fixture(autouse=True)
def cleanup_tmp_files():
    """Ensure temporary files are removed after each test."""
    tmp_files = []
    yield tmp_files
    for f in tmp_files:
        if os.path.exists(f):
            os.remove(f)


# ----------------------------
# load_json
# ----------------------------
def test_load_json_valid(tmp_path):
    file_path = tmp_path / "data.json"
    data = [{"SCEs": "hello world"}]
    file_path.write_text(json.dumps(data))

    result = plot_utils.load_json(str(file_path))
    assert isinstance(result, list)
    assert result == data


def test_load_json_invalid_json(tmp_path, caplog):
    file_path = tmp_path / "bad.json"
    file_path.write_text("{ invalid json }")

    with caplog.at_level(logging.ERROR):
        result = plot_utils.load_json(str(file_path))

    assert result == []
    assert "Failed to load JSON" in caplog.text


def test_load_json_missing_file(caplog):
    with caplog.at_level(logging.ERROR):
        result = plot_utils.load_json("nonexistent.json")

    assert result == []
    assert "Failed to load JSON" in caplog.text


# ----------------------------
# extract_scenario_lengths
# ----------------------------
def test_extract_scenario_lengths_basic():
    data = [
        {"SCEs": "one two three"},
        {"SCEs": "a b"},
        {"no_sces": "ignored"},
        "not a dict",
    ]
    lengths = plot_utils.extract_scenario_lengths(data)
    assert lengths == [3, 2]


def test_extract_scenario_lengths_empty():
    assert plot_utils.extract_scenario_lengths([]) == []


# ----------------------------
# plot_histogram
# ----------------------------
def test_plot_histogram_creates_file(tmp_path, cleanup_tmp_files):
    valid = [{"SCEs": "word " * 20}, {"SCEs": "word " * 30}]
    invalid = [{"SCEs": "word " * 15}]
    out_file = tmp_path / "hist.pdf"
    cleanup_tmp_files.append(str(out_file))

    plot_utils.plot_histogram(valid, invalid, "Test Plot", str(out_file))

    assert out_file.exists()
    assert out_file.stat().st_size > 0  # file should not be empty


def test_plot_histogram_only_valid(tmp_path, cleanup_tmp_files):
    valid = [{"SCEs": "word " * 40}]
    invalid = []
    out_file = tmp_path / "valid_only.pdf"
    cleanup_tmp_files.append(str(out_file))

    plot_utils.plot_histogram(valid, invalid, "Valid Only", str(out_file))
    assert out_file.exists()


def test_plot_histogram_only_invalid(tmp_path, cleanup_tmp_files):
    valid = []
    invalid = [{"SCEs": "word " * 50}]
    out_file = tmp_path / "invalid_only.pdf"
    cleanup_tmp_files.append(str(out_file))

    plot_utils.plot_histogram(valid, invalid, "Invalid Only", str(out_file))
    assert out_file.exists()


def test_plot_histogram_no_data(tmp_path, caplog):
    valid, invalid = [], []
    out_file = tmp_path / "empty.pdf"

    with caplog.at_level(logging.WARNING):
        plot_utils.plot_histogram(valid, invalid, "Empty Case", str(out_file))

    # No file should be created
    assert not out_file.exists()
    assert "Skipping Empty Case - No data available." in caplog.text
