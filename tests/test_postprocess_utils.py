import os
import json
import tempfile
import numpy as np
import pytest
import textstat
import modules.postprocess_utils as pp


# -------------------------
# Basic utilities
# -------------------------
def test_load_json(tmp_path):
    file = tmp_path / "data.json"
    file.write_text(json.dumps({"a": 1}))
    result = pp.load_json(file)
    assert result == {"a": 1}


def test_compute_fk_scores_valid(monkeypatch):
    monkeypatch.setattr(textstat, "flesch_kincaid_grade", lambda x: 5.0)
    cases = [{"SCEs": "This is a test sentence."}]
    scores = pp.compute_fk_scores(cases, label="Test")
    assert scores == [5.0]


def test_compute_fk_scores_empty_and_invalid(monkeypatch):
    monkeypatch.setattr(pp, "flesch_kincaid_grade", lambda x: "not_number")
    cases = [{"SCEs": ""}, {"SCEs": None}, {"SCEs": "valid"}]
    scores = pp.compute_fk_scores(cases, label="X")
    assert isinstance(scores, list)


def test_compute_ci_bounds_normal():
    low, high, margin = pp.compute_ci_bounds(10, 2, 25)
    assert low < 10 < high
    assert margin > 0


def test_compute_ci_bounds_zero_n():
    low, high, margin = pp.compute_ci_bounds(10, 2, 0)
    assert (low, high, margin) == (0.0, 0.0, 0.0)


def test_safe_compute_bootstrap_nd_basic():
    mean, low, high = pp.safe_compute_bootstrap_nd([1, 2, 3], [4, 5, 6], n_bootstrap=100)
    assert 0 <= mean <= 100
    assert low <= mean <= high


def test_safe_compute_bootstrap_nd_empty_lists():
    mean, low, high = pp.safe_compute_bootstrap_nd([], [], n_bootstrap=10)
    assert (mean, low, high) == (0.0, 0.0, 0.0)


def test_paired_permutation_test_reproducible():
    x = [1, 2, 3]
    y = [1, 2, 2]
    diff, p = pp.paired_permutation_test(x, y, n_permutations=100, seed=1)
    assert isinstance(diff, float)
    assert 0 <= p <= 1


def test_extract_edit_distances_various_formats():
    data = [
        {"Normalized Edit Distance Percentage": 10},
        {"Normalized Edit Distance Percentage": {"Normalized Edit Distance Percentage": 20}},
        {"Normalized Edit Distance Percentage": "bad"},
    ]
    dists = pp.extract_edit_distances(data)
    assert dists == [10, 20]


def test_calculate_statistics_with_values():
    values = [1, 2, 3, 4, 5]
    stats = pp.calculate_statistics(values)
    assert "Mean" in stats
    assert isinstance(stats["Outliers"], list)


def test_calculate_statistics_empty():
    assert pp.calculate_statistics([]) == {}


# -------------------------
# File-writing utilities
# -------------------------
def test_save_statistics_to_txt_and_read(tmp_path):
    stats = {
        "Mean": 10.0,
        "Standard Deviation": 2.0,
        "Mean - SD": 8.0,
        "Mean + SD": 12.0,
        "Q1": 9.0,
        "Median": 10.0,
        "Q3": 11.0,
        "Outliers": [20],
    }
    file_path = tmp_path / "stats.txt"
    pp.save_statistics_to_txt(stats, file_path, "DiscrimEval", n=10, extra_note="extra")
    text = file_path.read_text()
    assert "Mean: 10.00" in text
    assert "95% Confidence Interval" in text


def test_save_json_roundtrip(tmp_path):
    file = tmp_path / "data.json"
    data = {"a": 1}
    pp.save_json(file, data)
    reloaded = json.loads(file.read_text())
    assert reloaded == data


def test_compute_gen_val_normal_case():
    gen, val, val_c = pp.compute_gen_val(total=10, short=2, unknown=1, non_match_without=3, non_match_with=2)
    assert isinstance(gen, int)
    assert isinstance(val, int)
    assert isinstance(val_c, int)


def test_compute_gen_val_zero_total():
    assert pp.compute_gen_val(0, 0, 0, 0, 0) == (0, 0, 0)


def test_analyze_fk_match_vs_nonmatch(monkeypatch):
    monkeypatch.setattr(pp, "paired_permutation_test", lambda *a, **k: (1.0, 0.5))
    result = pp.analyze_fk_match_vs_nonmatch([1, 2, 3], [2, 3, 4], "Test")
    assert result["label"] == "Test"
    assert "ci1" in result and "ci2" in result


def test_compute_ci_rounded_normal():
    low, high, se = pp.compute_ci_rounded(50, 100)
    assert 0 <= low <= high <= 100


def test_compute_ci_rounded_zero_n():
    low, high, se = pp.compute_ci_rounded(50, 0)
    assert (low, high, se) == (0.0, 0.0, 0.0)


def test_compute_se_and_ci_for_normalized_diff_valid():
    result = pp.compute_se_and_ci_for_normalized_diff(10, 2, 20, 12, 3, 20)
    norm_diff, se, low, high = result
    assert se >= 0
    assert low < high


def test_compute_se_and_ci_for_normalized_diff_zero_samples():
    result = pp.compute_se_and_ci_for_normalized_diff(10, 2, 0, 12, 3, 20)
    assert result == (0.0, 0.0, 0.0, 0.0)


# -------------------------
# Integration-ish test
# -------------------------
def test_write_summary_creates_file(tmp_path):
    path = tmp_path
    classification_results = {
        "With History": {"match_cases": [{"SCEs": "a b"}], "non_match_cases": [{"SCEs": "c d"}]},
        "Without History": {"match_cases": [{"SCEs": "a"}], "non_match_cases": [{"SCEs": "b"}]},
    }
    stats = {"mean": 1.0, "std": 0.5, "min": 1, "max": 2, "median": 1.0}

    pp.write_summary(
        path=path,
        classification_results=classification_results,
        total=4,
        unknown_count=0,
        non_empty_spans_count=2,
        short_case_count=0,
        stats_match_with_history=stats,
        stats_match_without_history=stats,
        stats_non_match_with_history=stats,
        stats_non_match_without_history=stats,
        gen=80,
        val=50,
        val_c=25,
    )
    text = (path / "summary.txt").read_text()
    assert "Total cases" in text
