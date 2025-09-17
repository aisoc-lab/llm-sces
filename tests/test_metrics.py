import pytest
import torch
import numpy as np
from modules.metrics import (
    calculate_edit_distance_metrics,
    run_kmeans_and_variances,
    compute_cluster_variances,
    CosineDriftMetric
)

# -----------------------------
# Edit Distance Metrics
# -----------------------------
def test_edit_distance_identical():
    result = calculate_edit_distance_metrics("hello", "hello")
    assert result["Raw Edit Distance"] == 0
    assert result["Normalized Edit Distance Percentage"] == 0.0
    assert result["Levenshtein Ratio"] == 1.0

def test_edit_distance_completely_different():
    result = calculate_edit_distance_metrics("abc", "xyz")
    assert result["Raw Edit Distance"] == 3
    assert result["Normalized Edit Distance Percentage"] == 100.0
    assert result["Levenshtein Ratio"] < 1.0
    assert result["Character Overlap Percentage"] == 0.0

def test_edit_distance_empty_strings():
    result = calculate_edit_distance_metrics("", "")
    assert result["Raw Edit Distance"] == 0
    assert result["Normalized Edit Distance Percentage"] == 0.0

def test_edit_distance_with_dict_input():
    result = calculate_edit_distance_metrics({"scenario": "abc"}, {"filled_template": "abcd"})
    assert "Raw Edit Distance" in result
    assert result["Raw Edit Distance"] == 1

def test_edit_distance_invalid_input():
    result = calculate_edit_distance_metrics(123, ["not a string"])
    assert result == {}  # should fail gracefully

# -----------------------------
# KMeans & Variance Metrics
# -----------------------------
def test_run_kmeans_and_variances_euclidean():
    data = np.array([[0, 0], [1, 1], [10, 10], [11, 11]])
    kmeans, within, between = run_kmeans_and_variances(data, k=2, metric="euclidean")
    assert hasattr(kmeans, "labels_")
    assert within >= 0
    assert between >= 0

def test_run_kmeans_and_variances_cosine():
    data = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    kmeans, within, between = run_kmeans_and_variances(data, k=2, metric="cosine")
    assert hasattr(kmeans, "labels_")
    assert within >= 0
    assert between >= 0

def test_run_kmeans_and_variances_with_normalization():
    data = np.array([[1, 2], [1, 2], [10, 20], [11, 19]])
    kmeans, within, between = run_kmeans_and_variances(data, k=2, normalize=True)
    assert isinstance(within, float)
    assert isinstance(between, float)

def test_compute_cluster_variances():
    data = np.array([[0, 0], [0, 1], [10, 10], [10, 11]])
    labels = np.array([0, 0, 1, 1])
    within, between = compute_cluster_variances(data, labels)
    assert within >= 0
    assert between >= 0

# -----------------------------
# Cosine Drift Metric
# -----------------------------
def test_cosine_drift_metric_basic():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    metric = CosineDriftMetric(dim=0)
    drift = metric.compute(x)
    assert isinstance(drift, torch.Tensor)
    assert drift.shape[0] == 1  # only one pair compared

def test_cosine_drift_metric_multi_pairs():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    metric = CosineDriftMetric(dim=0)
    drift = metric.compute(x)
    assert drift.numel() == 2  # two comparisons

def test_cosine_drift_metric_invalid():
    x = torch.tensor([[1.0, 0.0]])  # only one element
    metric = CosineDriftMetric(dim=0)
    with pytest.raises(ValueError):
        metric.compute(x)
