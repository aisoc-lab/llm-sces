import Levenshtein
import torch
import numpy as np
from sklearn.cluster import KMeans  
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler

def calculate_edit_distance_metrics(original, revised):
    """Calculate edit distance metrics between two strings."""
    try:
        def extract_text(value):
            if isinstance(value, dict):
                return value.get("filled_template") or value.get("scenario") or str(value)
            return value

        original = extract_text(original)
        revised = extract_text(revised)

        if not isinstance(original, str) or not isinstance(revised, str):
            raise ValueError("Inputs must be strings")

        len_orig, len_rev = len(original), len(revised)
        longer_length = max(len_orig, len_rev)

        raw_distance = Levenshtein.distance(original, revised)
        normalized_distance_percentage = (raw_distance / longer_length * 100) if longer_length else 0.0

        # Character overlap using set for efficiency
        if len_orig > 0:
            overlap = len(set(original) & set(revised)) / len_orig * 100
        else:
            overlap = 0.0

        levenshtein_ratio = Levenshtein.ratio(original, revised)

        return {
            "Raw Edit Distance": raw_distance,
            "Normalized Edit Distance Percentage": round(normalized_distance_percentage, 2),
            "Character Overlap Percentage": round(overlap, 2),
            "Levenshtein Ratio": round(levenshtein_ratio, 2)
        }

    except Exception as e:
        print(f"Error in calculate_edit_distance_metrics: {e}")
        return {}

def run_kmeans_and_variances(data, k, metric="euclidean", normalize=False):
    """Run KMeans with optional normalization and cosine/euclidean metric."""
    X = data
    if normalize:
        X = StandardScaler().fit_transform(X)

    if metric == "cosine":
        # sklearn KMeans does not support cosine directly, so we project to unit sphere
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X)

    # within-cluster variance
    centroids = np.array([X[kmeans.labels_ == i].mean(axis=0) for i in range(k)])
    _, distances = pairwise_distances_argmin_min(X, centroids, metric=metric)
    within_var = np.mean(distances ** 2)

    # between-cluster variance
    between_var = np.var(centroids, axis=0).mean()

    return kmeans, within_var, between_var

def compute_cluster_variances(data, labels):
    k = len(set(labels))
    centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
    _, distances = pairwise_distances_argmin_min(data, centroids)
    within_var = np.mean(distances ** 2)
    between_var = np.var(centroids, axis=0).mean()
    return within_var, between_var

# Cosine drift metric
class CosineDriftMetric:
    def __init__(self, dim: int = 0):
        self.dim = dim

    def compute(self, x: torch.Tensor):
        dim = self.dim % x.dim()
        if x.shape[dim] < 2:
            raise ValueError("Need at least 2 elements to compute drift")
        x1 = x.narrow(dim, 0, x.shape[dim] - 1)
        x2 = x.narrow(dim, 1, x.shape[dim] - 1)
        x1_perm = x1.transpose(dim, -2)
        x2_perm = x2.transpose(dim, -2)
        drift = torch.nn.functional.cosine_similarity(x1_perm, x2_perm, dim=-1)
        return 1.0 - drift