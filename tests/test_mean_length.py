import pytest
import numpy as np
from modules.Mean_length import calculate_length_statistics


def test_all_empty():
    classification_results = {
        "With History": {"match_cases": [], "non_match_cases": []},
        "Without History": {"match_cases": [], "non_match_cases": []},
    }
    stats = calculate_length_statistics(classification_results)

    for s in stats:
        assert s == {"mean": 0.0, "std": 0.0, "min": 0, "max": 0, "median": 0.0}


def test_single_case():
    classification_results = {
        "With History": {"match_cases": [{"SCEs": "a b c"}], "non_match_cases": []},
        "Without History": {"match_cases": [], "non_match_cases": []},
    }
    stats = calculate_length_statistics(classification_results)

    # With History, match_cases = "a b c" → length 3
    assert stats[0]["mean"] == 3.0
    assert stats[0]["min"] == 3
    assert stats[0]["max"] == 3
    assert stats[0]["median"] == 3.0

    # Other categories should be all zero
    for s in stats[1:]:
        assert all(v == 0 or v == 0.0 for v in s.values())


def test_multiple_lengths():
    classification_results = {
        "With History": {
            "match_cases": [
                {"SCEs": "a b"},        # length 2
                {"SCEs": "c d e"},      # length 3
                {"SCEs": "f"}           # length 1
            ],
            "non_match_cases": []
        },
        "Without History": {"match_cases": [], "non_match_cases": []},
    }
    stats = calculate_length_statistics(classification_results)

    arr = np.array([2, 3, 1])
    expected = {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
        "median": float(np.median(arr)),
    }
    assert stats[0] == expected


def test_mixed_cases_and_missing_sces():
    classification_results = {
        "With History": {
            "match_cases": [{"SCEs": "one two"}, {}],  # second item missing "SCEs"
            "non_match_cases": [{"SCEs": ""}],         # empty string ignored
        },
        "Without History": {
            "match_cases": [{"SCEs": "alpha beta gamma delta"}],  # length 4
            "non_match_cases": [{"SCEs": "x y"}],                 # length 2
        },
    }
    stats = calculate_length_statistics(classification_results)

    # With History match_cases → only "one two" = length 2
    assert stats[0]["mean"] == 2.0
    # Without History match_cases → "alpha beta gamma delta" = length 4
    assert stats[1]["mean"] == 4.0
    # With History non_match_cases → ignored (empty string)
    assert stats[2]["mean"] == 0.0
    # Without History non_match_cases → "x y" = length 2
    assert stats[3]["mean"] == 2.0


def test_return_type_structure():
    classification_results = {
        "With History": {"match_cases": [{"SCEs": "a b"}], "non_match_cases": [{"SCEs": "c"}]},
        "Without History": {"match_cases": [{"SCEs": "d e f"}], "non_match_cases": [{"SCEs": "g h"}]},
    }
    stats = calculate_length_statistics(classification_results)

    assert isinstance(stats, tuple)
    assert len(stats) == 4
    for s in stats:
        assert set(s.keys()) == {"mean", "std", "min", "max", "median"}
        assert isinstance(s["mean"], float)
        assert isinstance(s["std"], float)
        assert isinstance(s["min"], int)
        assert isinstance(s["max"], int)
        assert isinstance(s["median"], float)
