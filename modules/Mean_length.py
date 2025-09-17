import numpy as np

def calculate_length_statistics(classification_results):
    categories = [
        ("With History", "match_cases"),
        ("Without History", "match_cases"),
        ("With History", "non_match_cases"),
        ("Without History", "non_match_cases"),
    ]

    def length_statistics(case_list):
        lengths = [len(item.get("SCEs", "").split()) for item in case_list if item.get("SCEs")]
        if not lengths:
            return {"mean": 0.0, "std": 0.0, "min": 0, "max": 0, "median": 0.0}

        arr = np.array(lengths)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": int(np.min(arr)),
            "max": int(np.max(arr)),
            "median": float(np.median(arr)),
        }

    results = [length_statistics(classification_results.get(cat[0], {}).get(cat[1], [])) for cat in categories]

    return tuple(results)
