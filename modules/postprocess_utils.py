import os
import json
import numpy as np
from tqdm import tqdm
from scipy.stats import levene, ttest_ind
from modules.dataset_config import get_dataset_config
from modules.extract_decision import get_extract_decision_function
from modules.case_filtering import is_short_case, count_words
from modules.case_classifier import classify_case
from modules.Mean_length import calculate_length_statistics
from modules.plot_utils import plot_histogram
from textstat import flesch_kincaid_grade

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def compute_fk_scores(cases, label=""):
    from textstat import flesch_kincaid_grade
    import logging

    scores = []
    for i, case in enumerate(cases):
        sce = case.get("SCEs", "")
        if not isinstance(sce, str) or not sce.strip():
            logging.warning(f"[{label}] Case {i}: Empty or non-string SCE — skipped")
            continue

        try:
            score = flesch_kincaid_grade(sce)
            if isinstance(score, (int, float)):
                scores.append(score)
            else:
                logging.warning(f"[{label}] Case {i}: Invalid FK score = {score} — skipped")
        except Exception as e:
            logging.warning(f"[{label}] Case {i}: Exception during FK scoring: {e}")
    
    print(f"FK Scores computed for [{label}]: {len(scores)} of {len(cases)} cases")
    return scores

# Compute CI overlap between with and without history groups
def compute_ci_bounds(mean, std, n):
    if n == 0:
        return 0.0, 0.0, 0.0
    margin = 1.96 * (std / np.sqrt(n))
    lower = mean - margin
    upper = mean + margin
    return lower, upper, margin

def safe_compute_bootstrap_nd(match_lengths, nonmatch_lengths, n_bootstrap=10000):
    if not match_lengths and not nonmatch_lengths:
        print("Both match and non-match lists are empty — cannot compute bootstrap.")
        return 0.0, 0.0, 0.0

    # If one list is empty, use its mean as 0
    if not match_lengths:
        print("Match list is empty — using 0 as its mean.")
        match_lengths = [0] * len(nonmatch_lengths)
    if not nonmatch_lengths:
        print("Non-match list is empty — using 0 as its mean.")
        nonmatch_lengths = [0] * len(match_lengths)

    nds = []
    for _ in range(n_bootstrap):
        sample_match = np.random.choice(match_lengths, size=len(match_lengths), replace=True)
        sample_nonmatch = np.random.choice(nonmatch_lengths, size=len(nonmatch_lengths), replace=True)

        mean_match = np.mean(sample_match)
        mean_nonmatch = np.mean(sample_nonmatch)

        max_mean = max(mean_match, mean_nonmatch)
        if max_mean > 0:
            nd = abs(mean_match - mean_nonmatch) / max_mean * 100
            nds.append(nd)
        else:
            nds.append(0.0)  # if both are 0

    if not nds:
        print("ND list is empty after bootstrapping.")
        return 0.0, 0.0, 0.0

    return np.mean(nds), np.percentile(nds, 2.5), np.percentile(nds, 97.5)

def paired_permutation_test(x, y, n_permutations=10000, seed=42):
    """
    Performs a paired permutation test between two paired samples.
    """
    np.random.seed(seed)
    diffs = np.array(x) - np.array(y)
    observed_mean_diff = np.mean(diffs)

    count = 0
    for _ in range(n_permutations):
        signs = np.random.choice([1, -1], size=len(diffs))
        permuted_diffs = diffs * signs
        permuted_mean = np.mean(permuted_diffs)
        if abs(permuted_mean) >= abs(observed_mean_diff):
            count += 1

    p_value = count / n_permutations
    return observed_mean_diff, p_value


# Extract "Edit Distance Percentage" values
def extract_edit_distances(data):
    distances = []
    for item in data:
        metric = item.get("Normalized Edit Distance Percentage")
        if isinstance(metric, dict):
            value = metric.get("Normalized Edit Distance Percentage")
            if isinstance(value, (int, float)):
                distances.append(value)
        elif isinstance(metric, (int, float)):
            distances.append(metric)
    return distances

def calculate_statistics(values):
    if not values:
        return {}

    arr = np.array(values)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    outliers = arr[(arr < q1 - 1.5 * iqr) | (arr > q3 + 1.5 * iqr)].tolist()

    return {
        "Mean": np.mean(arr),
        "Standard Deviation": np.std(arr, ddof=1),
        "Mean - SD": np.mean(arr) - np.std(arr, ddof=1),
        "Mean + SD": np.mean(arr) + np.std(arr, ddof=1),
        "Q1": q1,
        "Median": np.median(arr),
        "Q3": q3,
        "Outliers": outliers
    }



def save_statistics_to_txt(stats, file_path, dataset_name, n=None, extra_note=None, ci_comparison_note=None):

    with open(file_path, "w") as file:
        file.write("Statistical Metrics:\n")

        mean = stats.get('Mean', None)
        std_dev = stats.get('Standard Deviation', None)
        mean_minus_sd = stats.get('Mean - SD', None)
        mean_plus_sd = stats.get('Mean + SD', None)
        q1 = stats.get('Q1', None)
        median = stats.get('Median', None)
        q3 = stats.get('Q3', None)
        outliers = stats.get('Outliers', [])

        file.write(f"Mean: {mean:.2f}\n" if mean is not None else "Mean: N/A\n")
        file.write(f"Standard Deviation: {std_dev:.2f}\n" if std_dev is not None else "Standard Deviation: N/A\n")
        file.write(f"Mean - SD: {mean_minus_sd:.2f}\n" if mean_minus_sd is not None else "Mean - SD: N/A\n")
        file.write(f"Mean + SD: {mean_plus_sd:.2f}\n" if mean_plus_sd is not None else "Mean + SD: N/A\n")
        file.write(f"Q1: {q1:.2f}\n" if q1 is not None else "Q1: N/A\n")
        file.write(f"Median: {median:.2f}\n" if median is not None else "Median: N/A\n")
        file.write(f"Q3: {q3:.2f}\n" if q3 is not None else "Q3: N/A\n")
        file.write(f"Outliers: {outliers}\n")

        # Dataset-specific sample size for CI
        dataset_sample_sizes = {
            "DiscrimEval": 70,
            "Folktexts": 500,
            "GSM8K": 250,
            "SST2": 500,
            "Twitter": 750,
            "NLI": 750,
        }

        if n is None:
            n = dataset_sample_sizes.get(dataset_name, len(outliers))

        # Confidence Interval (95%): Mean ± 1.96 * (SD / sqrt(n))
        if mean is not None and std_dev is not None and n > 0:
            margin = 1.96 * (std_dev / np.sqrt(n))
            file.write(f"95% Confidence Interval: [{mean - margin:.2f}, {mean + margin:.2f}]\n")
        else:
            file.write("95% Confidence Interval: N/A\n")

        if extra_note:
            file.write(f"\nNote: {extra_note}\n")
        if ci_comparison_note:
            file.write(f"\nCI Comparison Note:\n{ci_comparison_note}\n")
        print(f"Statistics written to: {file_path}")


def compute_gen_val(total, short, unknown, non_match_without, non_match_with):
    if total == 0:
        return 0, 0, 0

    valid_count = total - short - unknown
    gen_ratio = valid_count / total if total > 0 else 0  
    val = (non_match_without / valid_count) * 100 if valid_count > 0 else 0
    val_c = (non_match_with / valid_count) * 100 if valid_count > 0 else 0

    return round(gen_ratio * 100), round(val), round(val_c)

def analyze_fk_match_vs_nonmatch(scores_match, scores_nonmatch, label):
    stats_match = calculate_statistics(scores_match)
    stats_nonmatch = calculate_statistics(scores_nonmatch)

    # Safe defaults
    mean1 = stats_match.get("Mean", 0.0)
    std1 = stats_match.get("Standard Deviation", 0.0)
    n1 = len(scores_match)

    mean2 = stats_nonmatch.get("Mean", 0.0)
    std2 = stats_nonmatch.get("Standard Deviation", 0.0)
    n2 = len(scores_nonmatch)

    ci1_lower, ci1_upper, _ = compute_ci_bounds(mean1, std1, n1)
    ci2_lower, ci2_upper, _ = compute_ci_bounds(mean2, std2, n2)
    ci_overlap = not (ci1_upper < ci2_lower or ci2_upper < ci1_lower)

    # Permutation test: only if both have data
    if n1 > 0 and n2 > 0:
        n = min(n1, n2)
        perm_diff, p_val = paired_permutation_test(scores_match[:n], scores_nonmatch[:n])
    else:
        perm_diff, p_val = 0.0, 1.0

    return {
        "label": label,
        "match_mean": mean1,
        "nonmatch_mean": mean2,
        "ci1": (ci1_lower, ci1_upper),
        "ci2": (ci2_lower, ci2_upper),
        "overlap": ci_overlap,
        "perm_diff": perm_diff,
        "p_val": p_val
    }


def process_results_with_statistics(json_file, dataset_name):
    # Extract clean base name (remove 'results_' and '.json')
    base_filename = os.path.basename(json_file)
    model_name = base_filename
    if model_name.startswith("results_"):
        model_name = model_name[len("results_"):]
    if model_name.endswith(".json"):
        model_name = model_name[:-len(".json")]

    # Construct output folder
    output_folder = os.path.join(os.path.dirname(json_file), f"{model_name}_Post_Processing")

    os.makedirs(output_folder, exist_ok=True)

    all_results = load_json(json_file)
    print(f"Loaded {len(all_results)} results from {json_file}")

    if not all_results:
        print(f"WARNING: No results to process in {json_file}")
        return

    print(f"DEBUG: Sample keys from first result: {list(all_results[0].keys())}")

    classification_results = {
        "With History": {"match_cases": [], "non_match_cases": [], "non_response_cases": [], "other_cases": [], "unclassified_cases": []},
        "Without History": {"match_cases": [], "non_match_cases": [], "non_response_cases": [], "other_cases": [], "unclassified_cases": []},
    }

    unknown_cases = []
    short_cases = []
    empty_spans_cases = []   # List to store cases where "SPANS" == "Rationales" is empty
    non_empty_spans_count = 0  # Counter for non-empty SPANS
    
    # === Safe defaults for optional summary fields ===
    fk_stats_with = fk_stats_with if 'fk_stats_with' in locals() else {}
    fk_stats_without = fk_stats_without if 'fk_stats_without' in locals() else {}
    fk_ci1_lower, fk_ci1_upper = (fk_ci1_lower, fk_ci1_upper) if 'fk_ci1_lower' in locals() else (0.0, 0.0)
    fk_ci2_lower, fk_ci2_upper = (fk_ci2_lower, fk_ci2_upper) if 'fk_ci2_lower' in locals() else (0.0, 0.0)
    fk_overlap_note = fk_overlap_note if 'fk_overlap_note' in locals() else "N/A"
    perm_note_fk = perm_note_fk if 'perm_note_fk' in locals() else "No FK permutation test performed."
    perm_note_nonmatch = perm_note_nonmatch if 'perm_note_nonmatch' in locals() else "No non-match permutation test performed."
    perm_note_validity = perm_note_validity if 'perm_note_validity' in locals() else "No validity test performed."
    perm_note_ed = perm_note_ed if 'perm_note_ed' in locals() else "No edit distance permutation test performed."
    fk_result_with = fk_result_with if 'fk_result_with' in locals() else {}
    fk_result_without = fk_result_without if 'fk_result_without' in locals() else {}


    scenario = all_results[0].get("Scenario", {})  # Use first result as reference

    _, _, decode_word, _, _ = get_dataset_config(dataset_name, scenario, questions=None, index=0)
    print(f"DEBUG: decode_word used for filtering: '{decode_word}'")

    extract_fn = get_extract_decision_function(dataset_name)
    extract_decision = get_extract_decision_function(dataset_name)


    for result in tqdm(all_results, desc=f"Processing results for {dataset_name}", total=len(all_results)):
        if str(result.get("Target", "")).strip().lower() == "unknown":
            unknown_cases.append(result)
            continue

        if is_short_case(dataset_name, result, decode_word):
            print("Filtered as short case.")
            short_cases.append(result)
            continue
        else:
            print("Kept for classification.")
        
        spans = result.get("Rational")

        if isinstance(spans, str):
            spans = spans.strip()
            if spans in [
                "", 
                "'.", 
                "'. \n\nRA", 
                "'.\nRATIONALES: (Yes / No)",
                "[yes or no]",
                "- \"the account holder has a",
                "1. 't",
                "Below is a decision-making scenario followed by a question about the scenario. Answer the question with 'Yes' or 'No'. Do not include any additional words in your answer. Your answer should start with '"
            ]:
                empty_spans_cases.append(result)
                continue
            else:
                non_empty_spans_count += 1

        # If spans is missing or not a string, just skip the check

        extract_fn = get_extract_decision_function(dataset_name)
        extract_decision = extract_fn  # Use same function unless special handling needed

        classify_case(result, dataset_name, classification_results, extract_fn, extract_decision)
        # Extract distances
        with_history = classification_results["With History"]["non_match_cases"]
        without_history = classification_results["Without History"]["non_match_cases"]
        
        # === Flesch-Kincaid Readability Analysis ===
        fk_scores = {
            "With History": {
                "match": compute_fk_scores(classification_results["With History"]["match_cases"], label="With-Match"),
                "non_match": compute_fk_scores(classification_results["With History"]["non_match_cases"], label="With-NonMatch"),
            },
            "Without History": {
                "match": compute_fk_scores(classification_results["Without History"]["match_cases"], label="Without-Match"),
                "non_match": compute_fk_scores(classification_results["Without History"]["non_match_cases"], label="Without-NonMatch"),
            }
        }

        fk_with = fk_scores["With History"]["match"] + fk_scores["With History"]["non_match"]
        fk_without = fk_scores["Without History"]["match"] + fk_scores["Without History"]["non_match"]

        fk_stats_with = calculate_statistics(fk_with)
        fk_stats_without = calculate_statistics(fk_without)


        # Permutation test on FK scores
        n_fk = min(len(fk_with), len(fk_without))
        fk_with_paired = fk_with[:n_fk]
        fk_without_paired = fk_without[:n_fk]
        fk_diff, p_val_fk = paired_permutation_test(fk_with_paired, fk_without_paired)

        perm_note_fk = (
            f"Paired permutation test (Flesch-Kincaid Readability):\n"
            f"  Mean difference = {fk_diff:.2f} → p-value = {p_val_fk:.4f} → "
            f"{'Statistically significant' if p_val_fk < 0.05 else 'Not significant'}"
        )

        # CI overlap analysis
        mean_fk_with = fk_stats_with.get("Mean", float('nan'))
        std_fk_with = fk_stats_with.get("Standard Deviation", 0)
        n1_fk = len(fk_with)

        mean_fk_without = fk_stats_without.get("Mean", float('nan'))
        std_fk_without = fk_stats_without.get("Standard Deviation", 0)
        n2_fk = len(fk_without)

        fk_ci1_lower, fk_ci1_upper, _ = compute_ci_bounds(mean_fk_with, std_fk_with, n1_fk)
        fk_ci2_lower, fk_ci2_upper, _ = compute_ci_bounds(mean_fk_without, std_fk_without, n2_fk)

        fk_ci_overlap = not (fk_ci1_upper < fk_ci2_lower or fk_ci2_upper < fk_ci1_lower)
        fk_overlap_note = (
            f"Flesch-Kincaid CI With History: [{fk_ci1_lower:.2f}, {fk_ci1_upper:.2f}]\n"
            f"Flesch-Kincaid CI Without History: [{fk_ci2_lower:.2f}, {fk_ci2_upper:.2f}]\n"
            f"CI Overlap: {'YES → not significant' if fk_ci_overlap else 'NO → significant'}"
        )

        # -------------------------------
        # VALIDITY permutation test
        # -------------------------------

        # Pair match + non-match cases
        with_cases = classification_results["With History"]["match_cases"] + classification_results["With History"]["non_match_cases"]
        without_cases = classification_results["Without History"]["match_cases"] + classification_results["Without History"]["non_match_cases"]

        n_valid_pairs = min(len(with_cases), len(without_cases))
        paired_with_cases = with_cases[:n_valid_pairs]
        paired_without_cases = without_cases[:n_valid_pairs]

        val_labels_with = [1 if case in classification_results["With History"]["non_match_cases"] else 0
                        for case in paired_with_cases]
        val_labels_without = [1 if case in classification_results["Without History"]["non_match_cases"] else 0
                            for case in paired_without_cases]

        val_diff, p_val_validity = paired_permutation_test(val_labels_with, val_labels_without)
        perm_note_validity = (
            f"Paired permutation test (Validity rate):\n"
            f"  Mean difference = {val_diff:.2f} → p-value = {p_val_validity:.4f} → "
            f"{'Statistically significant' if p_val_validity < 0.05 else 'Not significant'}"
        )

        
        # --- Paired permutation test on non-match cases ---
        # Extract SCE lengths for non-match cases
        lengths_with = [len(item.get("SCEs", "").split()) for item in with_history]
        lengths_without = [len(item.get("SCEs", "").split()) for item in without_history]

        # Ensure same length for pairing by truncating to the smallest group
        n = min(len(lengths_with), len(lengths_without))
        paired_with = lengths_with[:n]
        paired_without = lengths_without[:n]
        
        mean_diff_nonmatch, p_value_nonmatch = paired_permutation_test(paired_with, paired_without)

        # Compose result note
        perm_note_nonmatch = (
            f"Paired permutation test (non-match SCE lengths):\n"
            f"  Mean difference = {mean_diff_nonmatch:.2f} words\n"
            f"  p-value = {p_value_nonmatch:.4f} → "
            f"{'Statistically significant' if p_value_nonmatch < 0.05 else 'Not significant'}"
        )

        distances_with = extract_edit_distances(with_history)
        distances_without = extract_edit_distances(without_history)

        # -------------------------------
        # EDIT DISTANCE permutation test
        # -------------------------------
        n_ed = min(len(distances_with), len(distances_without))
        paired_ed_with = distances_with[:n_ed]
        paired_ed_without = distances_without[:n_ed]

        ed_diff, p_val_ed = paired_permutation_test(paired_ed_with, paired_ed_without)
        perm_note_ed = (
            f"Paired permutation test (Edit Distance):\n"
            f"  Mean difference = {ed_diff:.2f} → p-value = {p_val_ed:.4f} → "
            f"{'Statistically significant' if p_val_ed < 0.05 else 'Not significant'}"
        )

        # Stats per group
        stats_with = calculate_statistics(distances_with)
        stats_without = calculate_statistics(distances_without)

        # Step 1: Check for equality of variances
        levene_stat, levene_p = levene(distances_with, distances_without)

        # Step 2: Use appropriate t-test
        equal_variance = levene_p >= 0.05  # p ≥ 0.05 -> equal variances -> use Student's t-test
        t_stat, p_val = ttest_ind(distances_with, distances_without, equal_var=equal_variance)

        # Compose the note
        test_used = "Student's t-test" if equal_variance else "Welch's t-test"
        mean_with = stats_with.get("Mean", float('nan'))
        mean_without = stats_without.get("Mean", float('nan'))

        note = (
            f"{test_used} results:\n"
            f"  - Levene's test statistic = {levene_stat:.4f}\n"
            f"  - Levene's test p-value = {levene_p:.4f} → Variances are {'equal' if equal_variance else 'not equal'}\n"
            f"  - t-statistic = {t_stat:.4f}\n"
            f"  - p-value = {p_val:.4f} → Means are statistically {'different' if p_val < 0.05 else 'not different'}\n"
            f"  - Mean (With History) = {mean_with:.2f}\n"
            f"  - Mean (Without History) = {mean_without:.2f}\n"
        )

        if not fk_scores["With History"]["match"] and not fk_scores["With History"]["non_match"]:
            print("Skipping FK analysis for 'With History': insufficient match/non-match data.")
            fk_result_with = {
                "label": "With History",
                "match_mean": float('nan'),
                "nonmatch_mean": float('nan'),
                "ci1": (float('nan'), float('nan')),
                "ci2": (float('nan'), float('nan')),
                "overlap": None,
                "perm_diff": float('nan'),
                "p_val": float('nan')
            }
        else:
            fk_result_with = analyze_fk_match_vs_nonmatch(
                fk_scores["With History"]["match"],
                fk_scores["With History"]["non_match"],
                "With History"
            )


        if not fk_scores["Without History"]["match"] and not fk_scores["Without History"]["non_match"]:
            print("Skipping FK analysis for 'Without History': insufficient match/non-match data.")
            fk_result_without = {
                "label": "Without History",
                "match_mean": float('nan'),
                "nonmatch_mean": float('nan'),
                "ci1": (float('nan'), float('nan')),
                "ci2": (float('nan'), float('nan')),
                "overlap": None,
                "perm_diff": float('nan'),
                "p_val": float('nan')
            }
        else:
            fk_result_without = analyze_fk_match_vs_nonmatch(
                fk_scores["Without History"]["match"],
                fk_scores["Without History"]["non_match"],
                "Without History"
            )

        if not stats_with or not stats_without:
            print("WARNING: Skipping CI comparison due to insufficient data.")
            ci_comparison_note = "CI Comparison not available: insufficient non-match samples."
        else:
            mean1, std1 = stats_with["Mean"], stats_with["Standard Deviation"]
            mean2, std2 = stats_without["Mean"], stats_without["Standard Deviation"]

            n1 = len(distances_with)
            n2 = len(distances_without)

            ci1_lower, ci1_upper, margin1 = compute_ci_bounds(mean1, std1, n1)
            ci2_lower, ci2_upper, margin2 = compute_ci_bounds(mean2, std2, n2)


            ci_overlap = not (ci1_upper < ci2_lower or ci2_upper < ci1_lower)

            ci_comparison_note = (
                f"With History CI: [{ci1_lower:.2f}, {ci1_upper:.2f}] ± {margin1:.2f}\n"
                f"Without History CI: [{ci2_lower:.2f}, {ci2_upper:.2f}] ± {margin2:.2f}\n"
                f"CI Overlap: {'YES → not statistically significant' if ci_overlap else 'NO → statistically significant'}"
            )


        # Save both sets with shared comparison note
        save_statistics_to_txt(
            stats_with,
            os.path.join(output_folder, "non_match_cases_with_history_statistics.txt"),
            dataset_name,
            extra_note=note,
            ci_comparison_note=ci_comparison_note
        )

        save_statistics_to_txt(
            stats_without,
            os.path.join(output_folder, "non_match_cases_without_history_statistics.txt"),
            dataset_name,
            ci_comparison_note=ci_comparison_note
        )

        save_statistics_to_txt(
            fk_stats_with,
            os.path.join(output_folder, "fk_with_history_statistics.txt"),
            dataset_name,
            n=n1_fk,
            extra_note=perm_note_fk,
            ci_comparison_note=fk_overlap_note
        )

        save_statistics_to_txt(
            fk_stats_without,
            os.path.join(output_folder, "fk_without_history_statistics.txt"),
            dataset_name,
            n=n2_fk,
            ci_comparison_note=fk_overlap_note
        )

    # Save categorized outputs
    save_json(os.path.join(output_folder, "empty_spans_cases.json"), empty_spans_cases)
    save_json(os.path.join(output_folder, "unknown_cases.json"), unknown_cases)
    save_json(os.path.join(output_folder, "short_cases.json"), short_cases)

    for label in classification_results:
        label_dir = os.path.join(output_folder, label.replace(" ", "_"))
        os.makedirs(label_dir, exist_ok=True)
        for cat, cases in classification_results[label].items():
            save_json(os.path.join(label_dir, f"{cat}.json"), cases)

            # Save SCE length histogram plots
            plot_histogram(
                data_valid=classification_results["With History"]["match_cases"],
                data_invalid=classification_results["With History"]["non_match_cases"],
                title="With History SCE Lengths",
                output_pdf=os.path.join(output_folder, "With_History_SCE_Lengths.pdf")
            )

            plot_histogram(
                data_valid=classification_results["Without History"]["match_cases"],
                data_invalid=classification_results["Without History"]["non_match_cases"],
                title="Without History SCE Lengths",
                output_pdf=os.path.join(output_folder, "Without_History_SCE_Lengths.pdf")
            )

    stats_match_with_history, stats_match_without_history, stats_non_match_with_history, stats_non_match_without_history = calculate_length_statistics(classification_results)


    total = len(all_results)
    short = len(short_cases)
    unknown = len(unknown_cases)
    non_match_without = len(classification_results["Without History"]["non_match_cases"])
    non_match_with = len(classification_results["With History"]["non_match_cases"])

    gen, val, val_c = compute_gen_val(total, short, unknown, non_match_without, non_match_with)

    write_summary(
        path=output_folder,
        classification_results=classification_results,
        total=total,
        unknown_count=unknown,
        non_empty_spans_count=non_empty_spans_count,
        short_case_count=short,
        stats_match_with_history=stats_match_with_history,
        stats_match_without_history=stats_match_without_history,
        stats_non_match_with_history=stats_non_match_with_history,
        stats_non_match_without_history=stats_non_match_without_history,
        gen=gen, val=val, val_c=val_c,
        fk_stats_with=fk_stats_with,
        fk_stats_without=fk_stats_without,
        fk_ci1=(fk_ci1_lower, fk_ci1_upper),
        fk_ci2=(fk_ci2_lower, fk_ci2_upper),
        perm_note_fk=perm_note_fk, 
        fk_overlap_note=fk_overlap_note,
        perm_note_nonmatch=perm_note_nonmatch,
        perm_note_validity=perm_note_validity,
        perm_note_ed=perm_note_ed,
        fk_result_with=fk_result_with,
        fk_result_without=fk_result_without,
    )



def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def compute_ci_rounded(p, n):
    if n == 0:
        print("WARNING: compute_ci_rounded received n = 0")
        return 0.0, 0.0, 0.0
    p_fraction = p / 100
    sd = np.sqrt(p_fraction * (1 - p_fraction))
    se = sd / np.sqrt(n)
    margin = 1.96 * se
    lower = round((p_fraction - margin) * 100)
    upper = round((p_fraction + margin) * 100)
    return lower, upper, se


def compute_se_and_ci_for_normalized_diff(mean1, std1, n1, mean2, std2, n2):
    # Debug print
    print(f"DEBUG: mean1 = {mean1}, std1 = {std1}, n1 = {n1}")
    print(f"DEBUG: mean2 = {mean2}, std2 = {std2}, n2 = {n2}")

    if n1 == 0 or n2 == 0:
        print("WARNING: One or both sample sizes are zero.")
        return 0.0, 0.0, 0.0, 0.0

    max_mean = max(mean1, mean2)
    if max_mean == 0:
        print("WARNING: max(mean1, mean2) == 0. Returning zeros.")
        return 0.0, 0.0, 0.0, 0.0

    diff = abs(mean1 - mean2)
    norm_diff = diff / max_mean * 100

    var1 = (std1 ** 2) / n1
    var2 = (std2 ** 2) / n2

    if mean1 >= mean2:
        dD_dmean1 = (1 / mean1) - (diff / mean1 ** 2) if mean1 != 0 else 0.0
        dD_dmean2 = -1 / mean1 if mean1 != 0 else 0.0
    else:
        dD_dmean1 = 1 / mean2 if mean2 != 0 else 0.0
        dD_dmean2 = -(1 / mean2) + (diff / mean2 ** 2) if mean2 != 0 else 0.0

    var_D = (dD_dmean1 ** 2) * var1 + (dD_dmean2 ** 2) * var2
    se = np.sqrt(var_D) * 100
    margin = 1.96 * se
    ci_lower = norm_diff - margin
    ci_upper = norm_diff + margin

    return norm_diff, se, ci_lower, ci_upper


def write_summary(
    path,
    classification_results,
    total,
    unknown_count,
    non_empty_spans_count,
    short_case_count,
    stats_match_with_history,
    stats_match_without_history,
    stats_non_match_with_history,
    stats_non_match_without_history,
    gen, val, val_c,
    fk_stats_with=None,
    fk_stats_without=None,
    fk_ci1=None,
    fk_ci2=None,
    perm_note_nonmatch=None,
    perm_note_validity=None,
    perm_note_ed=None,
    perm_note_fk=None,
    fk_overlap_note=None,
    fk_result_with=None,
    fk_result_without=None,
):

    n_valid = total - short_case_count - unknown_count
    n_with = len(classification_results["With History"]["match_cases"]) + len(classification_results["With History"]["non_match_cases"])
    n_without = len(classification_results["Without History"]["match_cases"]) + len(classification_results["Without History"]["non_match_cases"])

    # CI for Gen, Val, Val_c
    ci_gen_lower, ci_gen_upper, se_gen = compute_ci_rounded(gen, total)
    ci_val_lower, ci_val_upper, se_val = compute_ci_rounded(val, n_valid)
    ci_val_c_lower, ci_val_c_upper, se_val_c = compute_ci_rounded(val_c, n_valid)

    # CI for normalized diffs
    with_lengths_nonmatch = [
        len(item["SCEs"].split())
        for item in classification_results["With History"]["non_match_cases"]
        if "SCEs" in item and item["SCEs"].strip()
    ]

    with_lengths_match = [
        len(item["SCEs"].split())
        for item in classification_results["With History"]["match_cases"]
        if "SCEs" in item and item["SCEs"].strip()
    ]

    without_lengths_match = [len(item.get("SCEs", "").split()) for item in classification_results["Without History"]["match_cases"]]
    without_lengths_nonmatch = [len(item.get("SCEs", "").split()) for item in classification_results["Without History"]["non_match_cases"]]

    nd_with, nd_with_low, nd_with_high = safe_compute_bootstrap_nd(with_lengths_match, with_lengths_nonmatch)
    nd_without, nd_without_low, nd_without_high = safe_compute_bootstrap_nd(without_lengths_match, without_lengths_nonmatch)

    overlap = not (ci_val_upper < ci_val_c_lower or ci_val_c_upper < ci_val_lower)
    sig_note = "do NOT overlap → statistically significant" if not overlap else "overlap → not statistically significant"

    with open(os.path.join(path, "summary.txt"), "w") as f:
        f.write(f"Total cases: {total}\n")
        f.write(f"Unknown: {unknown_count}\n")
        f.write(f"Short cases: {short_case_count}\n")
        f.write(f"Non empty spans: {non_empty_spans_count}\n")

        f.write(f"\n--- SCE Length Statistics ---\n")

        f.write(f"Match Cases (With History):\n")
        f.write(f"  Mean: {stats_match_with_history['mean']:.2f} words\n")
        f.write(f"  Std Dev: {stats_match_with_history['std']:.2f} words\n")
        f.write(f"  Min: {stats_match_with_history['min']} words\n")
        f.write(f"  Max: {stats_match_with_history['max']} words\n")
        f.write(f"  Median: {stats_match_with_history['median']:.2f} words\n")

        f.write(f"Match Cases (Without History):\n")
        f.write(f"  Mean: {stats_match_without_history['mean']:.2f} words\n")
        f.write(f"  Std Dev: {stats_match_without_history['std']:.2f} words\n")
        f.write(f"  Min: {stats_match_without_history['min']} words\n")
        f.write(f"  Max: {stats_match_without_history['max']} words\n")
        f.write(f"  Median: {stats_match_without_history['median']:.2f} words\n")

        f.write(f"Non-Match Cases (With History):\n")
        f.write(f"  Mean: {stats_non_match_with_history['mean']:.2f} words\n")
        f.write(f"  Std Dev: {stats_non_match_with_history['std']:.2f} words\n")
        f.write(f"  Min: {stats_non_match_with_history['min']} words\n")
        f.write(f"  Max: {stats_non_match_with_history['max']} words\n")
        f.write(f"  Median: {stats_non_match_with_history['median']:.2f} words\n")

        f.write(f"Non-Match Cases (Without History):\n")
        f.write(f"  Mean: {stats_non_match_without_history['mean']:.2f} words\n")
        f.write(f"  Std Dev: {stats_non_match_without_history['std']:.2f} words\n")
        f.write(f"  Min: {stats_non_match_without_history['min']} words\n")
        f.write(f"  Max: {stats_non_match_without_history['max']} words\n")
        f.write(f"  Median: {stats_non_match_without_history['median']:.2f} words\n")

        # Recompute margins from SEs
        margin_gen = 1.96 * se_gen * 100
        margin_val = 1.96 * se_val * 100
        margin_val_c = 1.96 * se_val_c * 100

        f.write(f"\nNormalized Difference in Lengths (With History): {nd_with:.2f}% (95% CI: [{nd_with_low:.2f}%, {nd_with_high:.2f}%])\n")
        f.write(f"Normalized Difference in Lengths (Without History): {nd_without:.2f}% (95% CI: [{nd_without_low:.2f}%, {nd_without_high:.2f}%])\n")

        f.write(f"\nSCE Histogram With History: With_History_SCE_Lengths.pdf\n")
        f.write(f"SCE Histogram Without History: Without_History_SCE_Lengths.pdf\n")

        f.write("\n--- Classification Counts ---\n")
        for label, categories in classification_results.items():
            f.write(f"{label}:\n")
            for cat, cases in categories.items():
                f.write(f"  {cat}: {len(cases)}\n")
        
        f.write("\n--- Permutation Test Results ---\n")
        if perm_note_nonmatch:
            f.write(perm_note_nonmatch + "\n")
        if perm_note_validity:
            f.write(perm_note_validity + "\n")
        if perm_note_ed:
            f.write(perm_note_ed + "\n")
        
        for res in [fk_result_with, fk_result_without]:
            if not res or 'label' not in res:
                f.write("\n--- FK Readability: N/A (insufficient data) ---\n")
                continue

            f.write(f"\n--- FK Readability: {res['label']} ---\n")
            f.write(f"  Match Mean: {res['match_mean']:.2f}  → CI: [{res['ci1'][0]:.2f}, {res['ci1'][1]:.2f}]\n")
            f.write(f"  Non-Match Mean: {res['nonmatch_mean']:.2f}  → CI: [{res['ci2'][0]:.2f}, {res['ci2'][1]:.2f}]\n")
            f.write(f"  CI Overlap: {'YES → not significant' if res['overlap'] else 'NO → significant'}\n")
            f.write(f"  Paired Permutation Test: Mean Δ = {res['perm_diff']:.2f}, p = {res['p_val']:.4f} → "
                    f"{'Significant' if res['p_val'] < 0.05 else 'Not significant'}\n")

        f.write("\n--- Confidence Intervals (95%) with ± Margin ---\n")
        f.write(f"Gen: {round(gen)}% ± {margin_gen:.2f}% → CI: [{ci_gen_lower}%, {ci_gen_upper}%]\n")
        f.write(f"Val: {round(val)}% ± {margin_val:.2f}% → CI: [{ci_val_lower}%, {ci_val_upper}%]\n")
        f.write(f"Val_c: {round(val_c)}% ± {margin_val_c:.2f}% → CI: [{ci_val_c_lower}%, {ci_val_c_upper}%]\n")
        f.write(f"CI Comparison between Val and Val_c: {sig_note}\n")
        