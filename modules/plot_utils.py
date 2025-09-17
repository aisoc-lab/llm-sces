import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # capture everything

# Send logs only to a file, not to console
file_handler = logging.FileHandler("postprocess_debug.log")
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

# Clear any default handlers (which send warnings to stderr)
if logger.hasHandlers():
    logger.handlers.clear()

logger.addHandler(file_handler)

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

def load_json(file_path):
    """Load JSON data safely from a given file path."""
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to load JSON from {file_path}: {e}", exc_info=True)
        return []

def extract_scenario_lengths(data):
    """Extract word counts of 'SCEs' values from data."""
    return [len(item["SCEs"].split()) for item in data if isinstance(item, dict) and "SCEs" in item]

def plot_histogram(data_valid, data_invalid, title, output_pdf):
    """Plot and save histogram comparing lengths of valid vs invalid SCEs."""
    colors = ["#92c5de", "#f4a582"]
    labels = ["Invalid SCEs", "Valid SCEs"]

    # Precompute lengths
    lengths_invalid = extract_scenario_lengths(data_invalid)
    lengths_valid = extract_scenario_lengths(data_valid)

    if lengths_invalid or lengths_valid:
        all_lengths = lengths_invalid + lengths_valid
        bin_edges = np.histogram_bin_edges(all_lengths, bins=15)

        plt.figure(figsize=(6, 2))
        for length_data, color, label in zip([lengths_invalid, lengths_valid], colors, labels):
            if length_data:  # plot only if non-empty
                sns.histplot(
                    length_data, bins=bin_edges, edgecolor="black", alpha=0.6,
                    color=color, label=label, stat="density", linewidth=2
                )

        plt.xlim(10, 110)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("")
        plt.ylabel("")
        plt.legend(fontsize=14, loc="upper right", frameon=False)
        plt.tight_layout()
        plt.savefig(output_pdf, format="pdf", bbox_inches="tight")
        plt.close()
        logger.info(f"Histogram saved as {output_pdf}")
    else:
        logger.warning(f"Skipping {title} - No data available.")
