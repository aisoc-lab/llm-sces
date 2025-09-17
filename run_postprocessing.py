import warnings
# Silence warnings ASAP (before importing transformers)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
import sys
import builtins
import argparse
from tqdm import tqdm
from transformers.utils import logging as hf_logging
from modules.postprocess_utils import process_results_with_statistics


def setup_debug_printing(log_file="postprocess_debug.log"):
    """Redirect all print() output to a log file, except tqdm progress bar."""
    original_print = builtins.print

    def debug_print(*args, **kwargs):
        with open(log_file, "a") as f:
            msg = " ".join(str(a) for a in args)
            f.write(msg + "\n")

    builtins.print = debug_print


def setup_warnings_and_logging():
    """Silence unnecessary warnings from libraries."""
    hf_logging.set_verbosity_error()  # silence HF logs


def initialize_environment():
    """Apply logging/printing setup for postprocessing."""
    setup_debug_printing("postprocess_debug.log")
    setup_warnings_and_logging()


def parse_args():
    parser = argparse.ArgumentParser(description="Run postprocessing on evaluation results.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_dir", type=str, help="Path to folder containing result JSON files")
    group.add_argument("--input_file", type=str, help="Path to a specific JSON file")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., DiscrimEval, Folktexts, GSM8K)")
    return parser.parse_args()


def main():
    initialize_environment()  # silence DEBUG
    args = parse_args()
    dataset_name = args.dataset

    if args.input_file:
        json_file_path = args.input_file
        print(f"Processing single file: {json_file_path}")
        process_results_with_statistics(json_file_path, dataset_name)

    elif args.input_dir:
        json_files = []
        for root, _, files in os.walk(args.input_dir):
            for file in files:
                if file.endswith(".json"):
                    json_files.append(os.path.join(root, file))

        for json_file_path in tqdm(json_files, desc="Postprocessing files"):
            process_results_with_statistics(json_file_path, dataset_name)


if __name__ == "__main__":
    main()