import argparse
import json
import re
import sys
from typing import Any, Dict, List

# Precompiled URL regex for efficiency
URL_PATTERN = re.compile(r"http\S+")


def add_id_to_json(json_file: str, output_json: str) -> None:
    """
    Extracts the list inside 'data', adds an 'id' to each item,
    and removes 'data' & 'count' keys.
    """
    try:
        with open(json_file, "r", encoding="utf-8") as file:
            data: Dict[str, Any] = json.load(file)
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to read or parse JSON file {json_file}: {e}")

    if not isinstance(data, dict) or "data" not in data or not isinstance(data["data"], list):
        raise ValueError("The JSON structure must contain a 'data' key with a list of dictionaries.")

    processed_data: List[Dict[str, Any]] = []
    for idx, item in enumerate(data["data"], start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid item at index {idx}: Each entry in 'data' must be a dictionary.")
        item["id"] = idx
        processed_data.append(item)

    try:
        with open(output_json, "w", encoding="utf-8") as file:
            json.dump(processed_data, file, indent=4, ensure_ascii=False)
    except OSError as e:
        raise ValueError(f"Failed to write JSON to {output_json}: {e}")

    print(f"JSON file processed and saved to: {output_json}")


def remove_urls_from_json(input_file: str, output_file: str) -> None:
    """Removes URLs from 'text' fields inside JSON data."""
    try:
        with open(input_file, "r", encoding="utf-8") as file:
            data: List[Dict[str, Any]] = json.load(file)
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to read or parse JSON file {input_file}: {e}")

    if not isinstance(data, list):
        raise ValueError("Expected a list of dictionaries at the root of the JSON file.")

    for entry in data:
        if isinstance(entry, dict) and "text" in entry and isinstance(entry["text"], str):
            entry["text"] = URL_PATTERN.sub("", entry["text"]).strip()

    try:
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
    except OSError as e:
        raise ValueError(f"Failed to write JSON to {output_file}: {e}")

    print(f"Processed JSON saved to: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess JSON datasets.")
    parser.add_argument("--dataset", required=True, help="Dataset name: MGNLI, SST2, Folktexts, Twitter")
    parser.add_argument("--input", required=True, help="Input JSON file path")
    parser.add_argument("--output", required=True, help="Output JSON file path")

    args = parser.parse_args()
    dataset = args.dataset.strip().lower()

    try:
        if dataset in {"mgnli", "sst2", "folktexts"}:
            add_id_to_json(args.input, args.output)
        elif dataset == "twitter":
            remove_urls_from_json(args.input, args.output)
        else:
            print("Unsupported dataset name. Choose from: MGNLI, SST2, Folktexts, Twitter.")
            sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
