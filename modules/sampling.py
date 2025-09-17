import json
import random
import argparse
from typing import Any, Dict, List
from datasets import load_dataset, Dataset
from modules.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Sampling helper")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--sample_size", type=int, default=250, help="Number of samples per class/group")
    return parser.parse_args()


args = parse_args()
SAMPLE_SIZE = args.sample_size

set_seed(args.seed)
random.seed(args.seed)


def save_to_json(data: List[Dict[str, Any]], filename: str) -> None:
    """Utility to save filtered dataset with count."""
    output = {"data": data, "count": len(data)}
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
    print(f"Saved {len(data)} records to '{filename}'.")


def sample_discrimeval() -> None:
    dataset = load_dataset("Anthropic/discrim-eval", "explicit")["train"]

    filtered_data = [
        {"filled_template": row["filled_template"]}
        for row in dataset
        if row["age"] == 20 and row["gender"] == "female" and row["race"] == "white"
    ]
    save_to_json(filtered_data, "filtered_data.json")


def sample_sst2() -> None:
    dataset = load_dataset("stanfordnlp/sst2")["train"]

    negative_samples = [row for row in dataset if row["label"] == 0][:SAMPLE_SIZE]
    positive_samples = [row for row in dataset if row["label"] == 1][:SAMPLE_SIZE]

    filtered_data = [
        {"idx": row["idx"], "sentence": row["sentence"], "label": row["label"]}
        for row in (negative_samples + positive_samples)
    ]
    save_to_json(filtered_data, "SST2.json")


def sample_twitter() -> None:
    dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")["train"]

    samples = []
    for label in (0, 1, 2):
        samples.extend([row for row in dataset if row["label"] == label][:SAMPLE_SIZE])

    filtered_data = [
        {"id": idx, "text": row["text"], "label": row["label"]}
        for idx, row in enumerate(samples)
    ]
    save_to_json(filtered_data, "Twitter.json")


def sample_folktexts() -> None:
    dataset = load_dataset("acruz/folktexts")["train"]

    negative_samples = [
        row for row in dataset if row["label"] == 0 and row["SEX"] == 1
    ][:SAMPLE_SIZE]
    positive_samples = [
        row for row in dataset if row["label"] == 1 and row["SEX"] == 2
    ][:SAMPLE_SIZE]

    filtered_data = [
        {
            "idx": row["id"],
            "description": row["description"],
            "question": row["question"],
            "choices": row["choices"],
            "label": row["label"],
            "SEX": row["SEX"],
        }
        for row in (negative_samples + positive_samples)
    ]
    save_to_json(filtered_data, "folktexts.json")


def sample_gsm8k() -> None:
    dataset = load_dataset("openai/gsm8k", "main")["train"][:SAMPLE_SIZE]

    combined = [
        {"id": idx + 1, "question": row["question"], "answer": row["answer"]}
        for idx, row in enumerate(dataset)
    ]

    with open("questions.json", "w", encoding="utf-8") as q_file:
        json.dump(
            [{"id": c["id"], "question": c["question"]} for c in combined],
            q_file,
            indent=4,
        )

    with open("answers.json", "w", encoding="utf-8") as a_file:
        json.dump(
            [{"id": c["id"], "answer": c["answer"]} for c in combined],
            a_file,
            indent=4,
        )

    with open("combined.json", "w", encoding="utf-8") as combined_file:
        json.dump(combined, combined_file, indent=4)

    print("Files created: questions.json, answers.json, combined.json")


def sample_mgnli() -> None:
    dataset = load_dataset("nyu-mll/multi_nli")["train"]

    filtered_data = [
        {
            "premise": row["premise"],
            "hypothesis": row["hypothesis"],
            "pairID": row["pairID"],
            "label": row["label"],
        }
        for row in dataset
        if row["genre"] == "government"
    ]

    label_groups = {0: [], 1: [], 2: []}
    for row in filtered_data:
        if row["label"] in label_groups:
            label_groups[row["label"]].append(row)

    sampled_data = []
    for label, rows in label_groups.items():
        if rows:  # avoid empty label group crash
            sampled_data.extend(random.sample(rows, min(SAMPLE_SIZE, len(rows))))

    save_to_json(sampled_data, "NLI.json")
