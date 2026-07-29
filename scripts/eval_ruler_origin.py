import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

from metrics import string_match_all


DEFAULT_RESULTS_DIR = "results_ruler"

DATASET_LIST = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multiquery",
    "niah_multivalue",
    "cwe",
    "fwe",
    "vt",
]

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args(args)


def canonical_method_name(method_name):
    mapping = {
        "fullkv": "FullKV",
        "random": "random",
        "snapkv": "SnapKV",
        "pyramidkv": "PyramidKV",
        "adakv": "AdaKV",
        "adaptivekv": "AdaptiveKV",
        "championkv": "ChampionKV",
    }
    key = method_name.strip().lower()
    return mapping.get(key, method_name)


def discover_result_files(results_dir):
    """Find JSONL result files saved as experiment/context/dataset/method.json."""
    records = []
    skipped = []

    for current_root, _, files in os.walk(results_dir):
        dataset = os.path.basename(current_root)
        if dataset not in DATASET_LIST:
            continue

        relative_dir = os.path.relpath(current_root, results_dir)
        parts = relative_dir.split(os.sep)
        if len(parts) == 3:
            experiment, context_length, dataset_name = parts
            model_name = ""
        elif len(parts) == 4:
            experiment, model_name, context_length, dataset_name = parts
        else:
            skipped.append(current_root)
            continue

        if dataset_name != dataset:
            skipped.append(current_root)
            continue

        for filename in files:
            if not filename.endswith(".json"):
                continue
            if filename.endswith(".metrics.json") or filename in {"metrics.json", "summary.json"}:
                continue

            method = canonical_method_name(os.path.splitext(filename)[0])
            records.append(
                {
                    "experiment": experiment,
                    "model": model_name,
                    "context_length": context_length,
                    "dataset": dataset_name,
                    "method": method,
                    "path": os.path.join(current_root, filename),
                }
            )

    if skipped:
        print(f"[WARN] Skipped {len(skipped)} dataset directories with unexpected layout.")
        for path in skipped[:5]:
            print(f"  {path}")
        if len(skipped) > 5:
            print(f"  ... and {len(skipped) - 5} more")

    return records


def evaluate_file(eval_file):
    predictions, answers = [], []

    with open(eval_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] Bad JSON line skipped in {eval_file}")
                continue

            if "pred" not in data or "answers" not in data:
                print(f"[WARN] Missing pred/answers line skipped in {eval_file}")
                continue

            predictions.append(data["pred"])
            answers.append(data["answers"])

    if not predictions:
        return -1

    return string_match_all(predictions, answers)


def write_ruler_all_csv(results_dir, grouped_scores):
    header = ["experiment", "model", "context_length", "method"] + DATASET_LIST + ["avg_score"]
    rows = [header]

    for experiment in sorted(grouped_scores):
        for model in sorted(grouped_scores[experiment]):
            for context_length in sorted(grouped_scores[experiment][model], key=str):
                for method in sorted(grouped_scores[experiment][model][context_length]):
                    scores = grouped_scores[experiment][model][context_length][method]
                    row = [experiment, model, context_length, method]
                    valid_scores = []
                    for dataset in DATASET_LIST:
                        score = scores.get(dataset, -1)
                        row.append(score)
                        if score != -1:
                            valid_scores.append(score)
                    row.append(round(float(np.mean(valid_scores)), 2) if valid_scores else -1)
                    rows.append(row)

    output_path = os.path.join(results_dir, "results_ruler_all.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerows(rows)
    print(f"Saved merged RULER CSV to: {output_path}")


def main():
    args = parse_args()

    if not os.path.isdir(args.results_dir):
        raise FileNotFoundError(f"results_dir not found: {args.results_dir}")

    result_files = discover_result_files(args.results_dir)
    if not result_files:
        print(f"No valid RULER result JSON files found under: {args.results_dir}")
        return

    print(f"Discovered {len(result_files)} RULER result files under: {args.results_dir}")

    grouped_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

    for item in result_files:
        try:
            score = evaluate_file(item["path"])
        except Exception as exc:
            print(f"[ERROR] Failed to evaluate {item['path']}: {exc}")
            score = -1

        grouped_scores[item["experiment"]][item["model"]][item["context_length"]][item["method"]][item["dataset"]] = score

        print(
            f"experiment={item['experiment']} model={item['model']} "
            f"context={item['context_length']} dataset={item['dataset']} "
            f"method={item['method']} score={score}"
        )

    write_ruler_all_csv(args.results_dir, grouped_scores)


if __name__ == "__main__":
    main()

