import os
import json
import argparse
import numpy as np
import csv
import traceback
from collections import defaultdict

from metrics import (
    ruler_score_all
)


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


def canonical_method_name(method_name: str) -> str:
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


def discover_result_jsons(results_root: str):
    discovered = []
    skipped = []

    if not os.path.isdir(results_root):
        raise FileNotFoundError(f"results_dir not found: {results_root}")

    for current_root, _, files in os.walk(results_root):
        for filename in files:
            if not filename.endswith(".json"):
                continue
            if filename in {"metrics.json", "summary.json"}:
                continue
            if filename.endswith(".metrics.json"):
                continue

            full_path = os.path.join(current_root, filename)

            dataset = os.path.basename(os.path.dirname(full_path))
            if dataset not in DATASET_LIST:
                continue

            relative_parts = os.path.relpath(full_path, results_root).split(os.sep)

            if len(relative_parts) == 5:
                config_name = relative_parts[0]
                model_name = relative_parts[1]
                context_len = relative_parts[2]
            elif len(relative_parts) == 4:
                config_name = relative_parts[0]
                model_name = ""
                context_len = relative_parts[1]
            else:
                skipped.append(full_path)
                continue

            method = canonical_method_name(os.path.splitext(filename)[0])

            discovered.append(
                {
                    "config_name": config_name,
                    "model_name": model_name,
                    "context_len": context_len,
                    "dataset": dataset,
                    "method": method,
                    "file": full_path,
                }
            )

    if skipped:
        print(f"[WARN] Skipped {len(skipped)} files with unexpected path depth (need 4 or 5 levels):")
        for s in skipped[:5]:
            print(f"  {s}")
        if len(skipped) > 5:
            print(f"  ... and {len(skipped) - 5} more")

    return discovered


def evaluate_one_file(eval_file: str, score_mode: str = "hybrid", soft_weight: float = 0.8):
    predictions, answers = [], []
    try:
        with open(eval_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "pred" not in data or "answers" not in data:
                    continue
                predictions.append(data["pred"])
                answers.append(data["answers"])
    except (OSError, IOError) as e:
        print(f"[ERROR] Failed to read file {eval_file}: {e}")
        return None

    if len(predictions) == 0:
        return None
    return float(ruler_score_all(predictions, answers, mode=score_mode, soft_weight=soft_weight))


def write_context_csv(context_dir: str, score_table: dict, method_order: list):
    try:
        os.makedirs(context_dir, exist_ok=True)
    except OSError as e:
        print(f"[ERROR] Cannot create directory {context_dir}: {e}")
        return

    csv_rows = [["dataset"] + DATASET_LIST]
    for method in method_order:
        row = [method]
        for dataset in DATASET_LIST:
            value = score_table.get(method, {}).get(dataset, -1)
            row.append(value)
        csv_rows.append(row)

    csv_path = os.path.join(context_dir, "results.csv")
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerows(csv_rows)
    except (OSError, IOError) as e:
        print(f"[ERROR] Failed to write {csv_path}: {e}")


def write_unified_csv(results_root: str, all_records: list):
    output_path = os.path.join(results_root, "results_ruler_all.csv")

    header = ["config_name", "model_name", "context_length", "method"] + DATASET_LIST + ["avg_score"]

    method_config_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))
    for rec in all_records:
        key = (rec["config_name"], rec["model_name"], rec["context_len"], rec["method"])
        method_config_scores[key][rec["dataset"]] = rec["score"]

    rows = [header]
    for key in sorted(method_config_scores.keys()):
        config_name, model_name, context_len, method = key
        ds_scores = method_config_scores[key]
        row = [config_name, model_name, context_len, method]
        valid_scores = []
        for ds in DATASET_LIST:
            val = ds_scores.get(ds, -1)
            row.append(val)
            if val != -1:
                valid_scores.append(val)
        avg = round(float(np.mean(valid_scores)), 2) if valid_scores else -1
        row.append(avg)
        rows.append(row)

    try:
        with open(output_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerows(rows)
        print(f"Unified CSV written to: {output_path}")
    except (OSError, IOError) as e:
        print(f"[ERROR] Failed to write unified CSV {output_path}: {e}")


def write_global_csv(results_root: str, aggregated_scores: dict):
    output_path = os.path.join(results_root, "results_all_experiments.csv")
    rows = [["experiment", "model", "context_length", "method", "avg_score", "available_datasets", "total_datasets"]]

    for config_name in sorted(aggregated_scores.keys()):
        for model_name in sorted(aggregated_scores[config_name].keys()):
            for context_len in sorted(aggregated_scores[config_name][model_name].keys(), key=str):
                method_scores = aggregated_scores[config_name][model_name][context_len]
                for method in sorted(method_scores.keys()):
                    values = [v for v in method_scores[method].values() if v != -1]
                    avg_score = float(np.mean(values)) if len(values) > 0 else -1
                    rows.append(
                        [
                            config_name,
                            model_name,
                            context_len,
                            method,
                            avg_score,
                            len(values),
                            len(DATASET_LIST),
                        ]
                    )

    try:
        with open(output_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerows(rows)
        print(f"Global summary written to: {output_path}")
    except (OSError, IOError) as e:
        print(f"[ERROR] Failed to write global CSV {output_path}: {e}")


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--results_dir',
        type=str,
        default='results_ruler'
    )
    parser.add_argument(
        '--score_mode',
        type=str,
        default='hybrid',
        choices=['exact', 'soft', 'hybrid'],
        help='Scoring mode: exact keeps original substring match, soft uses similarity, hybrid is exact-or-soft.',
    )
    parser.add_argument(
        '--soft_weight',
        type=float,
        default=0.8,
        help='Weight applied to soft similarity when exact match is absent in hybrid mode.',
    )
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()

    if not os.path.isdir(args.results_dir):
        raise FileNotFoundError(f"results_dir not found: {args.results_dir}")

    discovered = discover_result_jsons(args.results_dir)
    if len(discovered) == 0:
        print(f"No valid result json files found under: {args.results_dir}")
        raise SystemExit(0)

    print(f"Discovered {len(discovered)} result files under: {args.results_dir}")

    # aggregated_scores[config_name][model_name][context_len][method][dataset] = score
    aggregated_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    all_records = []
    error_count = 0

    for item in discovered:
        try:
            score = evaluate_one_file(item["file"], score_mode=args.score_mode, soft_weight=args.soft_weight)
        except Exception as e:
            print(f"[ERROR] Exception evaluating {item['file']}: {e}")
            traceback.print_exc()
            score = None
            error_count += 1

        if score is None:
            score = -1

        aggregated_scores[item["config_name"]][item["model_name"]][item["context_len"]][item["method"]][item["dataset"]] = score

        all_records.append(
            {
                "config_name": item["config_name"],
                "model_name": item["model_name"],
                "context_len": item["context_len"],
                "method": item["method"],
                "dataset": item["dataset"],
                "score": score,
            }
        )

        metric_path = item["file"].replace(".json", ".metrics.json")
        try:
            with open(metric_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        item["dataset"]: score,
                        "_meta": {
                            "score_mode": args.score_mode,
                            "soft_weight": args.soft_weight,
                        },
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except (OSError, IOError) as e:
            print(f"[ERROR] Failed to write metric file {metric_path}: {e}")

        print(
            f"config={item['config_name']} model={item['model_name']} "
            f"context={item['context_len']} dataset={item['dataset']} "
            f"method={item['method']} score={score}"
        )

    if error_count > 0:
        print(f"\n[WARN] {error_count} files encountered errors during evaluation.")

    base_method_order = [
        "FullKV",
        "ChampionKV",
        "AdaKV",
        "AdaptiveKV",
        "random",
        "SnapKV",
        "PyramidKV",
    ]

    for config_name in sorted(aggregated_scores.keys()):
        for model_name in sorted(aggregated_scores[config_name].keys()):
            for context_len in sorted(aggregated_scores[config_name][model_name].keys(), key=str):
                score_table = aggregated_scores[config_name][model_name][context_len]
                discovered_methods = sorted(score_table.keys())

                method_order = [m for m in base_method_order if m in discovered_methods]
                method_order.extend([m for m in discovered_methods if m not in method_order])

                for method in discovered_methods:
                    for dataset in DATASET_LIST:
                        if dataset not in score_table[method]:
                            score_table[method][dataset] = -1

                context_dir = os.path.join(args.results_dir, config_name, model_name, context_len)
                write_context_csv(context_dir, score_table, method_order)

    write_global_csv(args.results_dir, aggregated_scores)
    write_unified_csv(args.results_dir, all_records)

