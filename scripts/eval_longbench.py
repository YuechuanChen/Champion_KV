import os
import json
import csv
import io
import argparse
import numpy as np
from datetime import datetime

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

DATASET_LIST = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]

DATASET_SET = set(DATASET_LIST)

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Evaluate LongBench results by auto-scanning result folders")
    parser.add_argument('--results_dir', type=str, required=True, help="Root dir containing method folders")
    parser.add_argument('--longbench_e', action='store_true', help="Evaluate on LongBench-E (length buckets)")
    parser.add_argument('--output_name', type=str, default='results.csv', help="Summary output filename")
    return parser.parse_args(args)


def safe_mean(values):
    if not values:
        return -1
    return round(float(np.mean(values)), 2)


def find_method_dirs(results_dir):
    method_dirs = []
    for name in sorted(os.listdir(results_dir)):
        path = os.path.join(results_dir, name)
        if not os.path.isdir(path):
            continue
        if name.startswith('.') or name.lower() in {"logs", "log"}:
            continue
        method_dirs.append((name, path))
    return method_dirs


def find_result_file(dataset_dir):
    if not os.path.isdir(dataset_dir):
        return None

    candidates = []
    for file_name in sorted(os.listdir(dataset_dir)):
        if file_name == "metrics.json":
            continue
        if file_name.endswith(".json") or file_name.endswith(".jsonl"):
            candidates.append(os.path.join(dataset_dir, file_name))

    if not candidates:
        return None
    return candidates[0]


def find_method_result_files(dataset_dir):
    if not os.path.isdir(dataset_dir):
        return []

    method_files = {}
    for file_name in sorted(os.listdir(dataset_dir)):
        if file_name == "metrics.json":
            continue
        if not (file_name.endswith(".json") or file_name.endswith(".jsonl")):
            continue
        method_name = os.path.splitext(file_name)[0]
        method_files.setdefault(method_name, os.path.join(dataset_dir, file_name))
    return sorted(method_files.items())


def has_dataset_subdirs(path):
    for name in os.listdir(path):
        sub_path = os.path.join(path, name)
        if os.path.isdir(sub_path) and name in DATASET_SET:
            return True
    return False


def read_predictions(eval_file):
    predictions, answers, lengths = [], [], []
    all_classes = None

    with open(eval_file, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return predictions, answers, lengths, all_classes

    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            records = []
            break

    if not records:
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                records = obj
            elif isinstance(obj, dict):
                records = [obj]
        except json.JSONDecodeError:
            return predictions, answers, lengths, all_classes

    for data in records:
        if not isinstance(data, dict):
            continue
        if "pred" not in data or "answers" not in data:
            continue
        predictions.append(data["pred"])
        answers.append(data["answers"])
        all_classes = data.get("all_classes", all_classes)
        if "length" in data:
            lengths.append(data["length"])

    return predictions, answers, lengths, all_classes


def build_results_csv_text(results_dir, dataset_list, method_names, score_table):
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["LongBench Evaluation Summary"])
    writer.writerow(["Generated at", datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
    writer.writerow(["Results dir", results_dir])
    writer.writerow([])

    header = ["method"] + dataset_list + ["avg"]
    writer.writerow(header)

    for method in method_names:
        row_scores = [score_table[method].get(ds, -1) for ds in dataset_list]
        valid_scores = [s for s in row_scores if isinstance(s, (int, float)) and s >= 0]
        avg_score = safe_mean(valid_scores)
        row = [method] + row_scores + [avg_score]
        writer.writerow(row)

    writer.writerow([])
    writer.writerow(["Per-dataset best method"])
    writer.writerow(["dataset", "best_method", "best_score"])
    for ds in dataset_list:
        best_method = None
        best_score = -1
        for method in method_names:
            s = score_table[method].get(ds, -1)
            if isinstance(s, (int, float)) and s > best_score:
                best_score = s
                best_method = method
        if best_method is None:
            writer.writerow([ds, "no valid score", ""])
        else:
            writer.writerow([ds, best_method, best_score])

    return output.getvalue()

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        if scores[key]:
            scores[key] = round(100 * np.mean(scores[key]), 2)
        else:
            scores[key] = -1
    return scores

def scorer(dataset, predictions, answers, all_classes):
    if not predictions:
        return -1
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)


def evaluate_single_file(dataset, eval_file, longbench_e):
    predictions, answers, lengths, all_classes = read_predictions(eval_file)
    if not predictions:
        return -1

    if longbench_e:
        if not lengths:
            return -1
        return scorer_e(dataset, predictions, answers, lengths, all_classes)

    return scorer(dataset, predictions, answers, all_classes)


def evaluate_method_first_layout(method_dirs, longbench_e):
    score_table = {method_name: {} for method_name, _ in method_dirs}

    for method_name, method_path in method_dirs:
        print(f"Evaluating method: {method_name}")
        for dataset in DATASET_LIST:
            dataset_dir = os.path.join(method_path, dataset)
            eval_file = find_result_file(dataset_dir)
            if eval_file is None:
                score_table[method_name][dataset] = -1
                print(f"  {dataset}: missing file")
                continue

            score = evaluate_single_file(dataset, eval_file, longbench_e)
            score_table[method_name][dataset] = score

            with open(os.path.join(dataset_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump({dataset: score}, f, ensure_ascii=False, indent=4)

            print(f"  {dataset}: {score}")

    method_names = [name for name, _ in method_dirs]
    return method_names, score_table


def evaluate_branch_dataset_method_layout(results_dir, branch_dirs, longbench_e):
    method_names = []
    score_table = {}

    for branch_name, branch_path in branch_dirs:
        print(f"Evaluating branch: {branch_name}")
        for dataset in DATASET_LIST:
            dataset_dir = os.path.join(branch_path, dataset)
            method_files = find_method_result_files(dataset_dir)
            if not method_files:
                print(f"  {dataset}: missing method files")
                continue

            dataset_metrics = {}
            for method_name, eval_file in method_files:
                branch_method = f"{branch_name}/{method_name}"
                if branch_method not in score_table:
                    score_table[branch_method] = {}
                    method_names.append(branch_method)

                score = evaluate_single_file(dataset, eval_file, longbench_e)
                score_table[branch_method][dataset] = score
                dataset_metrics[method_name] = score
                print(f"  {dataset} [{method_name}]: {score}")

            with open(os.path.join(dataset_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(dataset_metrics, f, ensure_ascii=False, indent=4)

    return method_names, score_table


def evaluate_dataset_method_layout(results_dir, longbench_e):
    method_names = []
    score_table = {}

    for dataset in DATASET_LIST:
        dataset_dir = os.path.join(results_dir, dataset)
        method_files = find_method_result_files(dataset_dir)
        if not method_files:
            print(f"  {dataset}: missing method files")
            continue

        dataset_metrics = {}
        for method_name, eval_file in method_files:
            if method_name not in score_table:
                score_table[method_name] = {}
                method_names.append(method_name)

            score = evaluate_single_file(dataset, eval_file, longbench_e)
            score_table[method_name][dataset] = score
            dataset_metrics[method_name] = score
            print(f"  {dataset} [{method_name}]: {score}")

        with open(os.path.join(dataset_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(dataset_metrics, f, ensure_ascii=False, indent=4)

    return method_names, score_table

if __name__ == '__main__':
    args = parse_args()

    results_dir = os.path.abspath(args.results_dir)
    if not os.path.isdir(results_dir):
        raise ValueError(f"results_dir does not exist: {results_dir}")

    method_dirs = find_method_dirs(results_dir)
    if has_dataset_subdirs(results_dir):
        print("Detected layout: dataset/method-file")
        method_names, score_table = evaluate_dataset_method_layout(results_dir, args.longbench_e)
    else:
        if not method_dirs:
            raise ValueError(f"No method or branch directories found under: {results_dir}")

        if all(has_dataset_subdirs(path) for _, path in method_dirs):
            print("Detected layout: branch/dataset/method-file")
            method_names, score_table = evaluate_branch_dataset_method_layout(results_dir, method_dirs, args.longbench_e)
        else:
            print("Detected layout: method/dataset/result-file")
            method_names, score_table = evaluate_method_first_layout(method_dirs, args.longbench_e)

    if not method_names:
        raise ValueError(f"No evaluable method files found under: {results_dir}")

    summary_text = build_results_csv_text(results_dir, DATASET_LIST, method_names, score_table)
    output_path = os.path.join(results_dir, args.output_name)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"Saved summary to: {output_path}")
