import argparse
from collections import Counter
import glob
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_INPUT_DIR = "results_needle/performance"
DEFAULT_OUTPUT_DIR = "results_needle/performance_plots"

METRICS = {
	"peak_gpu_memory_gb": "Peak GPU Memory (GB)",
	"steady_state_peak_gpu_memory_gb": "Steady-State Peak GPU Memory (GB)",
	"prefill_time_ms": "Prefill Time (ms)",
	"decode_time_ms": "Decode Time (ms)",
	"decoding_latency_ms_per_token": "Decoding Latency (ms/token)",
	"total_generation_time_ms": "Total Generation Time (ms)",
}


def _csv_label(csv_path: str) -> str:
	label = os.path.splitext(os.path.basename(csv_path))[0]
	return label.removesuffix("_performance")


def _pick_method_name(df: pd.DataFrame) -> str | None:
	if "model_version" in df.columns and df["model_version"].nunique(dropna=True) == 1:
		return str(df["model_version"].dropna().iloc[0])
	if "method" in df.columns and df["method"].nunique(dropna=True) == 1:
		return str(df["method"].dropna().iloc[0])
	return None


def _resolve_method_name(csv_path: str, candidate: str | None, name_counts: Counter) -> str:
	fallback = _csv_label(csv_path)
	if candidate and name_counts[candidate] == 1:
		return candidate
	return fallback


def load_records(input_dir: str) -> list[dict]:
	csv_paths = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
	if not csv_paths:
		print(f"No CSV files found in {input_dir}")
		return []

	loaded_files: list[tuple[str, pd.DataFrame, str | None]] = []
	for csv_path in csv_paths:
		try:
			df = pd.read_csv(csv_path)
		except Exception as exc:
			print(f"Failed to read {csv_path}: {exc}")
			continue

		loaded_files.append((csv_path, df, _pick_method_name(df)))

	name_counts = Counter(candidate for _, _, candidate in loaded_files if candidate)
	records: list[dict] = []
	for csv_path, df, candidate_name in loaded_files:
		if "context_length" not in df.columns:
			print(f"Missing context_length in {csv_path}, skipping")
			continue

		method_name = _resolve_method_name(csv_path, candidate_name, name_counts)
		df = df.copy()
		df["context_length"] = pd.to_numeric(df["context_length"], errors="coerce")

		for metric in METRICS:
			if metric not in df.columns:
				print(f"Missing {metric} in {csv_path}, skipping metric")
				continue
			metric_series = pd.to_numeric(df[metric], errors="coerce")
			grouped = (
				pd.DataFrame({"context_length": df["context_length"], "value": metric_series})
				.dropna()
				.groupby("context_length", as_index=False)["value"]
				.mean()
				.sort_values("context_length")
			)
			for _, row in grouped.iterrows():
				records.append(
					{
						"metric": metric,
						"method": method_name,
						"context_length": int(row["context_length"]),
						"value": float(row["value"]),
					}
				)

	return records


def plot_metrics(records: list[dict], output_dir: str) -> None:
	if not records:
		print("No records to plot")
		return

	os.makedirs(output_dir, exist_ok=True)
	df = pd.DataFrame(records)

	for metric, ylabel in METRICS.items():
		metric_df = df[df["metric"] == metric]
		if metric_df.empty:
			print(f"No data for {metric}, skipping")
			continue

		plt.figure(figsize=(10, 6))
		for method in sorted(metric_df["method"].unique()):
			method_df = (
				metric_df[metric_df["method"] == method]
				.groupby("context_length", as_index=False)["value"]
				.mean()
				.sort_values("context_length")
			)
			plt.plot(
				method_df["context_length"],
				method_df["value"],
				marker="o",
				linewidth=2,
				label=method,
			)

		plt.xlabel("Context Length")
		plt.ylabel(ylabel)
		plt.title(f"{ylabel} vs Context Length")
		plt.grid(True, linestyle="--", alpha=0.4)
		plt.legend()
		plt.tight_layout()

		output_path = os.path.join(output_dir, f"{metric}.png")
		plt.savefig(output_path, dpi=150)
		plt.close()
		print(f"Saved {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Visualize performance CSV results.")
	parser.add_argument(
		"--input_dir",
		default=DEFAULT_INPUT_DIR,
		help="Folder containing performance CSV files",
	)
	parser.add_argument(
		"--output_dir",
		default=DEFAULT_OUTPUT_DIR,
		help="Folder to write plots",
	)
	return parser


def main() -> None:
	parser = build_arg_parser()
	args = parser.parse_args()
	records = load_records(args.input_dir)
	plot_metrics(records, args.output_dir)


if __name__ == "__main__":
	main()
