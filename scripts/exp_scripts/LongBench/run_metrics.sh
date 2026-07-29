#!/usr/bin/env bash
#
# Evaluate LongBench results and produce per-context CSV summaries.
# Usage:  bash run_metrics.sh <results_dir>
#

set -euo pipefail

RESULTS_DIR="${1:?Usage: bash run_metrics.sh <results_dir>}"

python3 "$(dirname "$0")/../../eval_longbench.py" --results_dir "${RESULTS_DIR}"
