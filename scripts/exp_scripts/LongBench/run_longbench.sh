#!/usr/bin/env bash
#
# Unified LongBench experiment launcher.
# Usage:
#   bash run_longbench.sh <model>
#
# Supported models: llama8b, llama70b, mistral7b
# Override any variable by setting it before calling, e.g.:
#   MODEL_PATH=/path/to/model METHODS="SnapKV AdaptiveKV" bash run_longbench.sh llama8b
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUN_LONGBENCH_PY="${REPO_ROOT}/run_longbench.py"

# Model selection
MODEL="${1:-llama8b}"

case "${MODEL}" in
    llama8b)
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
        MODEL_PATH="${MODEL_PATH:-/path/to/Llama-3.1-8B-Instruct}"
        SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/results_long_bench_Llama8B}"
        ;;
    llama70b)
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
        MODEL_PATH="${MODEL_PATH:-/path/to/Llama-3.1-70B-Instruct}"
        SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/results_long_bench_Llama70B}"
        ;;
    mistral7b)
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
        MODEL_PATH="${MODEL_PATH:-/path/to/Mistral-7B-Instruct-v0.3}"
        SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/results_long_bench_Mistral}"
        ;;
    *)
        echo "Unknown model: ${MODEL}"
        echo "Supported: llama8b, llama70b, mistral7b"
        exit 1
        ;;
esac

export CUDA_VISIBLE_DEVICES

mkdir -p "${SAVE_DIR}/logs"

# Experiment hyper-parameters
METHODS=("${METHODS[@]:-championkv}")       # fullkv, pyramidkv, snapkv, adaptivekv, championkv
MAX_CAPACITY_PROMPTS=("${MAX_CAPACITY_PROMPTS[@]:-128}")

BETA="${BETA:-20}"
GQA_SUPPORT="${GQA_SUPPORT:-true}"
GQA_FUNC="${GQA_FUNC:-mean}"
WINDOW_SIZE="${WINDOW_SIZE:-8}"
ALLOCATE_STRATEGY="${ALLOCATE_STRATEGY:-pyramid}"

INSPECT_ATTENTION_WEIGHT="${INSPECT_ATTENTION_WEIGHT:-false}"
ATTENTION_LOG_FILE="${ATTENTION_LOG_FILE:-${SAVE_DIR}/Attention_weight.log}"
ENABLE_RUN_LOG="${ENABLE_RUN_LOG:-false}"

ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
MERGE_METHOD="${MERGE_METHOD:-}"
MODEL_NAME_SUFFIX="${MODEL_NAME_SUFFIX:-}"
QUANT_METHOD="${QUANT_METHOD:-}"
NBITS="${NBITS:-8}"

# Parse datasets for logging
LONG_BENCH_DATASETS="<unknown>"
if [[ -f "${RUN_LONGBENCH_PY}" ]]; then
    LONG_BENCH_DATASETS="$(awk '
        BEGIN { in_list=0; out="" }
        /^[[:space:]]*datasets[[:space:]]*=[[:space:]]*\[/ { in_list=1 }
        in_list {
            line=$0; gsub(/\\/, "", line)
            while (match(line, /"[^"]+"/)) {
                item=substr(line, RSTART+1, RLENGTH-2)
                out = (out == "" ? item : out ", " item)
                line=substr(line, RSTART+RLENGTH)
            }
            if (line ~ /\]/) { print out; exit }
        }
    ' "${RUN_LONGBENCH_PY}")"
fi

# Print configuration
TOTAL_RUNS=$(( ${#METHODS[@]} * ${#MAX_CAPACITY_PROMPTS[@]} ))

echo "========== LongBench: ${MODEL} =========="
echo "GPU:            ${CUDA_VISIBLE_DEVICES}"
echo "model_path:     ${MODEL_PATH}"
echo "save_dir:       ${SAVE_DIR}"
echo "methods (${#METHODS[@]}):      ${METHODS[*]}"
echo "capacities (${#MAX_CAPACITY_PROMPTS[@]}):   ${MAX_CAPACITY_PROMPTS[*]}"
echo "attn:           ${ATTN_IMPLEMENTATION}"
echo "beta:           ${BETA}"
echo "gqa:            ${GQA_SUPPORT} / ${GQA_FUNC}"
echo "window_size:    ${WINDOW_SIZE}"
echo "allocate:       ${ALLOCATE_STRATEGY}"
echo "merge:          ${MERGE_METHOD:-<none>}"
echo "quant:          ${QUANT_METHOD:-<none>}"
echo "run_log:        ${ENABLE_RUN_LOG}"
echo "suffix:         ${MODEL_NAME_SUFFIX:-<none>}"
echo "datasets:       ${LONG_BENCH_DATASETS}"
echo "total runs:     ${TOTAL_RUNS}"
echo "========================================="

# Build argument arrays
MERGE_TAG="${MERGE_METHOD:-none}"; MERGE_TAG="${MERGE_TAG//\//-}"; MERGE_TAG="${MERGE_TAG// /}"

MODEL_NAME_SUFFIX_ARG=()
[[ -n "${MODEL_NAME_SUFFIX}" ]] && MODEL_NAME_SUFFIX_ARG=(--model_name_suffix "${MODEL_NAME_SUFFIX}")

QUANT_ARGS=()
[[ -n "${QUANT_METHOD}" ]] && QUANT_ARGS=(--quant_method "${QUANT_METHOD}" --nbits "${NBITS}")

MERGE_ARG=()
[[ -n "${MERGE_METHOD}" ]] && MERGE_ARG=(--merge "${MERGE_METHOD}")

GQA_ARGS=()
[[ "${GQA_SUPPORT}" == "true" ]] && GQA_ARGS=(--gqa_support)

ATTN_INSPECT_ARGS=()
[[ "${INSPECT_ATTENTION_WEIGHT}" == "true" ]] && \
    ATTN_INSPECT_ARGS=(--inspect_attention_weight --attention_log_file "${ATTENTION_LOG_FILE}")

# Run experiments
for METHOD in "${METHODS[@]}"; do
    for MAX_CAPACITY_PROMPT in "${MAX_CAPACITY_PROMPTS[@]}"; do
            config_tag="cap${MAX_CAPACITY_PROMPT}_beta${BETA}_gqa${GQA_SUPPORT}_${GQA_FUNC}_alloc${ALLOCATE_STRATEGY}_merge_${MERGE_TAG}"
            model_version="LongBench_${METHOD}_${config_tag}"
            run_name="${model_version}"
            [[ -n "${MODEL_NAME_SUFFIX}" ]] && run_name="${run_name}_${MODEL_NAME_SUFFIX}"

            echo "[RUN] method=${METHOD} cap=${MAX_CAPACITY_PROMPT}"

            if [[ "${ENABLE_RUN_LOG}" == "true" ]]; then
                (
                python -u "${RUN_LONGBENCH_PY}" \
                    --method "${METHOD}" \
                    --model_path "${MODEL_PATH}" \
                    --max_capacity_prompts "${MAX_CAPACITY_PROMPT}" \
                    --attn_implementation "${ATTN_IMPLEMENTATION}" \
                    --beta "${BETA}" \
                    --window_size "${WINDOW_SIZE}" \
                    --allocate_strategy "${ALLOCATE_STRATEGY}" \
                    --gqa_func "${GQA_FUNC}" \
                    --save_dir "${SAVE_DIR}" \
                    "${GQA_ARGS[@]}" \
                    "${MERGE_ARG[@]}" \
                    "${MODEL_NAME_SUFFIX_ARG[@]}" \
                    "${ATTN_INSPECT_ARGS[@]}" \
                    "${QUANT_ARGS[@]}"
                ) 2>&1 | tee "${SAVE_DIR}/logs/${run_name}.log"
            else
                python -u "${RUN_LONGBENCH_PY}" \
                    --method "${METHOD}" \
                    --model_path "${MODEL_PATH}" \
                    --max_capacity_prompts "${MAX_CAPACITY_PROMPT}" \
                    --attn_implementation "${ATTN_IMPLEMENTATION}" \
                    --beta "${BETA}" \
                    --window_size "${WINDOW_SIZE}" \
                    --allocate_strategy "${ALLOCATE_STRATEGY}" \
                    --gqa_func "${GQA_FUNC}" \
                    --save_dir "${SAVE_DIR}" \
                    "${GQA_ARGS[@]}" \
                    "${MERGE_ARG[@]}" \
                    "${MODEL_NAME_SUFFIX_ARG[@]}" \
                    "${ATTN_INSPECT_ARGS[@]}" \
                    "${QUANT_ARGS[@]}"
            fi
    done
done



