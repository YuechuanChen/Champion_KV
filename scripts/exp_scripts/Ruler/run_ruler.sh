#!/usr/bin/env bash
#
# Unified RULER experiment launcher.
# Usage:
#   bash run_ruler.sh <model>
#
# Supported models: llama8b, llama70b, mistral7b
# Override any variable:  METHODS="SnapKV" bash run_ruler.sh llama8b
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Model selection
MODEL="${1:-llama8b}"

case "${MODEL}" in
    llama8b)
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
        MODEL_PATH="${MODEL_PATH:-/path/to/Llama-3.1-8B-Instruct}"
        RESULT_ROOT="${RESULT_ROOT:-./results_ruler_Llama8B}"
        ;;
    llama70b)
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
        MODEL_PATH="${MODEL_PATH:-/path/to/Llama-3.1-70B-Instruct}"
        RESULT_ROOT="${RESULT_ROOT:-./results_ruler_Llama70B}"
        ;;
    mistral7b)
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
        MODEL_PATH="${MODEL_PATH:-/path/to/Mistral-7B-Instruct-v0.3}"
        RESULT_ROOT="${RESULT_ROOT:-./results_ruler_Mistral}"
        ;;
    *)
        echo "Unknown model: ${MODEL}"
        echo "Supported: llama8b, llama70b, mistral7b"
        exit 1
        ;;
esac

export CUDA_VISIBLE_DEVICES

# Experiment hyper-parameters
METHODS=("${METHODS[@]:-championkv}")
MAX_CAPACITY_PROMPTS_RATIO=("${MAX_CAPACITY_PROMPTS_RATIO[@]:-0.2}")
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"

BETA="${BETA:-20}"
GQA_SUPPORT="${GQA_SUPPORT:-true}"
GQA_FUNC="${GQA_FUNC:-mean}"
WINDOW_SIZE="${WINDOW_SIZE:-8}"
ALLOCATE_STRATEGIES=("${ALLOCATE_STRATEGIES[@]:-pyramid}")
KERNEL_SIZE="${KERNEL_SIZE:-7}"
POOLING="${POOLING:-avgpool}"

INSPECT_ATTENTION_WEIGHT="${INSPECT_ATTENTION_WEIGHT:-false}"
ATTENTION_LOG_FILE="${ATTENTION_LOG_FILE:-${RESULT_ROOT%/}/Attention_weight.log}"

# Compute total runs
TOTAL_RUNS=0
for METHOD in "${METHODS[@]}"; do
    if [[ "${METHOD,,}" == "fullkv" ]]; then
        RATIO_RUNS=1
    else
        RATIO_RUNS=${#MAX_CAPACITY_PROMPTS_RATIO[@]}
    fi
    if [[ "${METHOD,,}" == "championkv" ]]; then
        ALLOCATE_RUNS=${#ALLOCATE_STRATEGIES[@]}
    else
        ALLOCATE_RUNS=1
    fi
    TOTAL_RUNS=$((TOTAL_RUNS + RATIO_RUNS * ALLOCATE_RUNS))
done

# Print configuration
echo "========== RULER: ${MODEL} =========="
echo "GPU:            ${CUDA_VISIBLE_DEVICES}"
echo "model_path:     ${MODEL_PATH}"
echo "result_root:    ${RESULT_ROOT}"
echo "methods (${#METHODS[@]}):      ${METHODS[*]}"
echo "ratios (${#MAX_CAPACITY_PROMPTS_RATIO[@]}):       ${MAX_CAPACITY_PROMPTS_RATIO[*]}"
echo "attn:           ${ATTN_IMPLEMENTATION}"
echo "beta:           ${BETA}"
echo "gqa:            ${GQA_SUPPORT} / ${GQA_FUNC}"
echo "window_size:    ${WINDOW_SIZE}"
echo "kernel_size:    ${KERNEL_SIZE}"
echo "pooling:        ${POOLING}"
echo "alloc_strategies (${#ALLOCATE_STRATEGIES[@]}): ${ALLOCATE_STRATEGIES[*]}"
echo "total runs:     ${TOTAL_RUNS}"
echo "========================================="

# Run experiments
for METHOD in "${METHODS[@]}"; do
    METHOD_RATIOS=("${MAX_CAPACITY_PROMPTS_RATIO[@]}")
    [[ "${METHOD,,}" == "fullkv" ]] && METHOD_RATIOS=(1.0)

    METHOD_ALLOCATE_STRATEGIES=("")
    [[ "${METHOD,,}" == "championkv" ]] && METHOD_ALLOCATE_STRATEGIES=("${ALLOCATE_STRATEGIES[@]}")

    for ALLOCATE_STRATEGY in "${METHOD_ALLOCATE_STRATEGIES[@]}"; do
        for RATIO in "${METHOD_RATIOS[@]}"; do
            METHOD_ARGS=()
            if [[ "${METHOD,,}" == "championkv" ]]; then
                METHOD_ARGS+=(--window_size "${WINDOW_SIZE}")
                METHOD_ARGS+=(--kernel_size "${KERNEL_SIZE}")
                METHOD_ARGS+=(--pooling "${POOLING}")
                METHOD_ARGS+=(--allocate_strategy "${ALLOCATE_STRATEGY}")
                METHOD_ARGS+=(--gqa_func "${GQA_FUNC}")
                [[ "${GQA_SUPPORT}" == "true" ]] && METHOD_ARGS+=(--gqa_support)
                [[ "${INSPECT_ATTENTION_WEIGHT}" == "true" ]] && \
                    METHOD_ARGS+=(--inspect_attention_weight --attention_log_file "${ATTENTION_LOG_FILE}")
            fi

            RATIO_TAG="${RATIO}"
            [[ "${METHOD,,}" == "fullkv" ]] && RATIO_TAG="100"
            BASE_TAG="${METHOD}_ratio${RATIO_TAG}"
            if [[ "${METHOD,,}" == "championkv" ]]; then
                CONFIG_TAG="${BASE_TAG}_beta${BETA}_ws${WINDOW_SIZE}_ks${KERNEL_SIZE}_${POOLING}_alloc${ALLOCATE_STRATEGY}_gqa${GQA_SUPPORT}_${GQA_FUNC}"
            else
                CONFIG_TAG="${BASE_TAG}"
            fi
            SAVE_DIR="${RESULT_ROOT%/}/${CONFIG_TAG}"

            echo "[RUN] method=${METHOD} ratio=${RATIO} alloc=${ALLOCATE_STRATEGY:-<none>} -> ${SAVE_DIR}"

            python3 "${REPO_ROOT}/run_ruler.py" \
                --method "${METHOD}" \
                --model_path "${MODEL_PATH}" \
                --max_capacity_prompts_ratio "${RATIO}" \
                --attn_implementation "${ATTN_IMPLEMENTATION}" \
                --save_dir "${SAVE_DIR}" \
                --use_cache True \
                --beta "${BETA}" \
                "${METHOD_ARGS[@]}"
        done
    done
done



