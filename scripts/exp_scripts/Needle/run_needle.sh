#!/usr/bin/env bash
#
# Unified Needle-in-a-Haystack experiment launcher.
# Usage:
#   bash run_needle.sh <model> [model_provider]
#
# Supported models: llama8b, llama70b, mistral7b
# Default model_provider: LLaMA3 (use Mistral for mistral7b)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
NEEDLE_RUNNER="${REPO_ROOT}/run_needle_in_haystack.py"

# Model selection
MODEL="${1:-llama8b}"

case "${MODEL}" in
    llama8b)
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
        MODEL_NAME="${MODEL_NAME:-/path/to/Llama-3.1-8B-Instruct}"
        MODEL_PROVIDER="${MODEL_PROVIDER:-LLaMA3}"
        ;;
    llama70b)
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
        MODEL_NAME="${MODEL_NAME:-/path/to/Llama-3.1-70B-Instruct}"
        MODEL_PROVIDER="${MODEL_PROVIDER:-LLaMA3}"
        ;;
    mistral7b)
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
        MODEL_NAME="${MODEL_NAME:-/path/to/Mistral-7B-Instruct-v0.3}"
        MODEL_PROVIDER="${MODEL_PROVIDER:-Mistral}"
        ;;
    *)
        echo "Unknown model: ${MODEL}"
        echo "Supported: llama8b, llama70b, mistral7b"
        exit 1
        ;;
esac

export CUDA_VISIBLE_DEVICES
export ADAKV_USE_TINY_API_CUDA="${ADAKV_USE_TINY_API_CUDA:-1}"
export CHAMPIONKV_ROOT="${REPO_ROOT}"

mkdir -p "${REPO_ROOT}/results_needle/logs/"
mkdir -p "${REPO_ROOT}/results_needle/img/"
mkdir -p "${REPO_ROOT}/results_needle/performance/"

# Experiment hyper-parameters
METHOD="${METHOD:-championkv}"
METHOD_LC="${METHOD,,}"
MAX_CAPACITY_PROMPTS=("${MAX_CAPACITY_PROMPTS[@]:-1024}")
S_LEN="${S_LEN:-8000}"
E_LEN="${E_LEN:-128001}"
STEP="${STEP:-8000}"

ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
BETA="${BETA:-20}"
GQA_SUPPORT="${GQA_SUPPORT:-true}"
GQA_FUNC="${GQA_FUNC:-mean}"
WINDOW_SIZE="${WINDOW_SIZE:-8}"
KERNEL_SIZE="${KERNEL_SIZE:-7}"
POOLING="${POOLING:-avgpool}"
ALLOCATE_STRATEGY="${ALLOCATE_STRATEGY:-pyramid}"
MERGE_METHOD="${MERGE_METHOD:-}"

INSPECT_ATTENTION_WEIGHT="${INSPECT_ATTENTION_WEIGHT:-false}"
ATTENTION_LOG_FILE="${ATTENTION_LOG_FILE:-${REPO_ROOT}/results_needle/Attention_weight.log}"

ENABLE_PERFORMANCE_PROFILE="${ENABLE_PERFORMANCE_PROFILE:-true}"
RECORD_PERFORMANCE="${RECORD_PERFORMANCE:-true}"
PERFORMANCE_LOG_DIR="${PERFORMANCE_LOG_DIR:-${REPO_ROOT}/results_needle/performance}"
PERFORMANCE_LOG_FILE="${PERFORMANCE_LOG_FILE:-}"

MODEL_NAME_SUFFIX="${MODEL_NAME_SUFFIX:-}"
HAYSTACK_DIR="${HAYSTACK_DIR:-data/PaulGrahamEssays}"

TOTAL_RUNS=${#MAX_CAPACITY_PROMPTS[@]}

# Print configuration
echo "========== Needle: ${MODEL} =========="
echo "GPU:            ${CUDA_VISIBLE_DEVICES}"
echo "model:          ${MODEL_NAME}"
echo "provider:       ${MODEL_PROVIDER}"
echo "haystack_dir:   ${HAYSTACK_DIR}"
echo "method:         ${METHOD}"
echo "capacities (${#MAX_CAPACITY_PROMPTS[@]}):   ${MAX_CAPACITY_PROMPTS[*]}"
echo "context:        ${S_LEN} -> ${E_LEN}, step=${STEP}"
echo "attn:           ${ATTN_IMPLEMENTATION}"
echo "beta:           ${BETA}"
echo "gqa:            ${GQA_SUPPORT} / ${GQA_FUNC}"
echo "window/ks/pool: ${WINDOW_SIZE} / ${KERNEL_SIZE} / ${POOLING}"
echo "allocate:       ${ALLOCATE_STRATEGY}"
echo "merge:          ${MERGE_METHOD:-<none>}"
echo "perf_profile:   ${ENABLE_PERFORMANCE_PROFILE}"
echo "record_perf:    ${RECORD_PERFORMANCE}"
echo "perf_log_dir:   ${PERFORMANCE_LOG_DIR}"
echo "suffix:         ${MODEL_NAME_SUFFIX:-<none>}"
echo "total runs:     ${TOTAL_RUNS}"
echo "========================================="

# Run experiments
for MAX_CAPACITY_PROMPT in "${MAX_CAPACITY_PROMPTS[@]}"; do
    MODEL_NAME_SUFFIX_ARG=()
    [[ -n "${MODEL_NAME_SUFFIX}" ]] && MODEL_NAME_SUFFIX_ARG=(--model_name_suffix "${MODEL_NAME_SUFFIX}")

    COMMON_ARGS=(
        --s_len "${S_LEN}"
        --e_len "${E_LEN}"
        --model_provider "${MODEL_PROVIDER}"
        --model_name "${MODEL_NAME}"
        --haystack_dir "${HAYSTACK_DIR}"
        --attn_implementation "${ATTN_IMPLEMENTATION}"
        --step "${STEP}"
        --method "${METHOD}"
        --max_capacity_prompts "${MAX_CAPACITY_PROMPT}"
        --window_size "${WINDOW_SIZE}"
        --kernel_size "${KERNEL_SIZE}"
        --pooling "${POOLING}"
        --beta "${BETA}"
        --gqa_func "${GQA_FUNC}"
        --allocate_strategy "${ALLOCATE_STRATEGY}"
    )

    [[ -n "${MERGE_METHOD}" ]] && COMMON_ARGS+=(--merge "${MERGE_METHOD}")
    [[ "${GQA_SUPPORT}" == "true" ]] && COMMON_ARGS+=(--gqa_support)
    [[ "${INSPECT_ATTENTION_WEIGHT}" == "true" ]] && \
        COMMON_ARGS+=(--inspect_attention_weight --attention_log_file "${ATTENTION_LOG_FILE}")

    PERFORMANCE_ARGS=(--performance_log_dir "${PERFORMANCE_LOG_DIR}")
    [[ "${ENABLE_PERFORMANCE_PROFILE}" == "true" ]] && PERFORMANCE_ARGS+=(--enable_performance_profile)
    [[ "${RECORD_PERFORMANCE}" == "true" ]] && PERFORMANCE_ARGS+=(--record_performance)
    [[ -n "${PERFORMANCE_LOG_FILE}" ]] && PERFORMANCE_ARGS+=(--performance_log_file "${PERFORMANCE_LOG_FILE}")

    if [[ "${METHOD_LC}" == "full" || "${METHOD_LC}" == "fullkv" ]]; then
        model_version="FullKV"
    else
        model_version="${METHOD_LC}_cap${MAX_CAPACITY_PROMPT}"
    fi
    run_name="${model_version}"
    [[ -n "${MODEL_NAME_SUFFIX}" ]] && run_name="${run_name}_${MODEL_NAME_SUFFIX}"

    echo "[RUN] method=${METHOD} cap=${MAX_CAPACITY_PROMPT}"

    (
    cd "${REPO_ROOT}"
    python -u "${NEEDLE_RUNNER}" \
        "${COMMON_ARGS[@]}" \
        "${MODEL_NAME_SUFFIX_ARG[@]}" \
        "${PERFORMANCE_ARGS[@]}" \
        --model_version "${model_version}"
    ) 2>&1 | tee "${REPO_ROOT}/results_needle/logs/${run_name}.log"
done


