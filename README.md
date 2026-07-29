# ChampionKV

Official implementation of **ChampionKV: Champion-level Key-Value Cache Compression for Long-Context LLM Inference**.

ChampionKV is an inference-time KV cache compression method for long-context large language models. It does not train or modify model weights. Instead, it patches Hugging Face Transformers attention forwards at runtime and keeps a compact KV cache during generation.

## Highlights

- Adaptive KV cache compression for long-context inference.
- Layer-level budget allocation with `pyramid` or `uniform` strategies.
- Head-level budget allocation driven by attention-score patterns.
- Support for LLaMA and Mistral model families.
- Integration with FlashAttention 2 variable-length attention for compressed decode.
- Evaluation scripts for LongBench, RULER, and Needle-in-a-Haystack.

## Supported KV Methods

The public release supports the following methods:

- `fullkv`: no KV cache compression baseline.
- `pyramidkv`: layer-wise pyramid budget allocation with top-k token selection.
- `snapkv`: uniform budget with recent-window attention scoring.
- `adaptivekv`: adaptive per-head budget allocation.
- `adakv`: AdaKV-style per-head adaptive budget with floor capacity.
- `championkv`: ChampionKV attention-max-driven head budget allocation.

## Repository Structure

```text
ChampionKV/
|-- README.md
|-- LICENSE
|-- requirements.txt
|-- KV_Cache_Utils/                 # Source-tree import compatibility package
|-- src/
|   `-- KV_Cache_Utils/
|       |-- kv_cache_compression_utils.py  # Core KV compression algorithms
|       |-- llama_model.py                 # LLaMA forward overrides
|       |-- mistral_model.py               # Mistral forward overrides
|       |-- monkeypatch.py                 # Runtime Transformers patching
|       |-- quantcache.py                  # Optional KV cache quantization support
|       `-- minference.py                  # Optional inference helper
|-- scripts/
|   |-- run_longbench.py
|   |-- run_ruler.py
|   |-- run_needle_in_haystack.py
|   |-- eval_longbench.py
|   |-- eval_ruler.py
|   |-- metrics.py
|   `-- exp_scripts/
|       |-- LongBench/run_longbench.sh
|       |-- Ruler/run_ruler.sh
|       `-- Needle/run_needle.sh
|-- figures/
|   `-- plot_ruler.py
|-- results/
|   `-- README.md
`-- tests/
    `-- test_open_source_hygiene.py
```

## Installation

```bash
pip install -r requirements.txt
```

`flash-attn` requires a compatible CUDA/PyTorch environment. If FlashAttention is unavailable, try `--attn_implementation sdpa` or `--attn_implementation eager` for supported methods, but the optimized ChampionKV decode path is intended for FlashAttention 2.

## Quick Start

Use local model paths or Hugging Face model identifiers through `--model_path` / `--model_name`. The `/path/to/...` values below are placeholders, not required directory names.

### LongBench

```bash
python scripts/run_longbench.py \
    --method championkv \
    --model_path /path/to/Llama-3.1-8B-Instruct \
    --data_file /path/to/LongBench/narrativeqa.jsonl \
    --dataset narrativeqa \
    --max_capacity_prompts 512 \
    --window_size 8 \
    --allocate_strategy pyramid \
    --attn_implementation flash_attention_2 \
    --save_dir ./results_longbench
```

### RULER

```bash
python scripts/run_ruler.py \
    --method championkv \
    --model_path /path/to/Llama-3.1-8B-Instruct \
    --data_file /path/to/RULER/task.jsonl \
    --dataset niah_single_1 \
    --max_capacity_prompts 512 \
    --window_size 8 \
    --allocate_strategy pyramid \
    --attn_implementation flash_attention_2 \
    --save_dir ./results_ruler
```

### Needle-in-a-Haystack

```bash
python scripts/run_needle_in_haystack.py \
    --method championkv \
    --model_name /path/to/Llama-3.1-8B-Instruct \
    --model_provider LLaMA3 \
    --max_capacity_prompts 512 \
    --window_size 8 \
    --allocate_strategy pyramid \
    --s_len 1000 \
    --e_len 32000 \
    --step 1000
```

The legacy Needle argument `--max_capacity_prompt` remains accepted as an alias for `--max_capacity_prompts`.

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--method` | KV compression method | `championkv` |
| `--max_capacity_prompts` | KV cache budget per layer | `512` for LongBench/RULER, `128` for Needle |
| `--max_capacity_prompts_ratio` | Dynamic budget as a ratio of prompt length | `-1` (disabled) |
| `--window_size` | Recent-token window used for attention scoring | `8` |
| `--kernel_size` | Pooling kernel for attention-score smoothing | `7` |
| `--pooling` | Score pooling mode: `avgpool` or `maxpool` | benchmark-specific |
| `--allocate_strategy` | ChampionKV layer budget strategy: `pyramid` or `uniform` | `pyramid` |
| `--beta` | Pyramid allocation shape parameter | `20` |
| `--floor_alpha` | Floor ratio for AdaptiveKV-style methods | `0.2` |
| `--gqa_support` | Enable grouped-query-attention-aware compression | disabled |
| `--gqa_func` | GQA aggregation: `mean` or `max` | `mean` |
| `--attn_implementation` | Hugging Face attention backend | `flash_attention_2` |

## Reproducing Experiments

Set `MODEL_PATH` and optional output variables before running the launcher scripts.

```bash
# LongBench
MODEL_PATH=/path/to/Llama-3.1-8B-Instruct \
  bash scripts/exp_scripts/LongBench/run_longbench.sh llama8b

# RULER
MODEL_PATH=/path/to/Llama-3.1-8B-Instruct \
  bash scripts/exp_scripts/Ruler/run_ruler.sh llama8b

# Needle-in-a-Haystack
MODEL_NAME=/path/to/Llama-3.1-8B-Instruct \
  bash scripts/exp_scripts/Needle/run_needle.sh llama8b
```

Expected dataset locations can be overridden through script arguments or environment variables. Defaults are:

- LongBench: `data/LongBench/`
- RULER: `data/RULER/`
- Needle-in-a-Haystack: `data/PaulGrahamEssays/`

## Citation

Citation metadata will be updated after the paper metadata is finalized.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

This work builds upon several open-source projects, including Hugging Face Transformers, FlashAttention, AdaKV, SnapKV, and PyramidKV.
