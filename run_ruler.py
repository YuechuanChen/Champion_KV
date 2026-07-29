import os
import json
import random
import argparse

import sys

def _print_lightweight_help():
    print("usage: run_ruler.py [options]\n")
    print("Key options:")
    print("  --method {fullkv,pyramidkv,snapkv,adaptivekv,adakv,championkv} (default: championkv)")
    print("  --model_path MODEL_PATH")
    print("  --data_file DATA_FILE")
    print("  --dataset DATASET")
    print("  --save_dir SAVE_DIR")
    print("  --max_capacity_prompts N (default: 512)")
    print("  --max_capacity_prompts_ratio R (default: -1)")
    print("  --window_size N (default: 8)")
    print("  --kernel_size N (default: 7)")
    print("  --pooling {maxpool,avgpool}")
    print("  --allocate_strategy {pyramid,uniform} (default: pyramid)")
    print("  --gqa_support --gqa_func {max,mean}")
    print("  --attn_implementation {flash_attention_2,sdpa,eager}")

if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    _print_lightweight_help()
    raise SystemExit(0)

import numpy as np
import torch

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
from KV_Cache_Utils.kv_cache_compression_utils import kv_method_requires_batch_invariant_eval


context_length_list = [16384]
# context_length_list = [4096, 8192, 16384]

datasets = ["cwe", "fwe"]

"""
datasets = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
            "niah_multiquery", "niah_multivalue", "cwe", "fwe", "vt"]
"""

dataset2maxlen = {
    "niah_single_1": 64,
    "niah_single_2": 64,
    "niah_single_3": 64,
    "niah_multikey_1": 64,
    "niah_multikey_2": 64,
    "niah_multikey_3": 64,
    "niah_multiquery": 64,
    "niah_multivalue": 64,
    "cwe": 64,
    "fwe": 64,
    "vt": 64
}


model2maxlen = {
    "llama2": 3950,
    "llama-2": 3950,
    "llama3": 7950,
    "llama-3": 7950,
    "mistral": 31500
}



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def build_chat(prompt):
        prompt = f"[INST] {prompt} [/INST]"
        return prompt

def _get_model_input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cuda")

def _truncate_prompt_ids(prompt_ids, max_len):
    if len(prompt_ids) <= max_len:
        return prompt_ids
    half = max_len // 2
    return prompt_ids[:half] + prompt_ids[-half:]

def _pad_batch(prompt_ids_list, pad_token_id, padding_side="left", device=None):
    batch_size = len(prompt_ids_list)
    max_len = max(len(ids) for ids in prompt_ids_list)
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    lengths = []

    for i, ids in enumerate(prompt_ids_list):
        if not torch.is_tensor(ids):
            ids = torch.tensor(ids, dtype=torch.long)
        seq_len = ids.numel()
        lengths.append(seq_len)
        if padding_side == "left":
            input_ids[i, -seq_len:] = ids
            attention_mask[i, -seq_len:] = 1
        else:
            input_ids[i, :seq_len] = ids
            attention_mask[i, :seq_len] = 1

    if device is not None:
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)

    return input_ids, attention_mask, lengths

def _build_padding_aware_batches(
    indices,
    prompt_lens,
    eval_batch_size,
    max_batch_tokens=-1,
    max_padding_ratio=0.0,
    sort_by_length=False,
):
    if sort_by_length:
        indices = sorted(indices, key=lambda i: prompt_lens[i])

    batches = []
    batch_indices = []
    batch_max_len = 0
    batch_token_sum = 0

    for idx in indices:
        seq_len = prompt_lens[idx]
        next_max_len = max(batch_max_len, seq_len)
        next_size = len(batch_indices) + 1
        next_token_sum = batch_token_sum + seq_len
        next_padded_tokens = next_max_len * next_size

        exceeds_size = next_size > eval_batch_size
        exceeds_tokens = max_batch_tokens > 0 and next_padded_tokens > max_batch_tokens
        exceeds_padding = (
            max_padding_ratio > 0
            and len(batch_indices) > 0
            and next_padded_tokens / max(next_token_sum, 1) > max_padding_ratio
        )

        if batch_indices and (exceeds_size or exceeds_tokens or exceeds_padding):
            batches.append(batch_indices)
            batch_indices = []
            batch_max_len = 0
            batch_token_sum = 0

        batch_indices.append(idx)
        batch_max_len = max(batch_max_len, seq_len)
        batch_token_sum += seq_len

    if batch_indices:
        batches.append(batch_indices)

    return batches

def _summarize_batch_padding(batches, prompt_lens):
    if not batches:
        return {
            "num_batches": 0,
            "effective_tokens": 0,
            "padded_tokens": 0,
            "padding_tokens": 0,
            "padding_ratio": 1.0,
            "max_batch_size": 0,
            "max_batch_len": 0,
        }

    effective_tokens = 0
    padded_tokens = 0
    max_batch_size = 0
    max_batch_len = 0
    for batch in batches:
        batch_lens = [prompt_lens[i] for i in batch]
        batch_max_len = max(batch_lens)
        effective_tokens += sum(batch_lens)
        padded_tokens += batch_max_len * len(batch)
        max_batch_size = max(max_batch_size, len(batch))
        max_batch_len = max(max_batch_len, batch_max_len)

    padding_tokens = padded_tokens - effective_tokens
    return {
        "num_batches": len(batches),
        "effective_tokens": effective_tokens,
        "padded_tokens": padded_tokens,
        "padding_tokens": padding_tokens,
        "padding_ratio": padded_tokens / max(effective_tokens, 1),
        "max_batch_size": max_batch_size,
        "max_batch_len": max_batch_len,
    }

def load_completed_result_keys(result_path):
    completed_indices = set()
    completed_prompts = set()
    completed_count = 0

    if not os.path.exists(result_path):
        return completed_indices, completed_prompts, completed_count

    with open(result_path) as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                print(f"[Resume] Skip malformed result line {line_no}: {result_path}")
                continue

            if "index" in result:
                completed_indices.add(result["index"])
            if "prompt" in result:
                completed_prompts.add(result["prompt"])
            elif "input" in result:
                completed_prompts.add(result["input"])
            completed_count += 1

    return completed_indices, completed_prompts, completed_count

def ensure_trailing_newline(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return
    with open(path, "rb") as fin:
        fin.seek(-1, os.SEEK_END)
        has_newline = fin.read(1) == b"\n"
    if not has_newline:
        with open(path, "a") as fout:
            fout.write("\n")

# def build_prompt(prompt, dataset):
    
#     SYSTEM_PROMPT = model2prompt[dataset]

#     prompt = f"<<SYS>>\n {SYSTEM_PROMPT} \n<</SYS>>\n\n{prompt}"
#     return prompt

def main(args):
    

    print("Loading data...")
    
    test_data = []
    prompt_list = []
    input_list = []
    outputs_list: List[List[str]] = [] # List of List
    length_list = []
    index_list = []
    
    input_max_len = 0
    model_path = args.model_path.lower()
    
    model_max_len = None
    for key in model2maxlen:
        if key in model_path:
            model_max_len = model2maxlen[key]
    if model_max_len is None:
        raise ValueError(f"Cannot infer model_max_len from model_path={args.model_path}")
    
    output_max_len = dataset2maxlen[args.dataset]
    
    with open(args.data_file) as fp:
        for line in fp:

            example = json.loads(line)
            length = example["length"]
            if length > input_max_len: 
                input_max_len = length

            prompt = example["input"] #TODO tokenizer.apply_chat_template ?
            if "llama2" in args.model_path.lower():
                prompt = build_chat(prompt)
            example["prompt"] = prompt
                
            test_data.append(example)
        
    print(f"Max Length is {input_max_len}")
        
    if args.max_num_examples and len(test_data) > args.max_num_examples:
        if args.sample_method == "random":
            test_data = random.sample(test_data, args.max_num_examples)
        elif args.sample_method == "topk":
            test_data = test_data[:args.max_num_examples]
    
    print("Finish loading model and tokenizer")
    method_name = args.method.lower()
    if method_name == "fullkv":
        args.max_capacity_prompts_ratio = 1.0

    result_dir = os.path.join(args.save_dir, str(args.context_length), args.dataset)
    result_path = os.path.join(result_dir, f"{args.method}.json")
    os.makedirs(result_dir, exist_ok=True)

    if args.overwrite and os.path.exists(result_path):
        os.remove(result_path)

    completed_indices, completed_prompts, completed_count = load_completed_result_keys(result_path)
    skipped_count = 0

    prompt_ids_list = []
    for example in test_data:
        example_index = example["index"]
        prompt = example["prompt"]
        if example_index in completed_indices or prompt in completed_prompts:
            skipped_count += 1
            continue

        prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
        if prompt_ids is None:
            raise ValueError("Tokenizer returned None for input_ids.")
        if len(prompt_ids) > model_max_len:
            prompt_ids = _truncate_prompt_ids(prompt_ids, model_max_len)
            prompt = tokenizer.decode(prompt_ids, skip_special_tokens=True)

        prompt_list.append(prompt)
        prompt_ids_list.append(prompt_ids)
        input_list.append(example["input"])
        outputs_list.append(example["outputs"])
        length_list.append(example["length"])
        index_list.append(example_index)

    if completed_count:
        print(
            f"[Resume] found {completed_count} existing results in {result_path}; "
            f"skip {skipped_count}, pending {len(prompt_list)}"
        )
    if not prompt_list:
        print(f"[Resume] all examples already finished: {result_path}")
        return

    ensure_trailing_newline(result_path)
    fout = open(result_path, "a", buffering=1)

    batch_invariant_kv = kv_method_requires_batch_invariant_eval(method_name)
    batch_invariant_notice_shown = False

    def run_batch(batch_indices):
        nonlocal batch_invariant_notice_shown

        if batch_invariant_kv and len(batch_indices) > 1:
            if not batch_invariant_notice_shown:
                print(
                    f"[BatchInvariantKV] method={args.method} uses eval_batch_size={args.eval_batch_size} "
                    "for scheduling, but physical KV-compressed generation is run per prompt "
                    "to keep results identical to eval_batch_size=1."
                )
                batch_invariant_notice_shown = True
            for single_index in batch_indices:
                run_batch([single_index])
            return

        batch_prompts = [prompt_list[i] for i in batch_indices]
        batch_inputs = [input_list[i] for i in batch_indices]
        batch_answers = [outputs_list[i] for i in batch_indices]
        batch_lengths = [length_list[i] for i in batch_indices]
        batch_example_indices = [index_list[i] for i in batch_indices]
        batch_prompt_ids = [prompt_ids_list[i] for i in batch_indices]
        
        batch_input_ids, attention_mask, _ = _pad_batch(
            batch_prompt_ids,
            tokenizer.pad_token_id,
            padding_side=tokenizer.padding_side,
            device=input_device,
        )
        tokenized_prompts = {"input_ids": batch_input_ids, "attention_mask": attention_mask}

        if args.max_capacity_prompts_ratio != -1:
            max_capacity_prompts = max(1, round(batch_input_ids.shape[1] * args.max_capacity_prompts_ratio))
        elif args.max_capacity_prompts != -1:
            max_capacity_prompts = args.max_capacity_prompts
        else:
            max_capacity_prompts = args.max_capacity_prompts
        
        
        if method_name != "fullkv":
            layers = len(model.model.layers)

            if method_name in ["streamingllm"]:
                window_sizes = max(1, max_capacity_prompts - 4)
            else:
                window_sizes = args.window_size

            kernel_sizes = args.kernel_size
            pooling = args.pooling

            if not isinstance(max_capacity_prompts, list):
                max_capacity_prompts = [max_capacity_prompts] * layers
            if not isinstance(kernel_sizes, list):
                kernel_sizes = [kernel_sizes] * layers
            if not isinstance(window_sizes, list):
                window_sizes = [window_sizes] * layers

            for layer_idx in range(layers):
                model.model.layers[layer_idx].self_attn.config.window_size = window_sizes[layer_idx]
                model.model.layers[layer_idx].self_attn.config.max_capacity_prompt = max_capacity_prompts[layer_idx]
                model.model.layers[layer_idx].self_attn.config.kernel_size = kernel_sizes[layer_idx]
                model.model.layers[layer_idx].self_attn.config.pooling = pooling
                model.model.layers[layer_idx].self_attn.config.kv_method = method_name
                model.model.layers[layer_idx].self_attn.config.beta = args.beta
                model.model.layers[layer_idx].self_attn.config.floor = args.floor
                model.model.layers[layer_idx].self_attn.config.floor_ratio = args.floor
                model.model.layers[layer_idx].self_attn.config.floor_alpha = args.floor_alpha
                model.model.layers[layer_idx].self_attn.config.skip = args.skip
                model.model.layers[layer_idx].self_attn.config.normalize = args.normalize
                model.model.layers[layer_idx].self_attn.config.gqa_support = args.gqa_support
                model.model.layers[layer_idx].self_attn.config.gqa_func = args.gqa_func
                model.model.layers[layer_idx].self_attn.config.inspect_attention_weight = args.inspect_attention_weight
                model.model.layers[layer_idx].self_attn.config.attention_log_file = args.attention_log_file
                if method_name == "championkv":
                    model.model.layers[layer_idx].self_attn.config.allocate_strategy = args.allocate_strategy

        context_length = batch_input_ids.shape[-1]
        with torch.inference_mode():
            if args.quant_method == None:        
                output = model.generate(
                    **tokenized_prompts,
                    output_attentions = args.output_attentions,
                    max_new_tokens=output_max_len,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    min_length=context_length+1,
                    eos_token_id=[tokenizer.eos_token_id]
                )
            else:
                output = model.generate(
                    **tokenized_prompts,
                    output_attentions = args.output_attentions,
                    max_new_tokens=output_max_len,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    min_length=context_length+1,
                    eos_token_id=[tokenizer.eos_token_id],
                    cache_implementation="quantized", 
                    cache_config={"nbits": args.nbits, "backend": "HQQ","device":"cuda","residual_length":output_max_len,"axis_key":1,"q_group_size":64},
                )

        batch_generations = tokenizer.batch_decode(output[:, context_length:], skip_special_tokens=True)

        torch.cuda.empty_cache()
        
        for j in range(len(batch_prompts)):
            
            example = {}
            example["index"] = batch_example_indices[j]
            example["prompt"] = batch_prompts[j]
            example["input"] = batch_inputs[j]
            example["answers"] = batch_answers[j]
            example["pred"] = batch_generations[j]
            example["length"] = batch_lengths[j]

            fout.write(json.dumps(example) + "\n")

    prompt_lens = [len(prompt_ids) for prompt_ids in prompt_ids_list]
    indices = list(range(len(prompt_list)))
    optimize_padding = not args.disable_padding_optimization
    batches = _build_padding_aware_batches(
        indices,
        prompt_lens,
        eval_batch_size=args.eval_batch_size,
        max_batch_tokens=args.max_batch_tokens,
        max_padding_ratio=args.max_padding_ratio if optimize_padding else 0.0,
        sort_by_length=args.sort_by_length or optimize_padding,
    )
    padding_stats = _summarize_batch_padding(batches, prompt_lens)
    print(
        "[BatchPlan] "
        f"batches={padding_stats['num_batches']}, "
        f"max_batch_size={padding_stats['max_batch_size']}, "
        f"max_batch_len={padding_stats['max_batch_len']}, "
        f"effective_tokens={padding_stats['effective_tokens']}, "
        f"padded_tokens={padding_stats['padded_tokens']}, "
        f"padding_tokens={padding_stats['padding_tokens']}, "
        f"padding_ratio={padding_stats['padding_ratio']:.3f}, "
        f"max_padding_ratio={args.max_padding_ratio if optimize_padding else 'disabled'}"
    )

    pbar = tqdm(total=len(prompt_list))
    for batch_indices in batches:
        run_batch(batch_indices)
        pbar.update(len(batch_indices))

    pbar.close()

    fout.close()
    
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--base_dir", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--data_file", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing prediction files instead of resuming")

    parser.add_argument("--model_name", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--model_path", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--use_fast_tokenizer", type=bool, default=True, help="")
    parser.add_argument("--output_attentions", type=bool, default=False, help="")
    
    parser.add_argument("--max_num_examples", type=int, default=None, help="maximum number of examples to evaluate per task.")
    parser.add_argument("--sample_method", type=str, default="topk", choices=["random", "topk"], help="how to sample the examples.")
    
    parser.add_argument("--max_new_tokens", type=int, default=None, help="")
    
    parser.add_argument("--eval_batch_size", type=int, nargs="+", default=[1], help="batch size(s) for evaluation.")
    parser.add_argument("--max_batch_tokens", type=int, default=-1, help="if >0, cap batch by max_len * batch_size.")
    parser.add_argument("--sort_by_length", action="store_true", help="sort examples by prompt length to reduce padding.")
    parser.add_argument(
        "--max_padding_ratio",
        type=float,
        default=1.20,
        help=(
            "When padding optimization is enabled, close the current batch if adding the next sample would make "
            "padded_tokens / effective_tokens exceed this ratio. Use <=0 to disable this extra split rule."
        ),
    )
    parser.add_argument(
        "--disable_padding_optimization",
        action="store_true",
        help="disable default length-aware batch planning and use the original sequential batching behavior.",
    )
    
    parser.add_argument("--use_cache", type=bool, default=True, help="")
    parser.add_argument("--attn_implementation", type=str,  default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--method", type=str, default="championkv", choices=["fullkv", "pyramidkv", "snapkv", "adaptivekv", "adakv", "championkv"], help="KV cache compression method.")
    parser.add_argument("--quant_method",type=str,default=None,choices=["kivi","kvquant"])
    parser.add_argument("--nbits", type=int, default=8, help="")
    parser.add_argument("--max_capacity_prompts", type=int, default=512, help="")
    parser.add_argument("--max_capacity_prompts_ratio", type=float, default=-1, help="")
    parser.add_argument("--window_size", type=int, default=8, help="history window size for KV compression methods")
    parser.add_argument("--kernel_size", type=int, default=7, help="pooling kernel size for attention score smoothing")
    parser.add_argument("--pooling", type=str, default="maxpool", choices=["maxpool", "avgpool"], help="pooling method")
    parser.add_argument("--beta", type=int, default=20, help="pyramid allocation beta used by AdaptiveKV variants")
    parser.add_argument("--floor", type=float, default=0.2, help="floor ratio used by AdaKV")
    parser.add_argument("--floor_alpha", type=float, default=0.2, help="floor ratio used by AdaptiveKV")
    parser.add_argument("--skip", type=int, default=0, help="layer skip threshold used by AdaptiveKV")
    parser.add_argument("--normalize", type=bool, default=True, help="whether to enable score normalization for AdaKV/AdaptiveKV")
    parser.add_argument("--gqa_support", action="store_true", help="enable GQA path in adaptive KV methods")
    parser.add_argument("--gqa_func", type=str, default="mean", choices=["max", "mean"], help="aggregation in GQA path")
    parser.add_argument(
        "--allocate_strategy",
        type=str,
        default="pyramid",
        choices=["pyramid", "uniform"],
        help="ChampionKV layer budget strategy",
    )
    parser.add_argument("--inspect_attention_weight", action="store_true", help="log per-head history attention means")
    parser.add_argument("--attention_log_file", type=str, default=None, help="optional path for attention logs")
    parser.add_argument("--data_root", type=str, default="./data", help="root path of evaluation data")
    parser.add_argument("--steps", type=int, default=-1, help="maximum number of examples to evaluate per task.")
    
    parser.add_argument(
        "--use_chat_format", 
        action="store_true", 
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function", 
        type=str, 
        default="eval.templates.create_prompt_with_tulu_chat_format", 
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    
    args = parser.parse_args()

    if args.attention_log_file is None:
        args.attention_log_file = os.path.join(args.save_dir, "Attention_weight.log")

    if args.inspect_attention_weight:
        os.makedirs(os.path.dirname(args.attention_log_file), exist_ok=True)
        if os.path.exists(args.attention_log_file):
            os.remove(args.attention_log_file)
    
    set_seed(args.seed)
    if args.quant_method == "kvquant":
        from KV_Cache_Utils.quantcache import KVQuantizedCache
        from transformers import cache_utils
        cache_utils.HQQQuantizedCache = KVQuantizedCache
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=args.use_fast_tokenizer,
        padding_side="left"
    )


    from KV_Cache_Utils.monkeypatch import replace_llama,replace_mistral
    replace_llama(args.method.lower())
    replace_mistral(args.method.lower())
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_cache=args.use_cache,
        attn_implementation=args.attn_implementation
    )
    print(f"[Model] Running inference with {model.config.model_type}: {args.model_path}")
    if hasattr(model, "hf_device_map"):
        print(f"Model hf_device_map: {model.hf_device_map}")
    input_device = _get_model_input_device(model)
    print(f"Model input device: {input_device}")
    
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model.eval()
    
    base_save_dir = args.save_dir
    eval_batch_sizes = args.eval_batch_size

    for batch_size in eval_batch_sizes:
        args.eval_batch_size = int(batch_size)
        if len(eval_batch_sizes) > 1:
            args.save_dir = os.path.join(base_save_dir, f"bsz{args.eval_batch_size}")
        else:
            args.save_dir = base_save_dir

        for context_length in context_length_list:
            for idx, dataset in enumerate(datasets):

                print(
                    f"Working on context length {context_length}, "
                    f"eval_batch_size: {args.eval_batch_size}, "
                    f"max_capacity_prompts: {args.max_capacity_prompts}, "
                    f"dataset: {dataset} - {idx}/{len(datasets)}"
                )
                args.context_length = context_length
                args.dataset = dataset
                args.data_file = os.path.join(args.data_root, "RULER", str(context_length), f"{args.dataset}.jsonl")

                main(args)

