import os
import json
import random
import argparse


import sys

def _print_lightweight_help():
    print("usage: run_longbench.py [options]\n")
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
    print("  --allocate_strategy {pyramid,uniform} (default: pyramid)")
    print("  --gqa_support --gqa_func {max,mean}")
    print("  --attn_implementation {flash_attention_2,flash_attention_3,sdpa,eager}")

if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    _print_lightweight_help()
    raise SystemExit(0)

import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from KV_Cache_Utils.kv_cache_compression_utils import kv_method_requires_batch_invariant_eval

datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
            "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", \
            "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]

'''
datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
            "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", \
            "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]

datasets = ["narrativeqa", "multifieldqa_en", "hotpotqa", "2wikimqa", \
            "gov_report", "trec", "triviaqa", "samsum", \
            "passage_retrieval_en", "lcc", "repobench-p"]

datasets = ["narrativeqa", "hotpotqa", "gov_report", "trec", "passage_count", "lcc"]
'''

dataset2maxlen = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64
}

model2prompt = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh": "Read the following Chinese text and answer the question briefly. Only output the answer.\n\n{context}\n\nQuestion: {input}\nAnswer:",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "dureader": "Read the following Chinese passage and answer the question based on the passage. Only output the answer.\n\nPassage: {context}\n\nQuestion: {input}\nAnswer:",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum": "Read the following Chinese meeting transcript and write a concise summary.\n\nTranscript:\n{context}\n\nSummary:",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht": "Classify the topic category of the given Chinese news text. The following are examples.\n\n{context}\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "passage_retrieval_zh": "Given several Chinese paragraphs and an abstract, identify which paragraph the abstract comes from. Answer with the paragraph number, such as Paragraph 1.\n\n{context}\n\nAbstract:\n{input}\n\nAnswer:",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n"
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


def _load_processed_ids(output_file):
    """Load already predicted sample ids from an existing jsonl result file."""
    processed_ids = set()
    malformed_lines = 0
    if not os.path.exists(output_file):
        return processed_ids, malformed_lines

    with open(output_file, "r") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                malformed_lines += 1
                continue
            if "_id" in item:
                processed_ids.add(str(item["_id"]))
    return processed_ids, malformed_lines


def ensure_trailing_newline(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return
    with open(path, "rb") as fin:
        fin.seek(-1, os.SEEK_END)
        has_newline = fin.read(1) == b"\n"
    if not has_newline:
        with open(path, "a") as fout:
            fout.write("\n")


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


def build_model_version(args):
    method_raw = args.method
    method = method_raw.lower()
    if method == "fullkv":
        capacity_tag = "full"
    else:
        capacity_tag = str(args.max_capacity_prompts)
    merge_tag = args.merge if args.merge not in [None, ""] else "none"
    merge_tag = str(merge_tag).replace("/", "-").replace(" ", "")

    config_tags = [f"cap{capacity_tag}"]

    if method == "championkv":
        config_tags.append(f"alloc{args.allocate_strategy}")

    config_tags.append(f"merge_{merge_tag}")
    config_tag = "_".join(config_tags)

    model_version = args.model_version
    if model_version is None:
        model_version = f"LongBench_{method_raw}_{config_tag}"
    if args.model_name_suffix is not None:
        model_version += "_" + args.model_name_suffix
    # if not model_version.endswith("_CV"):
    #     model_version += "_CV"
    return model_version.replace(".", "_")

# def build_prompt(prompt, dataset):
    
#     SYSTEM_PROMPT = model2prompt[dataset]

#     prompt = f"<<SYS>>\n {SYSTEM_PROMPT} \n<</SYS>>\n\n{prompt}"
#     return prompt

def main(args):
    

    print("Loading data...")
    
    test_data = []
    
    prompts = []
    inputs = []
    contexts = []
    answerss = []
    lengths = []
    datasets = []
    languages = []
    all_classess = []
    _ids = []
    
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
            if length > input_max_len: input_max_len = length
            
            template = model2prompt[args.dataset]
            prompt = template.format(**example)
            
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

    if args.inspect_attention_weight and len(test_data) > 1:
        test_data = [test_data[0]]
    
    
    output_file = os.path.join(args.save_dir, args.model_version, args.dataset, f"{args.method}.json")
    processed_ids, malformed_lines = _load_processed_ids(output_file)
    if malformed_lines > 0:
        print(f"[Resume] ignored {malformed_lines} malformed lines in existing result file: {output_file}")

    if processed_ids:
        before_count = len(test_data)
        test_data = [
            ex for ex in test_data
            if "_id" not in ex or str(ex["_id"]) not in processed_ids
        ]
        skipped_count = before_count - len(test_data)
        print(
            f"[Resume] found {len(processed_ids)} existing predictions, "
            f"skipped {skipped_count} already inferred samples."
        )

    if len(test_data) == 0:
        print(f"[Resume] all samples already inferred for dataset={args.dataset}. Nothing to run.")
        return

    prompt_ids_list = []
    prompt_lens = []
    for example in test_data:
        prompt_text = example["prompt"]
        prompt_ids = tokenizer(prompt_text, add_special_tokens=True).input_ids
        if prompt_ids is None:
            raise ValueError("Tokenizer returned None for input_ids.")
        if len(prompt_ids) > model_max_len:
            prompt_ids = _truncate_prompt_ids(prompt_ids, model_max_len)
            prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
            example["prompt"] = prompt_text

        prompt_ids_list.append(prompt_ids)
        prompt_lens.append(len(prompt_ids))

        prompts.append(prompt_text)
        inputs.append(example["input"])
        contexts.append(example["context"])
        answerss.append(example["answers"])
        lengths.append(example["length"])
        datasets.append(example["dataset"])
        languages.append(example["language"])
        all_classess.append(example["all_classes"])
        _ids.append(example["_id"])

    print("Finish loading model and tokenizer")
    
    os.makedirs(os.path.join(args.save_dir, args.model_version, args.dataset), exist_ok=True)

    ensure_trailing_newline(output_file)
    fout = open(output_file, "a", buffering=1)

    method_name = args.method.lower()
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

        batch_prompts = [prompts[i] for i in batch_indices]
        batch_inputs = [inputs[i] for i in batch_indices]
        batch_contexts = [contexts[i] for i in batch_indices]
        batch_answerss = [answerss[i] for i in batch_indices]
        batch_lengths = [lengths[i] for i in batch_indices]
        batch_datasets = [datasets[i] for i in batch_indices]
        batch_languages = [languages[i] for i in batch_indices]
        batch_all_classess = [all_classess[i] for i in batch_indices]
        batch__ids = [_ids[i] for i in batch_indices]

        batch_prompt_ids = [prompt_ids_list[i] for i in batch_indices]
        batch_input_ids, attention_mask, _ = _pad_batch(
            batch_prompt_ids,
            tokenizer.pad_token_id,
            padding_side=tokenizer.padding_side,
            device=input_device,
        )

        if args.max_capacity_prompts != -1:
            max_capacity_prompts = args.max_capacity_prompts
        elif args.max_capacity_prompts_ratio != -1:
            max_capacity_prompts = round(batch_input_ids.shape[1] * args.max_capacity_prompts_ratio)
        else:
            max_capacity_prompts = args.max_capacity_prompts
        
        
        if args.method.lower() != "fullkv":
            layers = len(model.model.layers)
            window_sizes = [args.window_size] * layers
            max_capacity_prompts = [max_capacity_prompts] * layers
            kernel_sizes = [7] * layers

            for i in range(layers):
                model.model.layers[i].self_attn.config.window_size = window_sizes[i]
                model.model.layers[i].self_attn.config.max_capacity_prompt = max_capacity_prompts[i]
                model.model.layers[i].self_attn.config.kernel_size = kernel_sizes[i]
                model.model.layers[i].self_attn.config.pooling = "avgpool"
                model.model.layers[i].self_attn.config.kv_method = args.method.lower()
                model.model.layers[i].self_attn.config.merge = args.merge
                model.model.layers[i].self_attn.config.beta = args.beta
                model.model.layers[i].self_attn.config.gqa_support = args.gqa_support
                model.model.layers[i].self_attn.config.gqa_func = args.gqa_func
                model.model.layers[i].self_attn.config.kv_cache_ratio = args.kv_cache_ratio
                model.model.layers[i].self_attn.config.inspect_attention_weight = args.inspect_attention_weight
                model.model.layers[i].self_attn.config.attention_log_file = args.attention_log_file
                if args.method.lower() == "championkv":
                    model.model.layers[i].self_attn.config.allocate_strategy = args.allocate_strategy
            

        context_length = batch_input_ids.shape[-1]
        with torch.inference_mode():
            if args.quant_method == None:        
                output = model.generate(
                    input_ids=batch_input_ids,
                    attention_mask=attention_mask,
                    output_attentions=args.output_attentions,
                    max_new_tokens=output_max_len,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    min_length=context_length + 1,
                    eos_token_id=[tokenizer.eos_token_id],
                )
            else:
                output = model.generate(
                    input_ids=batch_input_ids,
                    attention_mask=attention_mask,
                    output_attentions=args.output_attentions,
                    max_new_tokens=output_max_len,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    min_length=context_length + 1,
                    eos_token_id=[tokenizer.eos_token_id],
                    cache_implementation="quantized",
                    cache_config={"nbits": args.nbits, "backend": "HQQ", "device": "cuda", "residual_length": output_max_len, "axis_key": 1, "q_group_size": 64},
                )

        batch_outputs = tokenizer.batch_decode(output[:, context_length:], skip_special_tokens=True)

        torch.cuda.empty_cache()

        for j in range(len(batch_prompts)):
            example = {}

            example["prompt"] = batch_prompts[j]
            example["input"] = batch_inputs[j]
            example["context"] = batch_contexts[j]
            example["answers"] = batch_answerss[j]
            example["pred"] = batch_outputs[j]
            example["length"] = batch_lengths[j]

            example["dataset"] = batch_datasets[j]
            example["language"] = batch_languages[j]
            example["all_classes"] = batch_all_classess[j]
            example["_id"] = batch__ids[j]

            fout.write(json.dumps(example) + "\n")

    indices = list(range(len(prompts)))
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

    pbar = tqdm(total=len(prompts))
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
    parser.add_argument("--data_dir", type=str, default="./data/LongBench", help="Directory containing LongBench jsonl datasets")

    parser.add_argument("--model_name", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--model_path", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--use_fast_tokenizer", type=bool, default=True, help="")
    parser.add_argument("--output_attentions", type=bool, default=False, help="")
    
    parser.add_argument("--max_num_examples", type=int, default=None, help="maximum number of examples to evaluate per task.")
    parser.add_argument("--sample_method", type=str, default="topk", choices=["random", "topk"], help="how to sample the examples.")
    
    parser.add_argument("--max_new_tokens", type=int, default=None, help="")
    
    parser.add_argument("--eval_batch_size", type=int, nargs="+", default=[2], help="batch size(s) for evaluation.")
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
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "flash_attention_3", "sdpa", "eager"],
    )
    parser.add_argument("--method", type=str, default="championkv", choices=["fullkv", "pyramidkv", "snapkv", "adaptivekv", "adakv", "championkv"], help="KV cache compression method.")
    parser.add_argument("--quant_method",type=str,default=None,choices=["kivi","kvquant"])
    parser.add_argument("--nbits", type=int, default=8, help="")
    parser.add_argument("--max_capacity_prompts", type=int, default=512, help="")
    parser.add_argument("--window_size", type=int, default=8, help="KV cache recent window size used by compression methods.")
    parser.add_argument("--kv_cache_ratio", type=float, default=-1, help="KV cache ratio in (0,1], e.g. 0.2 means cache about 20% of input length.")
    parser.add_argument("--max_capacity_prompts_ratio", type=float, default=-1, help="")
    parser.add_argument("--steps", type=int, default=-1, help="maximum number of examples to evaluate per task.")
    parser.add_argument("--merge", type=str, default=None, help="kv merge method(look-m)")
    parser.add_argument('--model_name_suffix', type=str, default=None, help='suffix appended to model_version')
    parser.add_argument('--model_version', type=str, default=None, help='experiment identifier for result directory')
    parser.add_argument('--floor', type=float, default=0.2, help='hyper-parameter used in AdaKV')
    parser.add_argument('--floor_alpha', type=float, default=0.2, help='floor ratio used in AdaptiveKV')
    parser.add_argument('--skip', type=int, default=0, help='layer index threshold used in AdaptiveKV')
    parser.add_argument('--beta', type=int, default=20, help='pyramid allocation beta used in AdaptiveKV')
    parser.add_argument('--gqa_support', action='store_true', help='enable GQA branch in AdaptiveKV')
    parser.add_argument('--gqa_func', type=str, default='mean', choices=['max', 'mean'], help='aggregation function for AdaptiveKV GQA path')
    parser.add_argument(
        '--allocate_strategy',
        type=str,
        default='pyramid',
        choices=['pyramid', 'uniform'],
        help='ChampionKV layer budget strategy',
    )
    parser.add_argument("--recent_size", type=int, default=32, help="")
    parser.add_argument("--pruning_ratio", type=float, default=0.4, help="pruning ratio of Key Cache")
    parser.add_argument('--inspect_attention_weight', action='store_true', help='enable attention-weight inspection mode: run only one dataset and one prompt, then log history token mean attention weights')
    parser.add_argument('--attention_log_file', type=str, default=None, help='path to attention-weight log file; default is <save_dir>/Attention_weight.log')

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

    if args.kv_cache_ratio != -1:
        print(
            f"[WARN] kv_cache_ratio={args.kv_cache_ratio} is ignored. "
            "Using max_capacity_prompts/max_capacity_prompts_ratio for KV budget."
        )
        args.kv_cache_ratio = -1

    # Treat empty merge string as disabled merge.
    if args.merge is not None and args.merge.strip() == "":
        args.merge = None
    
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
    
    attn_impl = args.attn_implementation
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            use_cache=args.use_cache,
            attn_implementation=attn_impl,
        )
    except Exception as exc:
        if attn_impl == "flash_attention_3":
            print(
                f"[WARN] flash_attention_3 is unavailable in current runtime ({exc}). "
                "Falling back to flash_attention_2."
            )
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto",
                use_cache=args.use_cache,
                attn_implementation="flash_attention_2",
            )
        else:
            raise

    if hasattr(model, "hf_device_map"):
        print(f"Model hf_device_map: {model.hf_device_map}")
    input_device = _get_model_input_device(model)
    print(f"Model input device: {input_device}")
        

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    

        
    model.eval()

    base_model_version = build_model_version(args)
    print(f"Experiment model_version base: {base_model_version}")
    
    save_dir = args.save_dir
    
        
    max_capacity_prompts = args.max_capacity_prompts
    
    eval_batch_sizes = args.eval_batch_size
    run_datasets = datasets[:1] if args.inspect_attention_weight else datasets
    for batch_size in eval_batch_sizes:
        args.eval_batch_size = int(batch_size)
        if len(eval_batch_sizes) > 1:
            args.model_version = f"{base_model_version}_bs{args.eval_batch_size}"
        else:
            args.model_version = base_model_version
        print(f"Experiment model_version: {args.model_version}")

        for idx, dataset in enumerate(run_datasets):
            print(f"Working on max_capacity_prompts {args.max_capacity_prompts} dataset {dataset} - {idx}/{len(run_datasets)}")

            args.dataset = dataset

            args.data_file = os.path.join(args.data_dir, f"{args.dataset}.jsonl")

            main(args)




