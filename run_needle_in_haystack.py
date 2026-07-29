"""
This script is adapted from 
https://github.com/FranxYao/Long-Context-Data-Engineering
"""

import glob
import json
import argparse
import csv
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
import time
import threading


def _print_lightweight_help():
    print("usage: run_needle_in_haystack.py [options]\n")
    print("Key options:")
    print("  --method {fullkv,pyramidkv,snapkv,adaptivekv,adakv,championkv} (default: championkv)")
    print("  --model_name MODEL_NAME")
    print("  --model_provider {LLaMA3,Mistral}")
    print("  --haystack_dir HAYSTACK_DIR")
    print("  --max_capacity_prompts N (default: 128; --max_capacity_prompt alias accepted)")
    print("  --window_size N (default: 8)")
    print("  --kernel_size N (default: 7)")
    print("  --pooling {maxpool,avgpool}")
    print("  --allocate_strategy {pyramid,uniform} (default: pyramid)")
    print("  --s_len N --e_len N --step N")
    print("  --gqa_support --gqa_func {max,mean}")
    print("  --attn_implementation {flash_attention_2,flash_attention_3,sdpa,eager,None}")

if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    _print_lightweight_help()
    raise SystemExit(0)

import numpy as np
import torch
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

_SCRIPT_PATH = Path(__file__).resolve()
_DEFAULT_CHAMPIONKV_ROOT = _SCRIPT_PATH.parents[1]
_CHAMPIONKV_ROOT = Path(os.environ.get("CHAMPIONKV_ROOT", _DEFAULT_CHAMPIONKV_ROOT)).resolve()
if str(_CHAMPIONKV_ROOT) not in sys.path:
    sys.path.insert(0, str(_CHAMPIONKV_ROOT))

from KV_Cache_Utils.monkeypatch import replace_llama, replace_mistral

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)


SUPPORTED_KV_METHODS = {"fullkv", "pyramidkv", "snapkv", "adaptivekv", "adakv", "championkv"}


def normalize_kv_method(method):
    method = (method or "fullkv").lower()
    aliases = {
        "full": "fullkv",
        "none": "fullkv",
    }
    method = aliases.get(method, method)
    if method not in SUPPORTED_KV_METHODS:
        supported = ", ".join(sorted(SUPPORTED_KV_METHODS))
        raise ValueError(
            f"Unsupported kv method for current ChampionKV build: {method}. "
            f"Supported methods: {supported}"
        )
    return method


def get_model_input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cuda")


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

class LLMNeedleHaystackTester:
    """
    This class is used to test the LLM Needle Haystack.
    """
    def __init__(self,
                 needle="\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n",#    \n鍦ㄦ棫閲戝北鍋氱殑鏈€妫掔殑鏄簨鏄悆涓€涓笁鏄庢不浠ュ強鍦ㄦ櫞澶╁潗鍦ㄥ娲涢浄鏂叕鍥噷\n
                 haystack_dir="data/PaulGrahamEssays", # PaulGrahamEssays  
                 retrieval_question="The best thing to do in San Francisco is: ", 
                 results_version = 1,
                 context_lengths_min = None,
                 context_lengths_max = None,
                 context_lengths_num_intervals = 40,
                 context_lengths = None,
                 document_depth_percent_min = 50,
                 document_depth_percent_max = 100,
                 document_depth_percent_intervals = 1,
                 document_depth_percents = None,
                 document_depth_percent_interval_type = "linear",
                 model_provider = "OpenAI",
                 openai_api_key=None,
                 anthropic_api_key = None,
                 model_name='',
                 model_name_suffix=None,
                 model_version=None, 
                 num_concurrent_requests = 1,
                 save_results = True,
                 save_contexts = True,
                 final_context_length_buffer = 200,
                 seconds_to_sleep_between_completions = None,
                 print_ongoing_status = True, 
                 step=100, 
                 method='championkv', 
                 attn_implementation='flash_attention_2',
                 max_capacity_prompts=128,
                 window_size=8,
                 kernel_size=7,
                 pooling="maxpool",
                 merge=None,
                 beta=20,
                 floor_alpha=0.2,
                 skip=0,
                 normalize=True,
                 gqa_support=False,
                 gqa_func="mean",
                 allocate_strategy="pyramid",
                 inspect_attention_weight=False,
                 attention_log_file=None,
                 enable_performance_profile=False,
                 record_performance=False,
                 performance_log_dir="results_needle/performance",
                 performance_log_file=None,
                 record_decoding_memory=False,
                 decoding_memory_warmup_tokens=3):
        """        
        :param needle: The needle to be found in the haystack. Default is None.
        :param haystack_dir: The directory of text files to use as background context (or a haystack) in which the needle is to be found. Default is Paul Graham Essays.
        :param retrieval_question: The question which with to prompt the model to do the retrieval.
        :param results_version: In case you would like to try the same combination of model, context length, and depth % multiple times, change the results version other than 1
        :param num_concurrent_requests: Due to volume, this object is set up to run concurrent requests, default = 1. Be careful of rate limits.
        :param save_results: Whether or not you would like to save your contexts to file. Warning: These will get long! Default = True
        :param save_contexts: Whether or not you would like to save your contexts to file. Warning: These will get long! Default is True.
        :param final_context_length_buffer: The amount of cushion you'd like to leave off the input context to allow for the output context. Default 200 tokens
        :param context_lengths_min: The minimum length of the context. Default is 0.
        :param context_lengths_max: The maximum length of the context. Default is 200000.
        :param context_lengths_num_intervals: The number of intervals for the context length. Default is 35.
        :param context_lengths: The lengths of the context. Default is None.
        :param document_depth_percent_min: The minimum depth percent of the document. Default is 0.
        :param document_depth_percent_max: The maximum depth percent of the document. Default is 100.
        :param document_depth_percent_intervals: The number of intervals for the document depth percent. Default is 35.
        :param document_depth_percents: The depth percentages of the document. Default is None.
        :param document_depth_percent_interval_type: The type of interval for the document depth percent. Must be either 'linear' or 'sigmoid'. Default is 'linear'.
        :param model_provider: The provider of the model. Must be either 'OpenAI' or 'Anthropic'. Default is 'OpenAI'.
        :param openai_api_key: The API key for OpenAI. Default is None.
        :param anthropic_api_key: The API key for Anthropic. Default is None.
        :param model_name: The name of the model. Default is 'gpt-4-1106-preview'.
        :param seconds_to_sleep_between_completions: The number of seconds to sleep between completions. Default is None.
        :param print_ongoing_status: Whether or not to print the ongoing status. Default is True.
        """
        if not needle or not haystack_dir or not retrieval_question:
            raise ValueError("Needle, haystack, and retrieval_question must be provided.")
        
        self.needle = needle
        self.haystack_dir = haystack_dir
        self.retrieval_question = retrieval_question
        self.results_version = results_version
        self.num_concurrent_requests = num_concurrent_requests
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.save_contexts = save_contexts
        self.seconds_to_sleep_between_completions = seconds_to_sleep_between_completions
        self.print_ongoing_status = print_ongoing_status
        self.model_provider = model_provider
        self.testing_results = []
        self.step = step
        self.method = normalize_kv_method(method)
        self.max_capacity_prompts = max_capacity_prompts
        self.attn_implementation = None if attn_implementation in [None, "None", "none"] else attn_implementation
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge if merge not in ["", "None", "none"] else None
        self.beta = beta
        self.floor_alpha = floor_alpha
        self.skip = skip
        self.normalize = normalize
        self.gqa_support = gqa_support
        self.gqa_func = gqa_func
        self.allocate_strategy = allocate_strategy
        self.inspect_attention_weight = inspect_attention_weight
        self.attention_log_file = attention_log_file
        self.record_performance = record_performance
        self.enable_performance_profile = enable_performance_profile or record_performance
        self.performance_log_dir = performance_log_dir
        self.performance_log_file = performance_log_file
        self.performance_log_path = None

        self.record_decoding_memory = record_decoding_memory or record_performance
        self.decoding_memory_warmup_tokens = decoding_memory_warmup_tokens


        self.model_version = model_version
        if self.model_version is None:
            capacity_tag = "full" if self.method == "fullkv" else str(self.max_capacity_prompts)
            self.model_version = f"Needle_{self.method}_cap{capacity_tag}"
        if(model_name_suffix is not None): self.model_version += "_" + model_name_suffix

        if self.record_performance:
            os.makedirs(self.performance_log_dir, exist_ok=True)
            if self.performance_log_file is None:
                safe_model_version = self.model_version.replace("/", "_").replace(".", "_")
                self.performance_log_file = f"{safe_model_version}_performance.csv"
            self.performance_log_path = os.path.join(self.performance_log_dir, self.performance_log_file)
            self.ensure_performance_log_header()

        if context_lengths is None:
            if context_lengths_min is None or context_lengths_max is None or context_lengths_num_intervals is None:
                raise ValueError("Either context_lengths_min, context_lengths_max, context_lengths_intervals need to be filled out OR the context_lengths_list needs to be supplied.")
            else:
                # self.context_lengths = np.round(np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals, endpoint=True)).astype(int)
                self.context_lengths = np.arange(context_lengths_min, context_lengths_max+1, step=self.step)
        else:
            self.context_lengths = context_lengths


        if document_depth_percents is None:
            if document_depth_percent_min is None or document_depth_percent_max is None or document_depth_percent_intervals is None:
                raise ValueError("Either document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals need to be filled out OR the document_depth_percents needs to be supplied.")
            else:
                if document_depth_percent_interval_type == 'linear':
                    self.document_depth_percents = np.round(np.linspace(document_depth_percent_min, document_depth_percent_max, num=document_depth_percent_intervals, endpoint=True)).astype(int)
                elif document_depth_percent_interval_type == 'sigmoid':
                    self.document_depth_percents = [self.logistic(x) for x in np.linspace(document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals)]
        else:
            self.document_depth_percents = document_depth_percents

        if document_depth_percent_interval_type not in [None, "linear", "sigmoid"]:
            raise ValueError("document_depth_percent_interval_type must be either None, 'linear' or 'sigmoid'. If you'd like your own distribution give a list of ints in via document_depth_percent_intervals")
        
        self.model_name = model_name

        if(self.model_provider in ["LLaMA3", "Mistral"]):
            self.enc = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.enc.padding_side = "left"
            if self.enc.pad_token is None:
                self.enc.pad_token = self.enc.eos_token
                self.enc.pad_token_id = self.enc.eos_token_id
            print("loading from %s" % model_name)
            print(f"ChampionKV root: {_CHAMPIONKV_ROOT}")
            print(
                f"KV method={self.method}, max_capacity_prompts={self.max_capacity_prompts}, "
                f"window_size={self.window_size}, allocate_strategy={self.allocate_strategy}"
            )

            if self.model_provider == 'LLaMA3':
                replace_llama(self.method)
            elif self.model_provider == 'Mistral':
                replace_mistral(self.method)

            model_kwargs = {
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "low_cpu_mem_usage": True,
            }
            if self.attn_implementation is not None:
                model_kwargs["attn_implementation"] = self.attn_implementation

            self.model_to_test = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs,
            ).eval()
            self.input_device = get_model_input_device(self.model_to_test)
            self.configure_kv_cache()
        else:
            raise ValueError("model_provider must be either 'LLaMA3' or 'Mistral'")

    def configure_kv_cache(self):
        if self.method == "fullkv":
            return

        layers = len(self.model_to_test.model.layers)
        window_size = self.window_size
        max_capacity_prompts = self.max_capacity_prompts
        kernel_size = self.kernel_size

        if not isinstance(window_size, list):
            window_size = [window_size] * layers
        if not isinstance(max_capacity_prompts, list):
            max_capacity_prompts = [max_capacity_prompts] * layers
        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size] * layers

        for layer_idx in range(layers):
            attn_config = self.model_to_test.model.layers[layer_idx].self_attn.config
            attn_config.window_size = window_size[layer_idx]
            attn_config.max_capacity_prompt = max_capacity_prompts[layer_idx]
            attn_config.kernel_size = kernel_size[layer_idx]
            attn_config.pooling = self.pooling
            attn_config.merge = self.merge
            attn_config.kv_method = self.method
            attn_config.beta = self.beta
            attn_config.floor_alpha = self.floor_alpha
            attn_config.floor_ratio = self.floor_alpha
            attn_config.skip = self.skip
            attn_config.normalize = self.normalize
            attn_config.gqa_support = self.gqa_support
            attn_config.gqa_func = self.gqa_func
            attn_config.allocate_strategy = self.allocate_strategy
            attn_config.inspect_attention_weight = self.inspect_attention_weight
            attn_config.attention_log_file = self.attention_log_file

    def ensure_performance_log_header(self):
        if self.performance_log_path is None:
            return
        if os.path.exists(self.performance_log_path) and os.path.getsize(self.performance_log_path) > 0:
            return
        with open(self.performance_log_path, "w", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=self.performance_log_fieldnames())
            writer.writeheader()

    def performance_log_fieldnames(self):
        return [
            "timestamp_utc",
            "model",
            "model_version",
            "method",
            "max_capacity_prompts",
            "window_size",
            "context_length",
            "depth_percent",
            "prompt_tokens",
            "generated_tokens",
            "peak_gpu_memory_gb",
            "steady_state_peak_gpu_memory_gb",
            "prefill_time_ms",
            "decode_time_ms",
            "decoding_latency_ms_per_token",
            "total_generation_time_ms",
        ]

    def append_performance_log(self, metrics):
        if self.performance_log_path is None:
            return
        try:
            with open(self.performance_log_path, "a", newline="") as fout:
                writer = csv.DictWriter(fout, fieldnames=self.performance_log_fieldnames())
                writer.writerow({key: metrics.get(key, "") for key in self.performance_log_fieldnames()})
        except (OSError, IOError) as e:
            print(f"[WARN] Failed to write performance log: {e}")

    def sync_cuda(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def reset_peak_gpu_memory(self):
        if not torch.cuda.is_available():
            return
        for device_idx in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(device_idx)

    def get_peak_gpu_memory_gb(self):
        if not torch.cuda.is_available():
            return 0.0
        peak_bytes = 0
        for device_idx in range(torch.cuda.device_count()):
            peak_bytes = max(peak_bytes, torch.cuda.max_memory_allocated(device_idx))
        return peak_bytes / (1024 ** 3)

    def extract_forward_input_length(self, args, kwargs):
        input_ids = kwargs.get("input_ids") if kwargs is not None else None
        if input_ids is None and args:
            input_ids = args[0]
        if torch.is_tensor(input_ids):
            return int(input_ids.shape[-1])

        inputs_embeds = kwargs.get("inputs_embeds") if kwargs is not None else None
        if torch.is_tensor(inputs_embeds):
            return int(inputs_embeds.shape[-2])
        return None

    def timed_generate(self, **generate_kwargs):
        if not self.enable_performance_profile:
            with torch.inference_mode():
                return self.model_to_test.generate(**generate_kwargs), None

        forward_times = []
        active_forward = {}
        steady_state_peak_memory_bytes = 0
        decode_step_counter = 0

        def start_forward(args, kwargs):
            self.sync_cuda()
            active_forward["start_time"] = time.perf_counter()
            active_forward["input_length"] = self.extract_forward_input_length(args, kwargs)

        def finish_forward():
            nonlocal steady_state_peak_memory_bytes, decode_step_counter
            self.sync_cuda()
            start_time = active_forward.pop("start_time", None)
            input_length = active_forward.pop("input_length", None)
            if start_time is not None:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                forward_times.append(
                    {
                        "input_length": input_length,
                        "elapsed_ms": elapsed_ms,
                    }
                )
                if input_length is not None and input_length == 1:
                    decode_step_counter += 1
                    if self.record_decoding_memory and decode_step_counter > self.decoding_memory_warmup_tokens:
                        for device_idx in range(torch.cuda.device_count()):
                            steady_state_peak_memory_bytes = max(
                                steady_state_peak_memory_bytes,
                                torch.cuda.memory_allocated(device_idx),
                            )

        def pre_hook_with_kwargs(module, args, kwargs):
            start_forward(args, kwargs)

        def post_hook_with_kwargs(module, args, kwargs, output):
            finish_forward()

        def pre_hook(module, args):
            start_forward(args, {})

        def post_hook(module, args, output):
            finish_forward()

        handles = []
        try:
            try:
                handles.append(self.model_to_test.register_forward_pre_hook(pre_hook_with_kwargs, with_kwargs=True))
                handles.append(self.model_to_test.register_forward_hook(post_hook_with_kwargs, with_kwargs=True))
            except TypeError:
                handles.append(self.model_to_test.register_forward_pre_hook(pre_hook))
                handles.append(self.model_to_test.register_forward_hook(post_hook))

            self.reset_peak_gpu_memory()
            self.sync_cuda()
            total_start_time = time.perf_counter()
            with torch.inference_mode():
                output_ids = self.model_to_test.generate(**generate_kwargs)
            self.sync_cuda()
            total_generation_time_ms = (time.perf_counter() - total_start_time) * 1000.0
            peak_gpu_memory_gb = self.get_peak_gpu_memory_gb()
        finally:
            for handle in handles:
                handle.remove()

        prefill_time_ms = None
        for item in forward_times:
            if item["input_length"] is None or item["input_length"] > 1:
                prefill_time_ms = item["elapsed_ms"]
                break
        if prefill_time_ms is None and forward_times:
            prefill_time_ms = forward_times[0]["elapsed_ms"]
        if prefill_time_ms is None:
            prefill_time_ms = total_generation_time_ms

        input_ids = generate_kwargs.get("input_ids")
        prompt_tokens = int(input_ids.shape[-1]) if torch.is_tensor(input_ids) else 0
        generated_tokens = int(output_ids.shape[-1] - prompt_tokens)
        decode_time_ms = max(total_generation_time_ms - prefill_time_ms, 0.0)
        decoding_latency_ms_per_token = decode_time_ms / max(generated_tokens, 1)

        steady_state_peak_gpu_memory_gb = steady_state_peak_memory_bytes / (1024 ** 3) if self.record_decoding_memory and steady_state_peak_memory_bytes > 0 else 0.0

        metrics = {
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "peak_gpu_memory_gb": peak_gpu_memory_gb,
            "steady_state_peak_gpu_memory_gb": steady_state_peak_gpu_memory_gb,
            "prefill_time_ms": prefill_time_ms,
            "decode_time_ms": decode_time_ms,
            "decoding_latency_ms_per_token": decoding_latency_ms_per_token,
            "total_generation_time_ms": total_generation_time_ms,
        }
        return output_ids, metrics

    def logistic(self, x, L=100, x0=50, k=.1):
        if x == 0:
            return 0
        if x == 100:
            return 100
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)
    
    def bound_evaluate_and_log(self, *args):
        self.evaluate_and_log(*args)

    def run_test(self, args):

        # Run through each iteration of context_lengths and depths
        tasks = []
        for context_length in self.context_lengths:
            if context_length < args.s_len or context_length > args.e_len: continue
            for depth_percent in self.document_depth_percents:
                task = self.bound_evaluate_and_log(context_length, depth_percent)

    def generate_prompt(self, context):
        # Generate the prompt for the Anthropic model
        # Replace the following line with the appropriate prompt structure
        if(self.model_provider not in ["OpenAI", "Anthropic"]):
            test_format=f"<|im_start|> This is a very long story book: <book> {context} </book>.\n Based on the content of the book, Question: {self.retrieval_question}\nAnswer:"
            return test_format
        else: 
            return [
                {
                    "role": "system",
                    "content": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct"
                },
                {
                    "role": "user",
                    "content": context
                    },
                {
                    "role": "user",
                    "content": f"{self.retrieval_question} Don't give information outside the document or repeat your findings. The document definitely contains the answer, and I'm 100% sure. So try your best to find it."
                },
                {
                    "role": "assistant",
                    "content":"",
                },
                
            ]

    def evaluate_and_log(self, context_length, depth_percent):
        # Checks to see if you've already checked a length/percent/version.
        # This helps if the program stop running and you want to restart later
        if self.save_results:
            if self.result_exists(context_length, depth_percent):
                print("result exists, skipping")
                return
            else:
                print("result does not exist, testing")

        # Go generate the required length context and place your needle statement in
        context = self.generate_context(context_length, depth_percent)

        # Prepare your message to send to the model you're going to evaluate
        prompt = self.generate_prompt(context)
        test_start_time = time.time()
        
        if(self.model_provider in ["LLaMA3", "Mistral"]):

            prompt = self.enc(prompt, return_tensors="pt")
            prompt = {key: value.to(self.input_device) for key, value in prompt.items()}
            input_ids = prompt['input_ids']
            
            output_ids, perf_metrics = self.timed_generate(
                input_ids=input_ids,
                attention_mask=prompt.get("attention_mask"),
                output_attentions=False,
                max_new_tokens=30,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                eos_token_id=[self.enc.eos_token_id, self.enc.encode("\n", add_special_tokens=False)[-1]],
            )
            response = self.enc.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        else:
            perf_metrics = None

        
        print(response)
        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time
        if len(response) != 0:
            score = scorer.score(self.needle, response)['rouge1'].fmeasure*10
        else:
            score = 0.0

        results = {
            'model' : self.model_name,
            'context_length' : int(context_length),
            'depth_percent' : float(depth_percent),
            'version' : self.results_version,
            'needle' : self.needle,
            'model_response' : response,
            'score' : score,
            'test_duration_seconds' : test_elapsed_time,
            'test_timestamp_utc' : datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z'), 
        }

        if perf_metrics is not None:
            print(
                "Peak GPU Memory: "
                f"{perf_metrics['peak_gpu_memory_gb']:.3f} GB | "
                f"Prefill Time: {perf_metrics['prefill_time_ms']:.3f} ms | "
                "Decoding Latency: "
                f"{perf_metrics['decoding_latency_ms_per_token']:.3f} ms/token"
            )
            if self.record_decoding_memory and perf_metrics.get("steady_state_peak_gpu_memory_gb", 0) > 0:
                print(
                    f"Steady-State Peak GPU Memory: "
                    f"{perf_metrics['steady_state_peak_gpu_memory_gb']:.3f} GB "
                    f"(skipped {self.decoding_memory_warmup_tokens} warmup tokens)"
                )

            performance_row = {
                "timestamp_utc": results["test_timestamp_utc"],
                "model": self.model_name,
                "model_version": self.model_version,
                "method": self.method,
                "max_capacity_prompts": self.max_capacity_prompts,
                "window_size": self.window_size,
                "context_length": int(context_length),
                "depth_percent": float(depth_percent),
                "prompt_tokens": perf_metrics["prompt_tokens"],
                "generated_tokens": perf_metrics["generated_tokens"],
                "peak_gpu_memory_gb": f"{perf_metrics['peak_gpu_memory_gb']:.6f}",
                "steady_state_peak_gpu_memory_gb": f"{perf_metrics['steady_state_peak_gpu_memory_gb']:.6f}",
                "prefill_time_ms": f"{perf_metrics['prefill_time_ms']:.6f}",
                "decode_time_ms": f"{perf_metrics['decode_time_ms']:.6f}",
                "decoding_latency_ms_per_token": f"{perf_metrics['decoding_latency_ms_per_token']:.6f}",
                "total_generation_time_ms": f"{perf_metrics['total_generation_time_ms']:.6f}",
            }
            self.append_performance_log(performance_row)

        self.testing_results.append(results)

        if self.print_ongoing_status:
            print (f"-- Test Summary -- ")
            print (f"Duration: {test_elapsed_time:.1f} seconds")
            print (f"Context: {context_length} tokens")
            print (f"Depth: {depth_percent}%")
            print (f"Score: {score}")
            print (f"Response: {response}\n")

        context_file_location = f'{self.model_version.replace(".", "_")}_len_{context_length}_depth_{int(depth_percent*100)}'

        if self.save_contexts:
            results['file_name'] : context_file_location

            # Save the context to file for retesting
            if not os.path.exists('results_needle/contexts'):
                os.makedirs('results_needle/contexts')

            if not os.path.exists(f'results_needle/contexts/{self.model_version}'):
                os.makedirs(f'results_needle/contexts/{self.model_version}')

            with open(f'results_needle/contexts/{self.model_version}/{context_file_location}_context.txt', 'w') as f:
                f.write(context)
            
        if self.save_results:
            # Save the context to file for retesting
            if not os.path.exists('results_needle/results'):
                os.makedirs('results_needle/results')
            
            if not os.path.exists(f'results_needle/results/{self.model_version}'):
                os.makedirs(f'results_needle/results/{self.model_version}')

            # Save the result to file for retesting
            p = f'results_needle/results/{self.model_version}/{context_file_location}_results.json'
            print("Writing at %s" % p)
            with open(p, 'w') as f:
                json.dump(results, f, ensure_ascii=False)
                
        # Clean up memory explicitly
        if "prompt" in locals():
            del prompt
        if "input_ids" in locals():
            del input_ids
        if "output_ids" in locals():
            del output_ids
        if "context" in locals():
            del context
            
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def result_exists(self, context_length, depth_percent):
        """
        Checks to see if a result has already been evaluated or not
        """

        results_dir = 'results_needle/results/' + self.model_version
        print("Searching existing results at %s" % results_dir)
        if not os.path.exists(results_dir):
            return False
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                with open(os.path.join(results_dir, filename), 'r') as f:
                    result = json.load(f)
                    context_length_met = result['context_length'] == context_length
                    depth_percent_met = result['depth_percent'] == depth_percent
                    version_met = result.get('version', 1) == self.results_version
                    model_met = result['model'] == self.model_name
                    # import ipdb; ipdb.set_trace()
                    if context_length_met and depth_percent_met and version_met and model_met:
                        return True
        return False

    def generate_context(self, context_length, depth_percent):
        # Load up tiktoken so we navigate tokens more easily

        # Get your Paul Graham files loaded into a string
        context = self.read_context_files()

        # Truncate the Paul Graham essays to the context length you desire
        context = self.encode_and_trim(context, context_length)

        # Insert your random statement according to your depth percent
        context = self.insert_needle(context, depth_percent, context_length)

        return context
    
    def encode_text_to_tokens(self, text):
        if self.model_provider in ["Mistral", "LLaMA3"]:
            return self.enc.encode(text, add_special_tokens=False)
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            return self.enc.encode(text).ids
        else:
            return self.enc.encode(text)
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")
    
    def insert_needle(self, context, depth_percent, context_length):
        tokens_needle = self.encode_text_to_tokens(self.needle)
        tokens_context = self.encode_text_to_tokens(context)

        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer

        # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle)]

        if depth_percent == 100:
            # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
            tokens_new_context = tokens_context + tokens_needle
        else:
            # Go get the position (in terms of tokens) to insert your needle
            insertion_point = int(len(tokens_context) * (depth_percent / 100))

            # tokens_new_context represents the tokens before the needle
            tokens_new_context = tokens_context[:insertion_point]

            # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
            if(self.model_provider in ["LLaMA", "LongLLaMA"]): period_tokens = [29889, 869]
            elif(self.model_provider == "LLaMA3"): period_tokens = [13]
            elif(self.model_provider == "Mistral"): period_tokens = [842, 28723]
            elif(self.model_provider == "GLM"): period_tokens = [918, 30930]
            else: period_tokens = self.encode_text_to_tokens('.')
            
            # Then we iteration backwards until we find the first period
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]

            print("insertion at %d" % insertion_point)
            # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
            # Now we have a needle in a haystack
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        # Convert back to a string and return it
        new_context = self.decode_tokens(tokens_new_context)
        return new_context

    def get_context_length_in_tokens(self, context):
        if self.model_provider in ["Mistral", "LLaMA3"]:
            return len(self.enc.encode(context, add_special_tokens=False))
        else:
            return len(self.enc.encode(context))
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

    def read_context_files(self):
        context = ""
        max_context_length = max(self.context_lengths)

        while self.get_context_length_in_tokens(context) < max_context_length:
            for file in glob.glob(f"{self.haystack_dir}/*.txt"):
                with open(file, 'r') as f:
                    context += f.read()
        return context

    def get_tokens_from_context(self, context):
        if self.model_provider in ["Mistral", "LLaMA3"]:
            return self.enc.encode(context, add_special_tokens=False)
        else:
            return self.enc.encode(context)
            # raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")
        
    def decode_tokens(self, tokens, context_length=None):
        if self.model_provider in ["Mistral", "LLaMA3"]:
            return self.enc.decode(tokens[:context_length], skip_special_tokens=True)
        else:
            return self.enc.decode(tokens[:context_length])
            # raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

    def encode_and_trim(self, context, context_length):
        tokens = self.get_tokens_from_context(context)
        if len(tokens) > context_length:
            context = self.decode_tokens(tokens, context_length)
        return context
    
    def get_results(self):
        return self.testing_results
    
    def print_start_test_summary(self):
        print ("\n")
        print ("Starting Needle In A Haystack Testing...")
        print (f"- Model: {self.model_name}")
        print (f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print (f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        print (f"- Needle: {self.needle.strip()}")
        print ("\n\n")

    def start_test(self, args):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        #asyncio.run(self.run_test())
        self.run_test(args)


if __name__ == "__main__":
    # Tons of defaults set, check out the LLMNeedleHaystackTester's init for more info
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--s_len', metavar='N', type=int, help='a number')
    parser.add_argument('-e', '--e_len', metavar='N', type=int, help='a number')
    parser.add_argument('--model_name', type=str, default=None, help='name of model')
    parser.add_argument('--haystack_dir', type=str, default='data/PaulGrahamEssays', help='directory of haystack txt files')
    parser.add_argument("--attn_implementation", type=str,  default="flash_attention_2", choices=["flash_attention_2", "flash_attention_3", "sdpa", "eager", "None"])
    parser.add_argument('--model_version', type=str, default=None, help='provider of model')
    parser.add_argument('--model_name_suffix', type=str, default=None, help='name of model')
    parser.add_argument('--model_provider', type=str, default="LLaMA3", choices=["LLaMA3", "Mistral"], help='which model to use')
    parser.add_argument('--api_key', type=str, default="", help='OpenAI API Key')
    parser.add_argument('--step', type=int, default=1000)
    parser.add_argument('--method', type=str, default="championkv", choices=['fullkv', 'pyramidkv', 'snapkv', 'adaptivekv', 'adakv', 'championkv'])
    parser.add_argument('--max_capacity_prompts', '--max_capacity_prompt', dest='max_capacity_prompts', type=int, default=128, help='Total KV cache budget per layer. The singular alias is kept for backward compatibility.')
    parser.add_argument('--window_size', type=int, default=8, help='KV recent window size')
    parser.add_argument('--kernel_size', type=int, default=7, help='pooling kernel size for attention score smoothing')
    parser.add_argument('--pooling', type=str, default='maxpool', choices=['maxpool', 'avgpool'], help='pooling method')
    parser.add_argument('--merge', type=str, default=None, help='optional LOOK-M merge method, e.g. pivot')
    parser.add_argument('--beta', type=int, default=20, help='pyramid layer allocation beta')
    parser.add_argument('--floor_alpha', type=float, default=0.2, help='floor ratio used by AdaptiveKV')
    parser.add_argument('--skip', type=int, default=0, help='layer skip threshold used by AdaptiveKV')
    parser.add_argument('--normalize', type=str2bool, default=True, help='whether to normalize AdaptiveKV scores')
    parser.add_argument('--gqa_support', action='store_true', help='enable GQA path')
    parser.add_argument('--gqa_func', type=str, default='mean', choices=['max', 'mean'], help='aggregation function for GQA path')
    parser.add_argument('--allocate_strategy', type=str, default='pyramid', choices=['pyramid', 'uniform'], help='ChampionKV layer budget strategy')
    parser.add_argument('--inspect_attention_weight', action='store_true', help='log attention-weight inspection data')
    parser.add_argument('--attention_log_file', type=str, default=None, help='optional path for attention logs')
    parser.add_argument('--enable_performance_profile', action='store_true', help='print peak GPU memory, prefill time, and decoding latency')
    parser.add_argument('--record_performance', action='store_true', help='write per-sample performance metrics to CSV')
    parser.add_argument('--performance_log_dir', type=str, default='results_needle/performance', help='directory for performance CSV files')
    parser.add_argument('--performance_log_file', type=str, default=None, help='optional performance CSV filename')
    parser.add_argument('--record_decoding_memory', action='store_true', help='record steady-state decoding GPU memory per token')
    parser.add_argument('--decoding_memory_warmup_tokens', type=int, default=3, help='number of initial decode tokens to skip as warmup (default: 3)')
    args = parser.parse_args()

    print(f"ADAKV_USE_TINY_API_CUDA (python): {os.environ.get('ADAKV_USE_TINY_API_CUDA', '<unset>')}")
    
    ht = LLMNeedleHaystackTester(model_name=args.model_name,
                                 haystack_dir=args.haystack_dir,
                                 model_name_suffix=args.model_name_suffix,
                                 model_provider=args.model_provider,
                                 model_version=args.model_version, 
                                 context_lengths_min=args.s_len,
                                 save_contexts=True,
                                 save_results=True,
                                 openai_api_key=args.api_key, 
                                 context_lengths_max=args.e_len, 
                                 step=args.step, 
                                 method=args.method, 
                                 max_capacity_prompts=args.max_capacity_prompts,
                                 attn_implementation=args.attn_implementation,
                                 window_size=args.window_size,
                                 kernel_size=args.kernel_size,
                                 pooling=args.pooling,
                                 merge=args.merge,
                                 beta=args.beta,
                                 floor_alpha=args.floor_alpha,
                                 skip=args.skip,
                                 normalize=args.normalize,
                                 gqa_support=args.gqa_support,
                                 gqa_func=args.gqa_func,
                                 allocate_strategy=args.allocate_strategy,
                                 inspect_attention_weight=args.inspect_attention_weight,
                                 attention_log_file=args.attention_log_file,
                                 enable_performance_profile=args.enable_performance_profile,
                                 record_performance=args.record_performance,
                                 performance_log_dir=args.performance_log_dir,
                                 performance_log_file=args.performance_log_file,
                                 record_decoding_memory=args.record_decoding_memory,
                                 decoding_memory_warmup_tokens=args.decoding_memory_warmup_tokens
                                 )

    ht.start_test(args)


