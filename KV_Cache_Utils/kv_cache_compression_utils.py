import torch
import os
import time
import importlib
import copy
import torch.nn.functional as F
import torch.nn as nn
import math
import warnings
from pathlib import Path

from typing import List

from typing import List, Optional, Tuple
from transformers.cache_utils import Cache

_BATCH_SHAPE_DEPENDENT_KV_METHODS = {
    "pyramidkv",
    "snapkv",
    "adakv",
    "adaptivekv",
    "championkv",
}

def kv_method_requires_batch_invariant_eval(method_name: Optional[str]) -> bool:
    """Legacy compatibility hook.

    Current public KV methods implement per-sample KV metadata inside the compression
    classes, so scripts can keep physical eval batches enabled.
    """
    if method_name is None:
        return False
    return False

def log_compression_start(cluster, method_name: str, max_capacity_prompts: int):
    if getattr(cluster, "_compression_started", False):
        return
    print(
        f"[KV-Compress] method={method_name}, max_capacity_prompts={max_capacity_prompts}"
    )
    cluster._compression_started = True



def key_pruner_query_driven(kv_states, q_states, recent_size=128, ratio=0.3):
    _, _, seqlen, head_dim = kv_states.shape
    k = int(head_dim * ratio)
    # new efficient implementation
    queries_norm = torch.pow(q_states[..., -32:, :], 2).mean(dim=2)
    keys_norm = torch.pow(kv_states, 2).mean(dim=2)
    key = queries_norm * keys_norm
    _, indices = torch.topk(key, k, dim=-1, largest=False)
    keep_idx = indices.sort().values
    mask = torch.zeros(key.shape, dtype=torch.bool).to(kv_states.device)
    mask = mask.scatter_(-1, keep_idx, 1)                   
    mask_k = mask.unsqueeze(2).expand(-1, -1, seqlen - recent_size, -1)

    return kv_states[:, :, :seqlen - recent_size, :][~mask_k].reshape(1,-1,seqlen - recent_size,head_dim-k), kv_states[:, :, seqlen - recent_size:, :], ~mask

class DynamicCacheSplitHeadFlatten(Cache):
    '''
    adapt from https://github.com/FFY0/AdaKV.
    '''
    def __init__(self) ->None:
        # Token wise List[]  Head wise KV List[torch.Tensor]
        try:
            super().__init__(layers=[])
        except TypeError:
            super().__init__()
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens = 0
        self._validated_layers: set = set()
        self._logged_update_backend = False

    def __len__(self):
        return len(self.key_cache)

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield (tuple(self.key_cache[layer_idx]),tuple(self.value_cache[layer_idx]))

    def __getitem__(self, layer_idx: int) -> Tuple[Tuple[torch.Tensor],Tuple[torch.Tensor]]:
        if layer_idx < len(self):
            return (tuple(self.key_cache[layer_idx]),tuple(self.value_cache[layer_idx]))
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")
    def _validate_flatten_update_inputs(self, current_flat, appended_flat, head_lens, cu_klen, batch_size=1):
        if head_lens is None or cu_klen is None:
            raise ValueError("cache_kwargs must include head_lens and cu_klen for flatten cache update")

        if not torch.is_tensor(head_lens) or not torch.is_tensor(cu_klen):
            raise TypeError("head_lens and cu_klen must be torch tensors")

        head_lens = head_lens.to(device=current_flat.device, dtype=torch.int64).contiguous()
        cu_klen = cu_klen.to(device=current_flat.device, dtype=torch.int64).contiguous()

        if head_lens.dim() != 1 or cu_klen.dim() != 1:
            raise ValueError("head_lens and cu_klen must be 1-D tensors")

        if cu_klen.numel() != head_lens.numel() + 1:
            raise ValueError(
                f"cu_klen length ({cu_klen.numel()}) must equal head_lens length + 1 ({head_lens.numel() + 1})"
            )

        if torch.any(head_lens < 0):
            raise ValueError("head_lens contains negative values")

        if torch.any(cu_klen[1:] < cu_klen[:-1]):
            raise ValueError("cu_klen must be non-decreasing")

        expected_old_tokens = int(head_lens.sum().item())
        expected_old_total = expected_old_tokens * int(batch_size)
        if current_flat.shape[0] != expected_old_total:
            raise ValueError(
                f"flatten cache token count mismatch: cache has {current_flat.shape[0]}, expected {expected_old_total} "
                f"(head_lens sum {expected_old_tokens} * batch_size {int(batch_size)})"
            )

        expected_new_tokens = int(head_lens.numel())
        expected_new_total = expected_new_tokens * int(batch_size)
        if appended_flat.shape[0] != expected_new_total:
            raise ValueError(
                f"new token count mismatch: appended tensor has {appended_flat.shape[0]}, expected {expected_new_total} "
                f"(per-batch {expected_new_tokens} * batch_size {int(batch_size)})"
            )

        return head_lens, cu_klen, expected_old_tokens, expected_new_tokens

    def _safe_update_flatten_view(self, current_flat, appended_flat, head_lens, cu_klen):
        # Safe reference update: append one token to each head segment in flatten layout.
        chunks = []
        for head_idx in range(head_lens.numel()):
            start = int(cu_klen[head_idx].item())
            length = int(head_lens[head_idx].item())
            end = start + length
            chunks.append(current_flat[start:end])
            chunks.append(appended_flat[head_idx : head_idx + 1])
        return torch.cat(chunks, dim=0).contiguous()

    def _vectorized_update_flatten_view(self, current_flat, appended_flat, head_lens, cu_klen):
        num_heads = head_lens.numel()
        old_total = current_flat.shape[0]
        new_total = old_total + num_heads
        head_dim = current_flat.shape[1]

        new_flat = current_flat.new_empty(new_total, head_dim)

        head_lens_i64 = head_lens.to(torch.int64)
        cu_klen_i64 = cu_klen.to(torch.int64)

        head_assignment = torch.arange(num_heads, device=current_flat.device, dtype=torch.int64).repeat_interleave(head_lens_i64)
        old_positions = torch.arange(old_total, device=current_flat.device, dtype=torch.int64)
        new_positions = old_positions + head_assignment
        new_flat[new_positions] = current_flat

        head_indices = torch.arange(num_heads, device=current_flat.device, dtype=torch.int64)
        new_token_positions = cu_klen_i64[:num_heads] + head_indices + head_lens_i64
        new_flat[new_token_positions] = appended_flat[:num_heads]

        return new_flat

    def _prepare_flatten_inputs(self, head_lens, cu_klen, ref_tensor):
        head_lens = head_lens.to(device=ref_tensor.device, dtype=torch.int64).contiguous()
        cu_klen = cu_klen.to(device=ref_tensor.device, dtype=torch.int64).contiguous()
        return head_lens, cu_klen

    def _log_update_backend_once(self, backend, layer_idx):
        if self._logged_update_backend:
            return
        print(
            f"[AdaKV-CacheUpdate] backend={backend}, layer={layer_idx}, "
            f"ADAKV_USE_TINY_API_CUDA={os.environ.get('ADAKV_USE_TINY_API_CUDA', '<unset>')}",
            flush=True,
        )
        self._logged_update_backend = True

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            assert self.key_cache[layer_idx].dim() == 2
            bs, head, seqlen, dim = key_states.shape
            assert seqlen == 1
            head_lens = cache_kwargs["head_lens"]
            cu_klen = cache_kwargs["cu_klen"]

            current_key = self.key_cache[layer_idx].reshape(-1, dim).contiguous()
            current_value = self.value_cache[layer_idx].reshape(-1, dim).contiguous()
            appended_key = key_states.reshape(-1, dim).contiguous()
            appended_value = value_states.reshape(-1, dim).contiguous()

            head_lens_prepared = head_lens.to(device=current_key.device, dtype=torch.int64).contiguous()
            cu_klen_prepared = cu_klen.to(device=current_key.device, dtype=torch.int64).contiguous()
            if (
                head_lens_prepared.dim() == 1
                and cu_klen_prepared.dim() == 1
                and cu_klen_prepared.numel() == head_lens_prepared.numel() + 1
                and head_lens_prepared.numel() == appended_key.shape[0]
            ):
                expected_old_total = int(head_lens_prepared.sum().item())
                if current_key.shape[0] != expected_old_total:
                    raise ValueError(
                        f"flatten cache token count mismatch: cache has {current_key.shape[0]}, "
                        f"expected {expected_old_total} from per-segment head_lens"
                    )
                self._log_update_backend_once("vectorized_prevalidated", layer_idx)
                self.key_cache[layer_idx] = self._vectorized_update_flatten_view(
                    current_key, appended_key, head_lens_prepared, cu_klen_prepared
                )
                self.value_cache[layer_idx] = self._vectorized_update_flatten_view(
                    current_value, appended_value, head_lens_prepared, cu_klen_prepared
                )
                return self.key_cache[layer_idx], self.value_cache[layer_idx]

            if layer_idx not in self._validated_layers:
                head_lens, cu_klen, old_tokens_per_batch, new_tokens_per_batch = self._validate_flatten_update_inputs(
                    current_key, appended_key, head_lens, cu_klen, batch_size=bs
                )
                self._validated_layers.add(layer_idx)
            else:
                head_lens, cu_klen = self._prepare_flatten_inputs(head_lens, cu_klen, current_key)

            use_tiny_kernel = bs == 1 and os.environ.get("ADAKV_USE_TINY_API_CUDA", "0") == "1"
            if use_tiny_kernel:
                try:
                    tiny_api_cuda = importlib.import_module("tiny_api_cuda")
                    update_flatten_view = tiny_api_cuda.update_flatten_view

                    head_lens_i32 = head_lens.to(dtype=torch.int32)
                    cu_klen_i32 = cu_klen.to(dtype=torch.int32)
                    new_key_cache = update_flatten_view(current_key, appended_key, head_lens_i32, cu_klen_i32)
                    new_value_cache = update_flatten_view(current_value, appended_value, head_lens_i32, cu_klen_i32)
                    self._log_update_backend_once("tiny_api_cuda", layer_idx)
                except Exception as exc:
                    warnings.warn(
                        f"tiny_api_cuda.update_flatten_view failed at layer {layer_idx}; fallback to vectorized path. error={exc}"
                    )
                    self._log_update_backend_once("vectorized_after_tiny_failure", layer_idx)
                    new_key_cache = self._vectorized_update_flatten_view(current_key, appended_key, head_lens, cu_klen)
                    new_value_cache = self._vectorized_update_flatten_view(current_value, appended_value, head_lens, cu_klen)
            elif bs == 1:
                self._log_update_backend_once("vectorized_env_disabled", layer_idx)
                new_key_cache = self._vectorized_update_flatten_view(current_key, appended_key, head_lens, cu_klen)
                new_value_cache = self._vectorized_update_flatten_view(current_value, appended_value, head_lens, cu_klen)
            else:
                self._log_update_backend_once("vectorized_batched", layer_idx)
                old_tokens_per_batch = int(head_lens.sum().item())
                new_tokens_per_batch = int(head_lens.numel())
                current_key = current_key.view(bs, old_tokens_per_batch, dim)
                current_value = current_value.view(bs, old_tokens_per_batch, dim)
                appended_key = appended_key.view(bs, new_tokens_per_batch, dim)
                appended_value = appended_value.view(bs, new_tokens_per_batch, dim)

                batch_keys = []
                batch_values = []
                for b in range(bs):
                    batch_keys.append(
                        self._vectorized_update_flatten_view(
                            current_key[b], appended_key[b], head_lens, cu_klen
                        )
                    )
                    batch_values.append(
                        self._vectorized_update_flatten_view(
                            current_value[b], appended_value[b], head_lens, cu_klen
                        )
                    )
                new_key_cache = torch.cat(batch_keys, dim=0).contiguous()
                new_value_cache = torch.cat(batch_values, dim=0).contiguous()

            self.key_cache[layer_idx] = new_key_cache
            self.value_cache[layer_idx] = new_value_cache

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        # TODO: return 1 to means has content for now
        return 1
        # return max(map(lambda states: states.shape[-2], self.key_cache[layer_idx]))

    def get_max_length(self) -> Optional[int]:
        return None

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCacheEachHead":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache

# perform qk calculation and get indices
# this version will not update in inference mode

# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def _apply_compression_attention_mask(attn_weights, attention_mask, window_size):
    query_len = attn_weights.shape[-2]
    key_len = attn_weights.shape[-1]

    if attention_mask is not None and attention_mask.dim() == 4:
        additive_mask = attention_mask[:, :, -query_len:, :key_len]
        if additive_mask.shape[-2:] == attn_weights.shape[-2:]:
            return attn_weights + additive_mask.to(device=attn_weights.device, dtype=attn_weights.dtype)

    mask_value = torch.finfo(attn_weights.dtype).min
    if attention_mask is not None and attention_mask.dim() == 2:
        padding_mask = attention_mask[:, -key_len:].to(device=attn_weights.device) == 0
        attn_weights = attn_weights.masked_fill(padding_mask[:, None, None, :], mask_value)

    causal_window = min(window_size, query_len, key_len)
    if causal_window > 0:
        causal_mask = torch.full(
            (causal_window, causal_window),
            mask_value,
            device=attn_weights.device,
            dtype=attn_weights.dtype,
        )
        mask_cond = torch.arange(causal_window, device=attn_weights.device)
        causal_mask.masked_fill_(mask_cond < (mask_cond + 1).view(causal_window, 1), 0)
        attn_weights[:, :, -causal_window:, -causal_window:] += causal_mask[None, None, :, :]

    return attn_weights

def _valid_token_indices_from_attention_mask(attention_mask, batch_idx, q_len, device):
    if attention_mask is None:
        return torch.arange(q_len, device=device, dtype=torch.long)

    if attention_mask.dim() == 2:
        valid_mask = attention_mask[batch_idx, -q_len:].to(device=device) != 0
    elif attention_mask.dim() == 4:
        valid_mask = attention_mask[batch_idx, 0, -1, -q_len:].to(device=device) == 0
    else:
        return torch.arange(q_len, device=device, dtype=torch.long)

    valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten().to(torch.long)
    if valid_indices.numel() == 0:
        return torch.arange(q_len, device=device, dtype=torch.long)
    return valid_indices

def _make_single_sample_attention_mask(attention_mask, seq_len, device):
    dtype = attention_mask.dtype if attention_mask is not None and attention_mask.dim() == 2 else torch.long
    return torch.ones((1, seq_len), device=device, dtype=dtype)

def _init_flatten_metadata_from_head_lens(cluster, head_lens, device):
    head_lens = head_lens.to(device=device, dtype=torch.int32).contiguous()
    cluster.head_lens = head_lens
    cluster.klen_sum = int(head_lens.sum().item())
    cluster.max_seqlen_k = int(head_lens.max().item()) if head_lens.numel() else 0
    cluster.cu_headlens = torch.cumsum(head_lens, dim=0, dtype=torch.int32)
    cluster.cu_klen = torch.cat(
        [head_lens.new_zeros(1), torch.cumsum(head_lens, dim=0, dtype=torch.int32)],
        dim=0,
    )
    cluster.layer_qlens = torch.ones(head_lens.numel(), dtype=torch.int32, device=device)
    cluster.qlen_sum = int(head_lens.numel())
    cluster.cu_qlen = torch.cat(
        [
            cluster.layer_qlens.new_zeros(1),
            torch.cumsum(cluster.layer_qlens, dim=0, dtype=torch.int32),
        ],
        dim=0,
    )
    cluster.cu_offset = torch.arange(0, head_lens.numel() + 1, dtype=torch.int32, device=device)
    cluster.cu_head_offset = torch.arange(1, head_lens.numel() + 1, dtype=torch.int32, device=device)

def _batch_update_as_independent_samples(
    cluster,
    key_states,
    query_states,
    value_states,
    attention_mask,
    update_single_sample,
):
    bsz, _, q_len, head_dim = query_states.shape
    if bsz <= 1:
        return None
    if attention_mask is not None and attention_mask.dim() not in (2, 4):
        return None

    key_outputs = []
    value_outputs = []
    single_clusters = []

    for batch_idx in range(bsz):
        valid_indices = _valid_token_indices_from_attention_mask(
            attention_mask, batch_idx, q_len, query_states.device
        )
        single_key_states = key_states[batch_idx : batch_idx + 1].index_select(2, valid_indices)
        single_query_states = query_states[batch_idx : batch_idx + 1].index_select(2, valid_indices)
        single_value_states = value_states[batch_idx : batch_idx + 1].index_select(2, valid_indices)
        single_attention_mask = _make_single_sample_attention_mask(
            attention_mask, valid_indices.numel(), query_states.device
        )

        single_cluster = copy.copy(cluster)
        single_cluster._compression_started = True
        single_key_out, single_value_out = update_single_sample(
            single_cluster,
            single_key_states,
            single_query_states,
            single_value_states,
            single_attention_mask,
        )
        key_outputs.append(single_key_out)
        value_outputs.append(single_value_out)
        single_clusters.append(single_cluster)

    first_key = key_outputs[0]
    if first_key.dim() == 4:
        max_cache_len = max(key_out.shape[2] for key_out in key_outputs)
        padded_keys = []
        padded_values = []
        valid_masks = []
        for key_out, value_out in zip(key_outputs, value_outputs):
            cache_len = key_out.shape[2]
            pad_len = max_cache_len - cache_len
            if pad_len > 0:
                key_pad = key_out.new_zeros(key_out.shape[0], key_out.shape[1], pad_len, key_out.shape[3])
                value_pad = value_out.new_zeros(value_out.shape[0], value_out.shape[1], pad_len, value_out.shape[3])
                key_out = torch.cat([key_out, key_pad], dim=2)
                value_out = torch.cat([value_out, value_pad], dim=2)
            padded_keys.append(key_out)
            padded_values.append(value_out)
            valid_mask = torch.zeros(max_cache_len, dtype=torch.bool, device=first_key.device)
            valid_mask[:cache_len] = True
            valid_masks.append(valid_mask)

        cluster.cache_valid_mask = torch.stack(valid_masks, dim=0).contiguous()
        cluster.cache_valid_lengths = cluster.cache_valid_mask.sum(dim=-1).to(torch.int32)
        return torch.cat(padded_keys, dim=0).contiguous(), torch.cat(padded_values, dim=0).contiguous()

    flat_keys = torch.cat(key_outputs, dim=0).contiguous()
    flat_values = torch.cat(value_outputs, dim=0).contiguous()
    head_lens = torch.cat(
        [single_cluster.head_lens.reshape(-1).to(device=flat_keys.device) for single_cluster in single_clusters],
        dim=0,
    )
    _init_flatten_metadata_from_head_lens(cluster, head_lens, flat_keys.device)
    return flat_keys, flat_values

def append_cache_valid_mask(cluster, batch_size, new_tokens, device):
    cache_valid_mask = getattr(cluster, "cache_valid_mask", None)
    if cache_valid_mask is None:
        return None
    append_mask = torch.ones((batch_size, new_tokens), dtype=torch.bool, device=device)
    cache_valid_mask = torch.cat([cache_valid_mask.to(device=device), append_mask], dim=-1)
    cluster.cache_valid_mask = cache_valid_mask
    cluster.cache_valid_lengths = cache_valid_mask.sum(dim=-1).to(torch.int32)
    return cache_valid_mask

def get_cache_valid_mask(cluster, key_len, device):
    cache_valid_mask = getattr(cluster, "cache_valid_mask", None)
    if cache_valid_mask is None or cache_valid_mask.shape[-1] != key_len:
        return None
    return cache_valid_mask.to(device=device)

def merge_kv(key_states, value_states, indices, window_size, merge):
    # merge methods in LOOK-M 

    bsz, num_heads, k_len, head_dim = key_states.shape

    # kv-selected
    selected_keys = key_states.gather(dim=2, index=indices)  # [bsz, num_heads, topk_len, head_dim]
    selected_values = value_states.gather(dim=2, index=indices)  # [bsz, num_heads, topk_len, head_dim]

    # kv-drop
    all_indices = torch.arange(k_len, device=key_states.device).unsqueeze(0).unsqueeze(0).expand(bsz, num_heads, k_len)
    all_indices_flattened = all_indices.flatten()  # [bsz * num_heads * (k_len-window_size)]
    selected_indices_flattened = indices.flatten()  # [bsz * num_heads * topk_len]
    is_selected = torch.isin(all_indices_flattened, selected_indices_flattened)
    drop_indices_flattened = all_indices_flattened[~is_selected] 
    drop_len = drop_indices_flattened.shape[0] // (all_indices.shape[0] * all_indices.shape[1])
    drop_indices = drop_indices_flattened.reshape(all_indices.shape[0], all_indices.shape[1], drop_len) # [bsz * num_heads * (k_len-window_size-topk_len)]
    drop_indices = drop_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)  # [bsz, num_heads, (k_len-window_size-topk_len), head_dim]
    drop_keys = key_states.gather(dim=2, index=drop_indices)
    drop_values = value_states.gather(dim=2, index=drop_indices)

    # kv-recent
    recent_keys = key_states[:, :, -window_size:, :]

    ##### apply merge #####
    # prepare for merge
    k_hh_pruned = drop_keys  # [bsz, num_heads, k_len-topk_len-window_size, head_dim]
    k_hh_recent = torch.cat([recent_keys, selected_keys], dim=2)  # [bsz, num_heads, topk_len+window_size, head_dim]
    v_hh_pruned = drop_values  # [bsz, num_heads, k_len-topk_len-window_size, head_dim]
    v_hh_recent = torch.cat([selected_values, value_states[:, :, -window_size:, :]], dim=2)  # [bsz, num_heads, topk_len+window_size, head_dim]
    # similarity matrix
    similarity = (k_hh_pruned / torch.norm(k_hh_pruned, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128)) @ ((k_hh_recent / (torch.norm(k_hh_recent, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128))).transpose(-1, -2)) # cosin
    max_values, max_indices = similarity.max(dim=-1)

    # pivot merge
    if merge=="pivot":
        print("Pivot merge")
        merged_indices = max_indices.unsqueeze(-1).repeat(1, 1, 1, 128)
        k_hh_selected = torch.gather(input=k_hh_recent, dim=2, index=merged_indices)
        k_hh_merged = (k_hh_pruned + k_hh_selected)/2
        k_hh_recent = torch.scatter_reduce(input=k_hh_recent, dim=2, index=merged_indices, src=k_hh_merged, reduce='mean', include_self=True) # include_self=True seems decrease the performance
        v_hh_selected = torch.gather(input=v_hh_recent, dim=2, index=merged_indices)
        v_hh_merged = (v_hh_pruned + v_hh_selected)/2
        v_hh_recent = torch.scatter_reduce(input=v_hh_recent, dim=2, index=merged_indices, src=v_hh_merged, reduce='mean', include_self=True)
    else:
        raise ValueError('Merge method not supported')
        
    # TODO: other merge strategies
    # average merge
    # weight merge

    return k_hh_recent, v_hh_recent

class PyramidKVCluster():
    def __init__(self, num_hidden_layers = 32, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', beta = 20, num_layers = 80, layer_idx=None, merge = None):
        
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers
        
        self.steps = -1
        self.beta = beta
        
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge
        self._compression_started = False

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge
        self._compression_started = False



    def _compute_attn_score(self, attn_weights: torch.Tensor) -> torch.Tensor:
        attn_slice = attn_weights[:, :, -self.window_size:, : -self.window_size]
        return attn_slice.mean(dim=-2)

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        batched_update = _batch_update_as_independent_samples(
            self,
            key_states,
            query_states,
            value_states,
            attention_mask,
            lambda cluster, key, query, value, mask: cluster.update_kv(
                key, query, value, mask, num_key_value_groups
            ),
        )
        if batched_update is not None:
            return batched_update
        
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        # TODO
        # window_sizes = 32
        min_num = (self.max_capacity_prompt - self.window_size) // self.beta
        max_num = (self.max_capacity_prompt - self.window_size) * 2 - min_num
        
        if max_num >= q_len - self.window_size:
            max_num = q_len - self.window_size
            min_num = (self.max_capacity_prompt - self.window_size) * 2 - max_num
    
        steps = (max_num - min_num) // (self.num_hidden_layers - 1)
        max_capacity_prompt = max_num - self.layer_idx * steps
        
        if self.layer_idx == 0:
            log_compression_start(self, "PyramidKV", self.max_capacity_prompt)

        if q_len < self.max_capacity_prompt:
            return key_states, value_states

        elif q_len < (self.max_capacity_prompt - self.window_size) * 2:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            attn_weights = _apply_compression_attention_mask(attn_weights, attention_mask, self.window_size)

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            attn_weights_multiply = self._compute_attn_score(attn_weights)

            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_multiply, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_multiply, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            
            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices

            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            
            if self.merge is not None:
                key_states, value_states = merge_kv(key_states, value_states, indices, self.window_size, self.merge)
                return key_states, value_states

            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)

            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]

            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)

            return key_states, value_states
        
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            attn_weights = _apply_compression_attention_mask(attn_weights, attention_mask, self.window_size)

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_multiply = self._compute_attn_score(attn_weights)

            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_multiply, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_multiply, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
               raise ValueError('Pooling method not supported')

            indices = attn_cache.topk(max_capacity_prompt, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

            if self.merge is not None:
                key_states, value_states = merge_kv(key_states, value_states, indices, self.window_size, self.merge)
                return key_states, value_states

            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)

            return key_states, value_states

class SnapKVCluster():
    def __init__(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None, layer_idx=None, recent_size = 32, ratio =  0.4, num_hidden_layers: int = 32):
        
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers
        self.recent_size = recent_size
        self.ratio = ratio
        self._compression_started = False

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None, recent_size = 32, ratio =  0.4, num_hidden_layers: Optional[int] = None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge
        self.ratio = ratio
        self.recent_size = recent_size
        if num_hidden_layers is not None:
            self.num_hidden_layers = num_hidden_layers
        self._compression_started = False


    def _compute_attn_score(self, attn_weights: torch.Tensor) -> torch.Tensor:
        attn_slice = attn_weights[:, :, -self.window_size:, : -self.window_size]
        return attn_slice.mean(dim=-2)

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        batched_update = _batch_update_as_independent_samples(
            self,
            key_states,
            query_states,
            value_states,
            attention_mask,
            lambda cluster, key, query, value, mask: cluster.update_kv(
                key, query, value, mask, num_key_value_groups
            ),
        )
        if batched_update is not None:
            return batched_update
        
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        if self.layer_idx == 10:
            log_compression_start(self, "SnapKV", self.max_capacity_prompt)
        
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            attn_weights = _apply_compression_attention_mask(attn_weights, attention_mask, self.window_size)

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_score = self._compute_attn_score(attn_weights)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_score, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_score, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

            if self.merge is not None:
                key_states, value_states = merge_kv(key_states, value_states, indices, self.window_size, self.merge)
                return key_states, value_states

            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states

    def update_think(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            attn_weights = _apply_compression_attention_mask(attn_weights, attention_mask, self.window_size)

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_score = self._compute_attn_score(attn_weights)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_score, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_score, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

            if self.merge is not None:
                key_states, value_states = merge_kv(key_states, value_states, indices, self.window_size, self.merge)
                return key_states, value_states

            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            kv_pruned, kv_recent, mask = key_pruner_query_driven(key_states, query_states, self.recent_size, self.ratio)
            return kv_pruned, kv_recent, mask, value_states

class AdaKVCluster():
    '''
    adapt from https://github.com/FFY0/AdaKV.
    '''
    def __init__(self, window_size = 32, kernel_size = 7, pooling = 'avgpool',max_capacity_prompt=None,floor = None,normalize=None, layer_idx = None, num_hidden_layers=None):
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.base_capacity = max_capacity_prompt - window_size
        self.floor_ratio = floor
        self.floor_capacity = int(self.base_capacity * self.floor_ratio)
        self.adaptive_capacity = self.base_capacity - self.floor_capacity
        self.num_hidden_layers = num_hidden_layers

        self.normalize = normalize
        self.layer_idx = layer_idx

        self.head_lens = None
        self.max_seqlen_k = 0
        self.klen_sum = 0
        self.cu_klen = 0
        self.cu_offset = None
        self.cu_headlens = None

    def _shared_batch_head_capacity(self, per_batch_capacity: torch.Tensor, target_sum: int) -> torch.Tensor:
        if per_batch_capacity.dim() != 2:
            raise ValueError("per_batch_capacity must have shape [batch_size, num_heads]")

        per_sample_sum = per_batch_capacity.sum(dim=-1)
        expected = torch.full_like(per_sample_sum, int(target_sum))
        if not torch.equal(per_sample_sum, expected):
            raise ValueError(
                f"head capacity allocation mismatch: got per-sample sums {per_sample_sum.tolist()}, "
                f"expected {int(target_sum)}"
            )

        if per_batch_capacity.shape[0] == 1:
            return per_batch_capacity[0].to(torch.int64)

        mean_capacity = per_batch_capacity.to(torch.float32).mean(dim=0)
        shared_capacity = torch.floor(mean_capacity).to(torch.int64)
        remainder = int(target_sum) - int(shared_capacity.sum().item())
        if remainder > 0:
            fractional = mean_capacity - shared_capacity.to(mean_capacity.dtype)
            order = torch.argsort(fractional, descending=True)
            shared_capacity[order[:remainder]] += 1
        elif remainder < 0:
            fractional = mean_capacity - shared_capacity.to(mean_capacity.dtype)
            order = torch.argsort(fractional, descending=False)
            removable = torch.nonzero(shared_capacity[order] > 0, as_tuple=False).flatten()
            if removable.numel() < -remainder:
                raise ValueError("cannot reduce shared head capacity to target sum")
            shared_capacity[order[removable[: -remainder]]] -= 1

        if int(shared_capacity.sum().item()) != int(target_sum):
            raise ValueError(
                f"shared head capacity sum {int(shared_capacity.sum().item())} does not match target {int(target_sum)}"
            )
        return shared_capacity

    def _gather_batchwise_flatten_kv(
        self,
        origin_heads_key_states,
        origin_heads_value_states,
        sorted_attn_score_indices,
        head_adaptive_capacity,
        head_dim,
    ):
        selected_keys = []
        selected_values = []
        k_lens = []
        klen_sum = 0
        max_seqlen_k = 0

        for head_idx, keep_tensor in enumerate(head_adaptive_capacity):
            keep_k = int(keep_tensor.item())
            cache_index = sorted_attn_score_indices[head_idx][..., :keep_k]

            l = keep_k + self.window_size
            k_lens.append(l)
            max_seqlen_k = max(max_seqlen_k, l)
            klen_sum += l

            cache_index = cache_index.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            top_Kcache = origin_heads_key_states[head_idx].gather(dim=2, index=cache_index)
            top_Vcache = origin_heads_value_states[head_idx].gather(dim=2, index=cache_index)
            selected_k = torch.cat(
                [top_Kcache, origin_heads_key_states[head_idx][:, :, -self.window_size:, :]], dim=2
            ).squeeze(1)
            selected_v = torch.cat(
                [top_Vcache, origin_heads_value_states[head_idx][:, :, -self.window_size:, :]], dim=2
            ).squeeze(1)
            selected_keys.append(selected_k)
            selected_values.append(selected_v)

        bsz = selected_keys[0].shape[0]
        batch_key_chunks = []
        batch_value_chunks = []
        for b in range(bsz):
            batch_key_chunks.extend(selected_k[b] for selected_k in selected_keys)
            batch_value_chunks.extend(selected_v[b] for selected_v in selected_values)

        heads_key_states = torch.cat(batch_key_chunks, dim=0).contiguous()
        heads_value_states = torch.cat(batch_value_chunks, dim=0).contiguous()
        return heads_key_states, heads_value_states, k_lens, klen_sum, max_seqlen_k

    def calculate_attn_score(self, key_states, query_states, attention_mask=None):
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(
            head_dim)
        attn_weights = _apply_compression_attention_mask(attn_weights, attention_mask, self.window_size)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_mean = attn_weights[:, :, -self.window_size:, : -self.window_size].mean(dim=-2)
        if self.pooling == 'avgpool':
            attn_weights_mean_pooling = F.avg_pool1d(attn_weights_mean, kernel_size=self.kernel_size,
                                                     padding=self.kernel_size // 2,
                                                     stride=1)
        elif self.pooling == 'maxpool':
            attn_weights_mean_pooling = F.max_pool1d(attn_weights_mean, kernel_size=self.kernel_size,
                                                     padding=self.kernel_size // 2,
                                                     stride=1)
        else:
            raise ValueError('Pooling method not supported')
        return attn_weights_mean_pooling

    def update_kv(self,  key_states, query_states, value_states, attention_mask=None):
        batched_update = _batch_update_as_independent_samples(
            self,
            key_states,
            query_states,
            value_states,
            attention_mask,
            lambda cluster, key, query, value, mask: cluster.update_kv(key, query, value, mask),
        )
        if batched_update is not None:
            return batched_update

        # check if prefix phase        assert key_states.shape[-2] == query_states.shape[-2]
        _device = key_states.device
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_score= self.calculate_attn_score(key_states,query_states,attention_mask)
        origin_heads_key_states = torch.split(key_states, 1, dim=1)
        origin_heads_value_states = torch.split(value_states, 1, dim=1)

        def init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k):
            # init metadata
            self.head_lens = torch.tensor(k_lens, dtype=torch.int32, device=_device)
            self.klen_sum = klen_sum
            self.max_seqlen_k = max_seqlen_k
            self.cu_headlens = torch.cumsum(self.head_lens, dim=0, dtype=torch.int32)
            # init varlen flash attention metadata
            self.cu_klen = self.cu_headlens - self.head_lens
            self.cu_klen = torch.cat(
                [self.cu_klen, torch.tensor([self.klen_sum], dtype=torch.int32, device=_device)], dim=0)
            self.layer_qlens = torch.ones(num_heads, dtype=torch.int32,device=_device)
            self.qlen_sum = num_heads
            self.cu_qlen = torch.cumsum(self.layer_qlens, dim=0, dtype=torch.int32) - self.layer_qlens
            self.cu_qlen = torch.cat(
                [self.cu_qlen, torch.tensor([self.qlen_sum], dtype=torch.int32, device=_device)], dim=0)
            self.cu_offset = torch.arange(0, num_heads + 1, dtype=torch.int32, device=_device)
            self.cu_head_offset = torch.arange(1, num_heads+1, dtype=torch.int32, device=_device)

        if self.base_capacity > attn_score.size(-1):
            init_metadata(num_heads, [q_len] * num_heads, q_len * num_heads, q_len)
            # not compress
            return key_states.reshape(-1, head_dim), value_states.reshape(-1, head_dim)

        # if you need to weight the attn_score
        sorted_attn_score,sorted_attn_score_indices = attn_score.sort(dim=-1,descending=True)
        adaptive_attn_score = sorted_attn_score
        length = adaptive_attn_score.size(dim=-1)

        if self.normalize:
            ratio_weight = sorted_attn_score[...,:self.base_capacity].sum(dim=-1,keepdim=True)/sorted_attn_score.sum(dim=-1,keepdim=True)
            adaptive_attn_score = adaptive_attn_score*ratio_weight
        adaptive_attn_score = adaptive_attn_score.reshape(bsz,length*num_heads)
        sorted_indices = torch.topk(adaptive_attn_score,k=num_heads*self.base_capacity,dim=-1).indices
        sorted_indices = sorted_indices//length
        # floor capacity set
        head_adaptive_capacity = torch.zeros((bsz,num_heads),device=_device,dtype = sorted_indices.dtype)
        head_adaptive_capacity.scatter_add_(-1,sorted_indices,torch.ones_like(sorted_indices,dtype=head_adaptive_capacity.dtype),)
        head_adaptive_capacity = self._shared_batch_head_capacity(
            head_adaptive_capacity,
            num_heads*self.base_capacity,
        )
        head_adaptive_capacity = torch.round(head_adaptive_capacity * (1-self.floor_ratio) + self.floor_capacity).int()
        sorted_attn_score_indices = sorted_attn_score_indices.split(1,dim=1)

        self.cu_klen = 0

        heads_key_states, heads_value_states, k_lens, klen_sum, max_seqlen_k = self._gather_batchwise_flatten_kv(
            origin_heads_key_states,
            origin_heads_value_states,
            sorted_attn_score_indices,
            head_adaptive_capacity,
            head_dim,
        )

        init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k)

        return heads_key_states,heads_value_states

class AdaptiveKVCluster():
    def __init__(self, window_size = 32, kernel_size = 7, pooling = 'avgpool',base_capacity=2048,floor_alpha = 0.2,skip = 0,normalize=True,
                 layer_idx = None, num_hidden_layers = None, beta = 20, gqa_support=False,num_key_value_groups = 1, gqa_func=None,
                 inspect_attention_weight: bool = False, attention_log_file: Optional[str] = None):
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.max_capacity_prompt = int(base_capacity)
        if self.max_capacity_prompt <= self.window_size:
            raise ValueError("base_capacity must be greater than window_size")
        self.base_capacity = self.max_capacity_prompt - self.window_size

        self.floor_ratio = float(floor_alpha)
        if not 0.0 <= self.floor_ratio <= 1.0:
            raise ValueError("floor_alpha must be in [0, 1]")
        self.floor_capacity = int(self.base_capacity * self.floor_ratio)
        self.adaptive_capacity = self.base_capacity - self.floor_capacity
        self.skip = int(skip)

        self.normalize = bool(normalize)
        self.beta = beta
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers

        # attention inspection switch for debugging one inference sample
        self.inspect_attention_weight = bool(inspect_attention_weight)
        # Always write per-layer/per-head logs under repo_root/Attention_weight.
        # This keeps output location stable even if external code passes a file path.
        self.attention_log_dir = str(Path(__file__).resolve().parents[1] / "Attention_weight")

        # NOTE: layer-wise meta-data
        self.head_lens = None
        self.max_seqlen_k = 0
        self.klen_sum = 0
        self.cu_klen = 0
        self.cu_offset = None
        self.cu_headlens = None

         # support gqa
        self.gqa_support = gqa_support
        self.num_key_value_groups = num_key_value_groups
        self.gqa_func = gqa_func
        if self.gqa_support:
            assert gqa_func is not None, "gqa_func should not be None"
            assert gqa_func in ['max','mean'], "currently gqa_func should be in ['max','mean']"
            if self.num_key_value_groups == 1:
                warnings.warn("gqa_support is enabled, but num_key_value_groups is 1, which means the model is not using gqa. Please check the model configuration.")

    def _apply_pyramid_cache_allocate_strategy(self, q_len):
        # Embed PyramidKV allocation directly: compute layer-wise cache budget every update.
        if self.num_hidden_layers is None or self.num_hidden_layers <= 1 or self.layer_idx is None:
            self.base_capacity = min(self.max_capacity_prompt - self.window_size, q_len - self.window_size)
            self.floor_capacity = int(self.base_capacity * self.floor_ratio)
            self.adaptive_capacity = self.base_capacity - self.floor_capacity
            return

        min_num = (self.max_capacity_prompt - self.window_size) // self.beta
        max_num = (self.max_capacity_prompt - self.window_size) * 2 - min_num

        if max_num >= q_len - self.window_size:
            max_num = q_len - self.window_size
            min_num = (self.max_capacity_prompt - self.window_size) * 2 - max_num

        steps = (max_num - min_num) // (self.num_hidden_layers - 1)
        self.base_capacity = max_num - self.layer_idx * steps
        self.floor_capacity = int(self.base_capacity * self.floor_ratio)
        self.adaptive_capacity = self.base_capacity - self.floor_capacity

    def _print_head_cache_capacity(self, head_adaptive_capacity, q_len):
        if isinstance(head_adaptive_capacity, torch.Tensor):
            head_adaptive_capacity = head_adaptive_capacity.detach().cpu().view(-1).tolist()
        total_capacity = [int(cap) + self.window_size for cap in head_adaptive_capacity]
        layer_total_capacity = sum(total_capacity)
        layer_total_past_capacity = sum(int(cap) for cap in head_adaptive_capacity)
        if self.gqa_support:
            print(
                f"[AdaptiveKVCluster] layer={self.layer_idx}, kv_head_cache_capacity={total_capacity}, "
                f"kv_head_past_capacity={head_adaptive_capacity}, layer_total_cache_capacity={layer_total_capacity}, "
                f"layer_total_past_capacity={layer_total_past_capacity}, q_len={q_len}, window_size={self.window_size}"
            )
        else:
            print(
                f"[AdaptiveKVCluster] layer={self.layer_idx}, head_cache_capacity={total_capacity}, "
                f"head_past_capacity={head_adaptive_capacity}, layer_total_cache_capacity={layer_total_capacity}, "
                f"layer_total_past_capacity={layer_total_past_capacity}, q_len={q_len}, window_size={self.window_size}"
            )

    

    def _compute_attn_score(self, attn_weights: torch.Tensor) -> torch.Tensor:
        attn_slice = attn_weights[:, :, -self.window_size:, : -self.window_size]
        return attn_slice.mean(dim=-2)

    def _select_score(self, attn_slice: torch.Tensor) -> torch.Tensor:
        return attn_slice.mean(dim=-2)


    def _maybe_log_history_attention_mean(self, attn_weights: torch.Tensor) -> None:
        if not self.inspect_attention_weight:
            return
        hist_mean = attn_weights[:, :, -self.window_size:, : -self.window_size].mean(dim=-2)
        hist_mean = hist_mean.detach().to(torch.float32).cpu()
        log_dir = Path(self.attention_log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        for b in range(hist_mean.shape[0]):
            for h in range(hist_mean.shape[1]):
                per_head_log = log_dir / f"layer{self.layer_idx}_head{h}.log"
                with open(per_head_log, "a", encoding="utf-8") as fout:
                    for t in range(hist_mean.shape[2]):
                        fout.write(
                            f"layer={self.layer_idx} batch={b} head={h} token={t} q_len={attn_weights.shape[-1]} mean_attn={float(hist_mean[b, h, t]):.8f}\n"
                        )

    def _shared_batch_head_capacity(self, per_batch_capacity: torch.Tensor, target_sum: int) -> torch.Tensor:
        if per_batch_capacity.dim() != 2:
            raise ValueError("per_batch_capacity must have shape [batch_size, num_heads]")

        per_sample_sum = per_batch_capacity.sum(dim=-1)
        expected = torch.full_like(per_sample_sum, int(target_sum))
        if not torch.equal(per_sample_sum, expected):
            raise ValueError(
                f"head capacity allocation mismatch: got per-sample sums {per_sample_sum.tolist()}, "
                f"expected {int(target_sum)}"
            )

        if per_batch_capacity.shape[0] == 1:
            return per_batch_capacity[0].to(torch.int64)

        mean_capacity = per_batch_capacity.to(torch.float32).mean(dim=0)
        shared_capacity = torch.floor(mean_capacity).to(torch.int64)
        remainder = int(target_sum) - int(shared_capacity.sum().item())
        if remainder > 0:
            fractional = mean_capacity - shared_capacity.to(mean_capacity.dtype)
            order = torch.argsort(fractional, descending=True)
            shared_capacity[order[:remainder]] += 1
        elif remainder < 0:
            fractional = mean_capacity - shared_capacity.to(mean_capacity.dtype)
            order = torch.argsort(fractional, descending=False)
            removable = torch.nonzero(shared_capacity[order] > 0, as_tuple=False).flatten()
            if removable.numel() < -remainder:
                raise ValueError("cannot reduce shared head capacity to target sum")
            shared_capacity[order[removable[: -remainder]]] -= 1

        if int(shared_capacity.sum().item()) != int(target_sum):
            raise ValueError(
                f"shared head capacity sum {int(shared_capacity.sum().item())} does not match target {int(target_sum)}"
            )
        return shared_capacity

    def _apply_floor_capacity(self, head_capacity: torch.Tensor) -> torch.Tensor:
        return torch.round(
            head_capacity.to(torch.float32) * (1 - self.floor_ratio) + self.floor_capacity
        ).to(torch.int32)

    def _gather_batchwise_flatten_kv(
        self,
        origin_heads_key_states,
        origin_heads_value_states,
        sorted_attn_score_indices,
        head_adaptive_capacity,
        head_dim,
    ):
        selected_keys = []
        selected_values = []
        k_lens = []
        klen_sum = 0
        max_seqlen_k = 0

        for head_idx, keep_tensor in enumerate(head_adaptive_capacity):
            keep_k = int(keep_tensor.item())
            cache_index = sorted_attn_score_indices[head_idx][..., :keep_k]

            l = keep_k + self.window_size
            k_lens.append(l)
            max_seqlen_k = max(max_seqlen_k, l)
            klen_sum += l

            cache_index = cache_index.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            top_Kcache = origin_heads_key_states[head_idx].gather(dim=2, index=cache_index)
            top_Vcache = origin_heads_value_states[head_idx].gather(dim=2, index=cache_index)
            selected_k = torch.cat(
                [top_Kcache, origin_heads_key_states[head_idx][:, :, -self.window_size:, :]], dim=2
            ).squeeze(1)
            selected_v = torch.cat(
                [top_Vcache, origin_heads_value_states[head_idx][:, :, -self.window_size:, :]], dim=2
            ).squeeze(1)
            selected_keys.append(selected_k)
            selected_values.append(selected_v)

        bsz = selected_keys[0].shape[0]
        batch_key_chunks = []
        batch_value_chunks = []
        for b in range(bsz):
            batch_key_chunks.extend(selected_k[b] for selected_k in selected_keys)
            batch_value_chunks.extend(selected_v[b] for selected_v in selected_values)

        heads_key_states = torch.cat(batch_key_chunks, dim=0).contiguous()
        heads_value_states = torch.cat(batch_value_chunks, dim=0).contiguous()
        return heads_key_states, heads_value_states, k_lens, klen_sum, max_seqlen_k

    def calculate_attn_score(self, key_states, query_states, attention_mask=None):
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(
            head_dim)
        attn_weights = _apply_compression_attention_mask(attn_weights, attention_mask, self.window_size)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        self._maybe_log_history_attention_mean(attn_weights)
        attn_weights_score = self._compute_attn_score(attn_weights)

        if self.gqa_support:
            attn_weights_score = attn_weights_score.view(attn_weights_score.shape[0],num_heads//self.num_key_value_groups,self.num_key_value_groups,-1)
            if self.gqa_func == 'max':
                attn_weights_score = attn_weights_score.max(dim=-2).values
            elif self.gqa_func == 'mean':
                attn_weights_score = attn_weights_score.mean(dim=-2)
            else:
                raise ValueError('gqa_func not supported')

        if self.pooling == 'avgpool':
            attn_weights_mean_pooling = F.avg_pool1d(attn_weights_score, kernel_size=self.kernel_size,
                                                     padding=self.kernel_size // 2,
                                                     stride=1)
        elif self.pooling == 'maxpool':
            attn_weights_mean_pooling = F.max_pool1d(attn_weights_score, kernel_size=self.kernel_size,
                                                     padding=self.kernel_size // 2,
                                                     stride=1)
        else:
            raise ValueError('Pooling method not supported')
        return attn_weights_mean_pooling

    def update_kv(self, origin_key_states, query_states, origin_value_states, attention_mask=None):
        batched_update = _batch_update_as_independent_samples(
            self,
            origin_key_states,
            query_states,
            origin_value_states,
            attention_mask,
            lambda cluster, key, query, value, mask: cluster.update_kv(key, query, value, mask),
        )
        if batched_update is not None:
            return batched_update

        if self.gqa_support:
            return self.update_kv_gqa(origin_key_states, query_states, origin_value_states, attention_mask)
        else:
            return self.update_kv_wo_gqa(origin_key_states, query_states, origin_value_states, attention_mask)

    def _align_kv_to_query_heads(self, key_states, value_states, query_states):
        """Align KV head count to query head count without double-repeating."""
        q_heads = query_states.shape[1]
        k_heads = key_states.shape[1]
        if k_heads == q_heads:
            return key_states, value_states
        if q_heads % k_heads != 0:
            raise ValueError(
                f"KV head count ({k_heads}) is not compatible with query heads ({q_heads})"
            )
        n_rep = q_heads // k_heads
        return repeat_kv(key_states, n_rep), repeat_kv(value_states, n_rep)

    # 
    def _collapse_repeated_kv_heads(self, key_states, value_states, query_states):
        """Recover KV-head view when inputs are already repeated to query heads."""
        q_heads = query_states.shape[1]
        k_heads = key_states.shape[1]
        if self.num_key_value_groups <= 1:
            return key_states, value_states

        expected_kv_heads = q_heads // self.num_key_value_groups
        if k_heads == expected_kv_heads:
            return key_states, value_states
        if k_heads == q_heads:
            return (
                key_states[:, :: self.num_key_value_groups, :, :],
                value_states[:, :: self.num_key_value_groups, :, :],
            )
        raise ValueError(
            f"Cannot collapse KV heads: got {k_heads}, expected {expected_kv_heads} or {q_heads}"
        )

    # update kv with gqa_support
    def update_kv_gqa(self, origin_key_states, query_states, origin_value_states, attention_mask=None):
        key_states, value_states = self._align_kv_to_query_heads(
            origin_key_states, origin_value_states, query_states
        )
        kv_key_states, kv_value_states = self._collapse_repeated_kv_heads(
            origin_key_states, origin_value_states, query_states
        )

        # check if prefix phase        assert key_states.shape[-2] == query_states.shape[-2]
        _device = key_states.device
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_score= self.calculate_attn_score(key_states,query_states,attention_mask)
        origin_heads_key_states = torch.split(kv_key_states, 1, dim=1)
        origin_heads_value_states = torch.split(kv_value_states, 1, dim=1)

        self._apply_pyramid_cache_allocate_strategy(q_len)

        def init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k):
            # init metadata
            self.head_lens = torch.tensor(k_lens, dtype=torch.int32, device=_device)
            self.klen_sum = klen_sum
            self.max_seqlen_k = max_seqlen_k
            self.cu_headlens = torch.cumsum(self.head_lens, dim=0, dtype=torch.int32)
            # init varlen flash attention metadata
            self.cu_klen = self.cu_headlens - self.head_lens
            self.cu_klen = torch.cat(
                [self.cu_klen, torch.tensor([self.klen_sum], dtype=torch.int32, device=_device)], dim=0)
            # check bug
            self.layer_qlens = torch.ones(num_heads//self.num_key_value_groups, dtype=torch.int32,device=_device)
            self.qlen_sum = num_heads//self.num_key_value_groups
            self.cu_qlen = torch.cumsum(self.layer_qlens, dim=0, dtype=torch.int32) - self.layer_qlens
            self.cu_qlen = torch.cat(
                [self.cu_qlen, torch.tensor([self.qlen_sum], dtype=torch.int32, device=_device)], dim=0)

            if self.gqa_support:
                self.cu_offset = torch.arange(0, num_heads//self.num_key_value_groups + 1, dtype=torch.int32, device=_device)
                self.cu_head_offset = torch.arange(1, num_heads//self.num_key_value_groups +1, dtype=torch.int32, device=_device)

            else:
                self.cu_offset = torch.arange(0, num_heads + 1, dtype=torch.int32, device=_device)
                self.cu_head_offset = torch.arange(1, num_heads+1, dtype=torch.int32, device=_device)

        if self.base_capacity > attn_score.size(-1):
            init_metadata(num_heads, [q_len] * (num_heads//self.num_key_value_groups), q_len * (num_heads//self.num_key_value_groups), q_len)
            # self._print_head_cache_capacity([q_len] * (num_heads // self.num_key_value_groups), q_len)
            # not compress
            return kv_key_states.reshape(-1, head_dim), kv_value_states.reshape(-1, head_dim)

        sorted_attn_score,sorted_attn_score_indices = attn_score.sort(dim=-1,descending=True)
        if self.layer_idx >= self.skip:
            adaptive_attn_score = sorted_attn_score
            length = adaptive_attn_score.size(dim=-1)
            if self.normalize:
                ratio_weight = sorted_attn_score[...,:self.base_capacity].sum(dim=-1,keepdim=True)/sorted_attn_score.sum(dim=-1,keepdim=True)
                adaptive_attn_score = adaptive_attn_score*ratio_weight
            adaptive_attn_score = adaptive_attn_score.reshape(bsz,length*num_heads//self.num_key_value_groups)
            sorted_indices = torch.topk(adaptive_attn_score,k=num_heads*self.base_capacity//self.num_key_value_groups,dim=-1).indices
            sorted_indices = sorted_indices//length

            # floor_alpha capacity set
            head_adaptive_capacity = torch.zeros((bsz,num_heads//self.num_key_value_groups),device=_device,dtype = sorted_indices.dtype)
            head_adaptive_capacity.scatter_add_(-1,sorted_indices,torch.ones_like(sorted_indices,dtype=head_adaptive_capacity.dtype),)
            head_adaptive_capacity = self._shared_batch_head_capacity(
                head_adaptive_capacity,
                num_heads*self.base_capacity//self.num_key_value_groups,
            )
            head_adaptive_capacity = self._apply_floor_capacity(head_adaptive_capacity)
        else:
            head_adaptive_capacity = torch.ones(
                (num_heads//self.num_key_value_groups,),
                device=_device,
                dtype=torch.int32,
            ) * self.base_capacity
        # self._print_head_cache_capacity(head_adaptive_capacity, q_len)
        sorted_attn_score_indices = sorted_attn_score_indices.split(1,dim=1)

        self.cu_klen = 0

        heads_key_states, heads_value_states, k_lens, klen_sum, max_seqlen_k = self._gather_batchwise_flatten_kv(
            origin_heads_key_states,
            origin_heads_value_states,
            sorted_attn_score_indices,
            head_adaptive_capacity,
            head_dim,
        )

        init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k)

        return heads_key_states, heads_value_states

    # update without gqa_support
    def update_kv_wo_gqa(self,  origin_key_states, query_states, origin_value_states, attention_mask=None):
        key_states, value_states = self._align_kv_to_query_heads(
            origin_key_states, origin_value_states, query_states
        )

        # check if prefix phase        assert key_states.shape[-2] == query_states.shape[-2]
        _device = key_states.device
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_score= self.calculate_attn_score(key_states,query_states,attention_mask)
        origin_heads_key_states = torch.split(key_states, 1, dim=1)
        origin_heads_value_states = torch.split(value_states, 1, dim=1)

        self._apply_pyramid_cache_allocate_strategy(q_len)

        def init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k):
            # init metadata
            self.head_lens = torch.tensor(k_lens, dtype=torch.int32, device=_device)
            self.klen_sum = klen_sum
            self.max_seqlen_k = max_seqlen_k
            self.cu_headlens = torch.cumsum(self.head_lens, dim=0, dtype=torch.int32)
            # init varlen flash attention metadata
            self.cu_klen = self.cu_headlens - self.head_lens
            self.cu_klen = torch.cat(
                [self.cu_klen, torch.tensor([self.klen_sum], dtype=torch.int32, device=_device)], dim=0)
            self.layer_qlens = torch.ones(num_heads, dtype=torch.int32,device=_device)
            self.qlen_sum = num_heads
            self.cu_qlen = torch.cumsum(self.layer_qlens, dim=0, dtype=torch.int32) - self.layer_qlens
            self.cu_qlen = torch.cat(
                [self.cu_qlen, torch.tensor([self.qlen_sum], dtype=torch.int32, device=_device)], dim=0)
            self.cu_offset = torch.arange(0, num_heads + 1, dtype=torch.int32, device=_device)
            self.cu_head_offset = torch.arange(1, num_heads+1, dtype=torch.int32, device=_device)

        if self.base_capacity > attn_score.size(-1):
            init_metadata(num_heads, [q_len] * num_heads, q_len * num_heads, q_len)
            # self._print_head_cache_capacity([q_len] * num_heads, q_len)
            # not compress
            return key_states.reshape(-1, head_dim), value_states.reshape(-1, head_dim)

        # if you need to weight the attn_score
        pass
        sorted_attn_score,sorted_attn_score_indices = attn_score.sort(dim=-1,descending=True)

        if self.layer_idx >= self.skip:
            adaptive_attn_score = sorted_attn_score
            length = adaptive_attn_score.size(dim=-1)
            if self.normalize:
                ratio_weight = sorted_attn_score[...,:self.base_capacity].sum(dim=-1,keepdim=True)/sorted_attn_score.sum(dim=-1,keepdim=True)
                adaptive_attn_score = adaptive_attn_score*ratio_weight

            adaptive_attn_score = adaptive_attn_score.reshape(bsz,length*num_heads)

            sorted_indices = torch.topk(adaptive_attn_score,k=num_heads*self.base_capacity,dim=-1).indices
            sorted_indices = sorted_indices//length

            # floor_alpha capacity set
            head_adaptive_capacity = torch.zeros((bsz,num_heads),device=_device,dtype = sorted_indices.dtype)
            head_adaptive_capacity.scatter_add_(-1,sorted_indices,torch.ones_like(sorted_indices,dtype=head_adaptive_capacity.dtype),)
            head_adaptive_capacity = self._shared_batch_head_capacity(
                head_adaptive_capacity,
                num_heads*self.base_capacity,
            )
            head_adaptive_capacity = self._apply_floor_capacity(head_adaptive_capacity)
        else:
            head_adaptive_capacity = torch.ones(
                (num_heads,),
                device=_device,
                dtype=torch.int32,
            ) * self.base_capacity
        # self._print_head_cache_capacity(head_adaptive_capacity, q_len)
        sorted_attn_score_indices = sorted_attn_score_indices.split(1,dim=1)

        self.cu_klen = 0

        heads_key_states, heads_value_states, k_lens, klen_sum, max_seqlen_k = self._gather_batchwise_flatten_kv(
            origin_heads_key_states,
            origin_heads_value_states,
            sorted_attn_score_indices,
            head_adaptive_capacity,
            head_dim,
        )

        init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k)

        return heads_key_states,heads_value_states

class ChampionKVCluster():
    """
    Adaptive KV compression with head-wise reallocation driven by inverse max attention.
    """

    def __init__(
        self,
        window_size=32,
        kernel_size=7,
        pooling='avgpool',
        base_capacity=2048,
        allocate_strategy: str = "pyramid",
        min_head_budget_ratio: float = 0.2,
        abnormal_head_max_tau: float = 0.0001,
        layer_idx=None,
        num_hidden_layers=None,
        beta=20,
        gqa_support=False,
        num_key_value_groups=1,
        gqa_func=None,
        inspect_attention_weight: bool = False,
        attention_log_file: Optional[str] = None,
    ):
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.max_capacity_prompt = int(base_capacity)
        if self.max_capacity_prompt <= self.window_size:
            raise ValueError("base_capacity must be greater than window_size")
        self.base_capacity = self.max_capacity_prompt - self.window_size

        self.allocate_strategy = str(allocate_strategy)
        if self.allocate_strategy not in ["pyramid", "uniform"]:
            raise ValueError("allocate_strategy must be in ['pyramid', 'uniform']")

        if not (0.0 <= float(min_head_budget_ratio) <= 1.0):
            raise ValueError(
                f"min_head_budget_ratio must be within [0, 1], got {min_head_budget_ratio}."
            )
        if float(abnormal_head_max_tau) < 0.0:
            raise ValueError(
                f"abnormal_head_max_tau must be non-negative, got {abnormal_head_max_tau}."
            )
        self.min_head_budget_ratio = float(min_head_budget_ratio)
        self.abnormal_head_max_tau = float(abnormal_head_max_tau)

        self.beta = beta
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers

        self.inspect_attention_weight = bool(inspect_attention_weight)
        self.attention_log_dir = str(Path(__file__).resolve().parents[1] / "Attention_weight")

        self.head_lens = None
        self.max_seqlen_k = 0
        self.klen_sum = 0
        self.cu_klen = 0
        self.cu_offset = None
        self.cu_headlens = None

        self.gqa_support = gqa_support
        self.num_key_value_groups = num_key_value_groups
        self.gqa_func = gqa_func
        if self.gqa_support:
            assert gqa_func is not None, "gqa_func should not be None"
            assert gqa_func in ['max', 'mean'], "currently gqa_func should be in ['max','mean']"
            if self.num_key_value_groups == 1:
                warnings.warn(
                    "gqa_support is enabled, but num_key_value_groups is 1, which means the model is not using gqa. "
                    "Please check the model configuration."
                )

    def reset(
        self,
        window_size=32,
        kernel_size=7,
        pooling='avgpool',
        base_capacity=2048,
        allocate_strategy: str = "pyramid",
        min_head_budget_ratio: float = 0.2,
        abnormal_head_max_tau: float = 0.0001,
        layer_idx=None,
        num_hidden_layers=None,
        beta=20,
        gqa_support=False,
        num_key_value_groups=1,
        gqa_func=None,
        inspect_attention_weight: Optional[bool] = None,
        attention_log_file: Optional[str] = None,
    ):
        self.__init__(
            window_size=window_size,
            kernel_size=kernel_size,
            pooling=pooling,
            base_capacity=base_capacity,
            allocate_strategy=allocate_strategy,
            min_head_budget_ratio=min_head_budget_ratio,
            abnormal_head_max_tau=abnormal_head_max_tau,
            layer_idx=layer_idx,
            num_hidden_layers=num_hidden_layers,
            beta=beta,
            gqa_support=gqa_support,
            num_key_value_groups=num_key_value_groups,
            gqa_func=gqa_func,
            inspect_attention_weight=self.inspect_attention_weight if inspect_attention_weight is None else inspect_attention_weight,
            attention_log_file=attention_log_file,
        )

    def _apply_layer_budget(self, q_len: int) -> None:
        if self.allocate_strategy == "uniform":
            self.base_capacity = min(self.max_capacity_prompt - self.window_size, q_len - self.window_size)
            self.base_capacity = max(0, int(self.base_capacity))
            return
        if self.num_hidden_layers is None or self.num_hidden_layers <= 1 or self.layer_idx is None:
            self.base_capacity = min(self.max_capacity_prompt - self.window_size, q_len - self.window_size)
            self.base_capacity = max(0, int(self.base_capacity))
            return

        min_num = (self.max_capacity_prompt - self.window_size) // self.beta
        max_num = (self.max_capacity_prompt - self.window_size) * 2 - min_num

        if max_num >= q_len - self.window_size:
            max_num = q_len - self.window_size
            min_num = (self.max_capacity_prompt - self.window_size) * 2 - max_num

        steps = (max_num - min_num) // (self.num_hidden_layers - 1)
        self.base_capacity = max_num - self.layer_idx * steps

    def _compute_attn_score(self, attn_weights: torch.Tensor) -> torch.Tensor:
        attn_slice = attn_weights[:, :, -self.window_size:, : -self.window_size]
        return attn_slice.mean(dim=-2)

    def _maybe_log_history_attention_mean(self, attn_weights: torch.Tensor) -> None:
        if not self.inspect_attention_weight:
            return
        hist_mean = attn_weights[:, :, -self.window_size:, : -self.window_size].mean(dim=-2)
        hist_mean = hist_mean.detach().to(torch.float32).cpu()
        log_dir = Path(self.attention_log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        for b in range(hist_mean.shape[0]):
            for h in range(hist_mean.shape[1]):
                per_head_log = log_dir / f"layer{self.layer_idx}_head{h}.log"
                with open(per_head_log, "a", encoding="utf-8") as fout:
                    for t in range(hist_mean.shape[2]):
                        fout.write(
                            f"layer={self.layer_idx} batch={b} head={h} token={t} q_len={attn_weights.shape[-1]} mean_attn={float(hist_mean[b, h, t]):.8f}\n"
                        )

    def calculate_attn_score(self, key_states, query_states, attention_mask=None):
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(
            head_dim
        )
        attn_weights = _apply_compression_attention_mask(attn_weights, attention_mask, self.window_size)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        self._maybe_log_history_attention_mean(attn_weights)
        attn_weights_score = self._compute_attn_score(attn_weights)

        if self.gqa_support:
            attn_weights_score = attn_weights_score.view(
                attn_weights_score.shape[0], num_heads // self.num_key_value_groups, self.num_key_value_groups, -1
            )
            if self.gqa_func == 'max':
                attn_weights_score = attn_weights_score.max(dim=-2).values
            elif self.gqa_func == 'mean':
                attn_weights_score = attn_weights_score.mean(dim=-2)
            else:
                raise ValueError('gqa_func not supported')

        return attn_weights_score

    def _align_kv_to_query_heads(self, key_states, value_states, query_states):
        q_heads = query_states.shape[1]
        k_heads = key_states.shape[1]
        if k_heads == q_heads:
            return key_states, value_states
        if q_heads % k_heads != 0:
            raise ValueError(
                f"KV head count ({k_heads}) is not compatible with query heads ({q_heads})"
            )
        n_rep = q_heads // k_heads
        return repeat_kv(key_states, n_rep), repeat_kv(value_states, n_rep)

    def _collapse_repeated_kv_heads(self, key_states, value_states, query_states):
        q_heads = query_states.shape[1]
        k_heads = key_states.shape[1]
        if self.num_key_value_groups <= 1:
            return key_states, value_states

        expected_kv_heads = q_heads // self.num_key_value_groups
        if k_heads == expected_kv_heads:
            return key_states, value_states
        if k_heads == q_heads:
            return (
                key_states[:, :: self.num_key_value_groups, :, :],
                value_states[:, :: self.num_key_value_groups, :, :],
            )
        raise ValueError(
            f"Cannot collapse KV heads: got {k_heads}, expected {expected_kv_heads} or {q_heads}"
        )

    def _compute_head_weights(self, attn_score: torch.Tensor) -> torch.Tensor:
        attn_score_f32 = attn_score.to(torch.float32)
        head_max = attn_score_f32.max(dim=-1).values
        self._last_head_max = head_max
        eps = torch.finfo(attn_score_f32.dtype).eps
        inv_max = 1.0 / (head_max + eps)
        inv_sum = inv_max.sum(dim=-1, keepdim=True)
        if torch.any(inv_sum <= 0):
            num_heads = inv_max.shape[-1]
            uniform = torch.full_like(inv_max, 1.0 / float(num_heads))
            return torch.where(inv_sum > 0, inv_max / inv_sum, uniform)
        weights_L1 = inv_max / inv_sum
        # weights_L2 = weights_L1 / (weights_L1.sum(dim=-1, keepdim=True) + eps)

        # temperature = 1
        # weights_softmax = torch.nn.functional.softmax(inv_max * temperature, dim=-1)
        return weights_L1

    def _compute_abnormal_head_mask(
        self,
        attn_score: torch.Tensor,
        head_max: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.abnormal_head_max_tau <= 0.0:
            return torch.zeros(
                attn_score.shape[0],
                attn_score.shape[1],
                dtype=torch.bool,
                device=attn_score.device,
            )
        if head_max is None:
            cached_head_max = getattr(self, "_last_head_max", None)
            if cached_head_max is not None and cached_head_max.shape == attn_score.shape[:2]:
                head_max = cached_head_max.to(device=attn_score.device)
            else:
                head_max = attn_score.to(torch.float32).max(dim=-1).values
        return head_max.to(torch.float32) < self.abnormal_head_max_tau

    def _allocate_head_budget_masked(
        self,
        head_weights: torch.Tensor,
        total_budget: int,
        score_len: int,
        eligible_mask: torch.Tensor,
    ) -> torch.Tensor:
        if total_budget <= 0 or score_len <= 0:
            return torch.zeros(head_weights.shape, dtype=torch.int32, device=head_weights.device)

        num_heads = head_weights.shape[-1]
        alloc_total_budget = min(int(total_budget), int(num_heads * score_len))
        budgets = torch.zeros(head_weights.shape, dtype=torch.int32, device=head_weights.device)
        if alloc_total_budget <= 0:
            return budgets

        bsz = head_weights.shape[0]
        for b in range(bsz):
            eligible = eligible_mask[b]
            if not torch.any(eligible):
                continue
            weights = head_weights[b].clone()
            weights[~eligible] = 0
            weight_sum = weights.sum()
            if weight_sum <= 0:
                weights = eligible.to(torch.float32)
                weight_sum = weights.sum()
                if weight_sum <= 0:
                    continue
            add = torch.floor(weights / weight_sum * float(alloc_total_budget)).to(torch.int32)
            add = torch.minimum(add, torch.full_like(add, score_len))
            add = torch.where(eligible, add, torch.zeros_like(add))
            add_sum = int(add.sum().item())
            remaining = alloc_total_budget - add_sum
            if remaining > 0:
                _, order = torch.sort(weights, descending=True)
                for idx in order.tolist():
                    if remaining <= 0:
                        break
                    if not bool(eligible[idx]):
                        continue
                    if add[idx] >= score_len:
                        continue
                    add[idx] += 1
                    remaining -= 1
            budgets[b] = add
        return budgets

    def _allocate_head_budget_with_floor(
        self,
        head_weights: torch.Tensor,
        attn_score: torch.Tensor,
        total_budget: int,
        score_len: int,
        head_max: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if total_budget <= 0 or score_len <= 0:
            return torch.zeros(head_weights.shape, dtype=torch.int32, device=head_weights.device)

        num_heads = head_weights.shape[-1]

        alloc_total_budget = min(int(total_budget), int(num_heads * score_len))

        min_head_budget = int(math.floor(self.base_capacity * self.min_head_budget_ratio))
        min_head_budget = max(0, min(min_head_budget, score_len))

        budgets = torch.zeros(head_weights.shape, dtype=torch.int32, device=head_weights.device)

        if min_head_budget > 0:
            floor_budget = min(min_head_budget, alloc_total_budget // max(num_heads, 1))
            budgets.fill_(floor_budget)

        # All heads receive the same floor budget first. Abnormal heads are then
        # excluded from the adaptive allocation and keep only this floor budget.
        abnormal_mask = self._compute_abnormal_head_mask(attn_score, head_max=head_max)

        for b in range(head_weights.shape[0]):
            abnormal = abnormal_mask[b]
            remaining = int(alloc_total_budget) - int(budgets[b].sum().item())
            if remaining <= 0:
                continue

            eligible = ~abnormal
            max_extra = max(score_len - int(budgets[b].max().item()), 0)
            if not torch.any(eligible) or max_extra <= 0:
                continue

            extra_budget = self._allocate_head_budget_masked(
                head_weights[b : b + 1],
                remaining,
                max_extra,
                eligible.unsqueeze(0),
            )[0]
            budgets[b] = torch.clamp(budgets[b] + extra_budget, max=score_len)

        return budgets

    def _redistribute_remaining(self, head_budget, head_weights, score_len, total_budget):
        bsz = head_budget.shape[0]
        budgets = head_budget.clone()
        for b in range(bsz):
            remaining = int(total_budget) - int(budgets[b].sum().item())
            if remaining <= 0:
                continue
            budgets[b] = self._redistribute_remaining_one(
                budgets[b], head_weights[b], score_len, remaining
            )
        return budgets
    def _redistribute_remaining_one(self, head_budget, head_weights, score_len, remaining):
        head_budget = head_budget.clone()
        head_weights = head_weights.clone()
        while remaining > 0:
            eligible = head_budget < score_len
            if not torch.any(eligible):
                break
            weights = head_weights.clone()
            weights[~eligible] = 0
            weight_sum = weights.sum()
            if weight_sum <= 0:
                weights = eligible.to(torch.float32)
                weight_sum = weights.sum()
                if weight_sum <= 0:
                    break
            add = torch.floor(weights / weight_sum * float(remaining)).to(torch.int32)
            add = torch.minimum(add, (score_len - head_budget))
            add_sum = int(add.sum().item())
            if add_sum == 0:
                _, order = torch.sort(weights, descending=True)
                for idx in order.tolist():
                    if remaining <= 0:
                        break
                    if head_budget[idx] >= score_len:
                        continue
                    head_budget[idx] += 1
                    remaining -= 1
                continue
            head_budget += add
            remaining -= add_sum
        return head_budget

    def _allocate_head_budget(self, head_weights: torch.Tensor, total_budget: int, score_len: int) -> torch.Tensor:
        if total_budget <= 0 or score_len <= 0:
            return torch.zeros(head_weights.shape, dtype=torch.int32, device=head_weights.device)

        num_heads = head_weights.shape[-1]
        alloc_total_budget = min(int(total_budget), int(num_heads * score_len))
        raw_budget = torch.floor(head_weights * float(alloc_total_budget)).to(torch.int32)
        head_budget = torch.clamp(raw_budget, min=0, max=score_len)
        head_budget = self._redistribute_remaining(head_budget, head_weights, score_len, alloc_total_budget)
        return head_budget

    def _init_metadata(self, num_heads, k_lens, klen_sum, max_seqlen_k, device):
        self.head_lens = torch.tensor(k_lens, dtype=torch.int32, device=device)
        self.klen_sum = klen_sum
        self.max_seqlen_k = max_seqlen_k
        self.cu_headlens = torch.cumsum(self.head_lens, dim=0, dtype=torch.int32)
        self.cu_klen = self.cu_headlens - self.head_lens
        self.cu_klen = torch.cat(
            [self.cu_klen, torch.tensor([self.klen_sum], dtype=torch.int32, device=device)], dim=0)

        if self.gqa_support:
            q_heads = num_heads // self.num_key_value_groups
            self.layer_qlens = torch.ones(q_heads, dtype=torch.int32, device=device)
            self.qlen_sum = q_heads
            self.cu_qlen = torch.cumsum(self.layer_qlens, dim=0, dtype=torch.int32) - self.layer_qlens
            self.cu_qlen = torch.cat(
                [self.cu_qlen, torch.tensor([self.qlen_sum], dtype=torch.int32, device=device)], dim=0)
            self.cu_offset = torch.arange(0, q_heads + 1, dtype=torch.int32, device=device)
            self.cu_head_offset = torch.arange(1, q_heads + 1, dtype=torch.int32, device=device)
        else:
            self.layer_qlens = torch.ones(num_heads, dtype=torch.int32, device=device)
            self.qlen_sum = num_heads
            self.cu_qlen = torch.cumsum(self.layer_qlens, dim=0, dtype=torch.int32) - self.layer_qlens
            self.cu_qlen = torch.cat(
                [self.cu_qlen, torch.tensor([self.qlen_sum], dtype=torch.int32, device=device)], dim=0)
            self.cu_offset = torch.arange(0, num_heads + 1, dtype=torch.int32, device=device)
            self.cu_head_offset = torch.arange(1, num_heads + 1, dtype=torch.int32, device=device)

    def _shared_batch_head_capacity(self, per_batch_capacity: torch.Tensor, target_sum: int) -> torch.Tensor:
        if per_batch_capacity.dim() != 2:
            raise ValueError("per_batch_capacity must have shape [batch_size, num_heads]")

        if per_batch_capacity.shape[0] == 1:
            return per_batch_capacity[0].to(torch.int64)

        per_sample_sums = per_batch_capacity.sum(dim=-1).to(torch.float32)
        if not torch.all(per_sample_sums == float(target_sum)):
            target_sum = int(torch.round(per_sample_sums.mean()).item())

        mean_capacity = per_batch_capacity.to(torch.float32).mean(dim=0)
        shared_capacity = torch.floor(mean_capacity).to(torch.int64)
        remainder = int(target_sum) - int(shared_capacity.sum().item())
        if remainder > 0:
            fractional = mean_capacity - shared_capacity.to(mean_capacity.dtype)
            order = torch.argsort(fractional, descending=True)
            shared_capacity[order[:remainder]] += 1
        elif remainder < 0:
            fractional = mean_capacity - shared_capacity.to(mean_capacity.dtype)
            order = torch.argsort(fractional, descending=False)
            removable = torch.nonzero(shared_capacity[order] > 0, as_tuple=False).flatten()
            if removable.numel() < -remainder:
                raise ValueError("cannot reduce shared head capacity to target sum")
            shared_capacity[order[removable[: -remainder]]] -= 1

        if int(shared_capacity.sum().item()) != int(target_sum):
            raise ValueError(
                f"shared head capacity sum {int(shared_capacity.sum().item())} does not match target {int(target_sum)}"
            )
        return shared_capacity

    def _gather_batchwise_flatten_kv(
        self,
        origin_heads_key_states,
        origin_heads_value_states,
        sorted_attn_score_indices,
        head_adaptive_capacity,
        head_dim,
    ):
        selected_keys = []
        selected_values = []
        k_lens = []
        klen_sum = 0
        max_seqlen_k = 0

        for head_idx, keep_tensor in enumerate(head_adaptive_capacity):
            keep_k = int(keep_tensor.item())
            cache_index = sorted_attn_score_indices[head_idx][..., :keep_k]

            l = keep_k + self.window_size
            k_lens.append(l)
            max_seqlen_k = max(max_seqlen_k, l)
            klen_sum += l

            cache_index = cache_index.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            top_Kcache = origin_heads_key_states[head_idx].gather(dim=2, index=cache_index)
            top_Vcache = origin_heads_value_states[head_idx].gather(dim=2, index=cache_index)
            selected_k = torch.cat(
                [top_Kcache, origin_heads_key_states[head_idx][:, :, -self.window_size:, :]], dim=2
            ).squeeze(1)
            selected_v = torch.cat(
                [top_Vcache, origin_heads_value_states[head_idx][:, :, -self.window_size:, :]], dim=2
            ).squeeze(1)
            selected_keys.append(selected_k)
            selected_values.append(selected_v)

        bsz = selected_keys[0].shape[0]
        batch_key_chunks = []
        batch_value_chunks = []
        for b in range(bsz):
            batch_key_chunks.extend(selected_k[b] for selected_k in selected_keys)
            batch_value_chunks.extend(selected_v[b] for selected_v in selected_values)

        heads_key_states = torch.cat(batch_key_chunks, dim=0).contiguous()
        heads_value_states = torch.cat(batch_value_chunks, dim=0).contiguous()
        return heads_key_states, heads_value_states, k_lens, klen_sum, max_seqlen_k

    def update_kv(self, origin_key_states, query_states, origin_value_states, attention_mask=None):
        batched_update = _batch_update_as_independent_samples(
            self,
            origin_key_states,
            query_states,
            origin_value_states,
            attention_mask,
            lambda cluster, key, query, value, mask: cluster.update_kv(key, query, value, mask),
        )
        if batched_update is not None:
            return batched_update

        if self.gqa_support:
            return self.update_kv_gqa(origin_key_states, query_states, origin_value_states, attention_mask)
        return self.update_kv_wo_gqa(origin_key_states, query_states, origin_value_states, attention_mask)

    def update_kv_gqa(self, origin_key_states, query_states, origin_value_states, attention_mask=None):
        key_states, value_states = self._align_kv_to_query_heads(
            origin_key_states, origin_value_states, query_states
        )
        kv_key_states, kv_value_states = self._collapse_repeated_kv_heads(
            origin_key_states, origin_value_states, query_states
        )

        _device = key_states.device
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_score = self.calculate_attn_score(key_states, query_states, attention_mask)
        origin_heads_key_states = torch.split(kv_key_states, 1, dim=1)
        origin_heads_value_states = torch.split(kv_value_states, 1, dim=1)

        self._apply_layer_budget(q_len)

        if self.base_capacity > attn_score.size(-1):
            kv_heads = num_heads // self.num_key_value_groups
            self._init_metadata(num_heads, [q_len] * kv_heads, q_len * kv_heads, q_len, _device)
            return kv_key_states.reshape(-1, head_dim), kv_value_states.reshape(-1, head_dim)

        sorted_attn_score_indices = attn_score.sort(dim=-1, descending=True).indices
        kv_heads = num_heads // self.num_key_value_groups
        total_budget = kv_heads * self.base_capacity
        head_weights = self._compute_head_weights(attn_score)
        head_adaptive_capacity = self._allocate_head_budget_with_floor(
            head_weights, attn_score, total_budget, attn_score.size(-1)
        )
        head_adaptive_capacity = self._shared_batch_head_capacity(head_adaptive_capacity, total_budget)
        sorted_attn_score_indices = sorted_attn_score_indices.split(1, dim=1)

        self.cu_klen = 0

        heads_key_states, heads_value_states, k_lens, klen_sum, max_seqlen_k = self._gather_batchwise_flatten_kv(
            origin_heads_key_states,
            origin_heads_value_states,
            sorted_attn_score_indices,
            head_adaptive_capacity,
            head_dim,
        )

        self._init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k, _device)
        return heads_key_states, heads_value_states

    def update_kv_wo_gqa(self, origin_key_states, query_states, origin_value_states, attention_mask=None):
        key_states, value_states = self._align_kv_to_query_heads(
            origin_key_states, origin_value_states, query_states
        )

        _device = key_states.device
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_score = self.calculate_attn_score(key_states, query_states, attention_mask)

        if self.pooling == 'avgpool':
            attn_score = F.avg_pool1d(attn_score, kernel_size=self.kernel_size,
                                                     padding=self.kernel_size // 2,
                                                     stride=1)
        elif self.pooling == 'maxpool':
            attn_score = F.max_pool1d(attn_score, kernel_size=self.kernel_size,
                                                     padding=self.kernel_size // 2,
                                                     stride=1)
        else:
            raise ValueError('Pooling method not supported')

        origin_heads_key_states = torch.split(key_states, 1, dim=1)
        origin_heads_value_states = torch.split(value_states, 1, dim=1)
        self._apply_layer_budget(q_len)

        if self.base_capacity > attn_score.size(-1):
            self._init_metadata(num_heads, [q_len] * num_heads, q_len * num_heads, q_len, _device)
            return key_states.reshape(-1, head_dim), value_states.reshape(-1, head_dim)

        sorted_attn_score_indices = attn_score.sort(dim=-1, descending=True).indices

        total_budget = num_heads * self.base_capacity
        head_weights = self._compute_head_weights(attn_score)
        head_adaptive_capacity = self._allocate_head_budget_with_floor(
            head_weights, attn_score, total_budget, attn_score.size(-1)
        )
        head_adaptive_capacity = self._shared_batch_head_capacity(head_adaptive_capacity, total_budget)
        sorted_attn_score_indices = sorted_attn_score_indices.split(1, dim=1)

        self.cu_klen = 0

        heads_key_states, heads_value_states, k_lens, klen_sum, max_seqlen_k = self._gather_batchwise_flatten_kv(
            origin_heads_key_states,
            origin_heads_value_states,
            sorted_attn_score_indices,
            head_adaptive_capacity,
            head_dim,
        )

        self._init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k, _device)

        return heads_key_states, heads_value_states

def init_pyramidkv(self, num_hidden_layers):
    if not hasattr(self.config, 'window_size'):
        self.config.window_size = 32
    if not hasattr(self.config, 'max_capacity_prompt'):
        self.config.max_capacity_prompt = 2048
    if not hasattr(self.config, 'kernel_size'):
        self.config.kernel_size = 5
    if not hasattr(self.config, 'pooling'):
        self.config.pooling = 'avgpool'
    if not hasattr(self.config, 'merge'):
        self.config.merge = None

    should_reset_cluster = not hasattr(self, "kv_cluster") or getattr(self, "kv_seq_len", 0) == 0
    if should_reset_cluster:
        self.kv_cluster = PyramidKVCluster(
            num_hidden_layers = num_hidden_layers,
            layer_idx = self.layer_idx,
            window_size = self.config.window_size,
            max_capacity_prompt = self.config.max_capacity_prompt,
            kernel_size = self.config.kernel_size,
            pooling = self.config.pooling,
            merge = self.config.merge,

            )
 
def init_snapkv(self):
    if not hasattr(self.config, 'window_size'):
        self.config.window_size = 32
    if not hasattr(self.config, 'max_capacity_prompt'):
        self.config.max_capacity_prompt = 4096
    if not hasattr(self.config, 'kernel_size'):
        self.config.kernel_size = 5
    if not hasattr(self.config, 'pooling'):
        self.config.pooling = 'avgpool'
    if not hasattr(self.config, 'merge'):
        self.config.merge = None

    should_reset_cluster = not hasattr(self, "kv_cluster") or getattr(self, "kv_seq_len", 0) == 0
    if should_reset_cluster:
        self.kv_cluster = SnapKVCluster(
            window_size = self.config.window_size,
            max_capacity_prompt = self.config.max_capacity_prompt,
            kernel_size = self.config.kernel_size,
            pooling = self.config.pooling,
            merge = self.config.merge,
            layer_idx = self.layer_idx,
            num_hidden_layers = self.config.num_hidden_layers,

            )

def init_adaptivekv(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 7
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'floor_alpha'):
            self.config.floor_alpha = getattr(self.config, 'floor_ratio', 0.2)
        if not hasattr(self.config, 'skip'):
            self.config.skip = 0
        if not hasattr(self.config, 'normalize'):
            self.config.normalize = True
        if not hasattr(self.config, 'beta'):
            self.config.beta = 20
        if not hasattr(self.config, 'gqa_support'):
            self.config.gqa_support = False
        if not hasattr(self.config, 'inspect_attention_weight'):
            self.config.inspect_attention_weight = False
        if not hasattr(self.config, 'attention_log_file'):
            self.config.attention_log_file = None

    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = AdaptiveKVCluster(
            window_size = self.config.window_size,
            kernel_size = self.config.kernel_size,
            pooling = self.config.pooling,
            base_capacity = self.config.max_capacity_prompt,
            floor_alpha = self.config.floor_alpha,
            skip = self.config.skip,
            normalize = self.config.normalize,
            layer_idx = self.layer_idx,
            num_hidden_layers = self.config.num_hidden_layers,
            beta = self.config.beta,
            gqa_support = self.config.gqa_support,
            num_key_value_groups = getattr(self, 'num_key_value_groups', 1),
            gqa_func = getattr(self.config, 'gqa_func', None),
            inspect_attention_weight = self.config.inspect_attention_weight,
            attention_log_file = self.config.attention_log_file,
        )

def init_adakv(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'floor'):
            self.config.floor = getattr(self.config, 'floor_ratio', 0.2)
        if not hasattr(self.config, 'floor_ratio'):
            self.config.floor_ratio = self.config.floor
        if not hasattr(self.config, 'normalize'):
            self.config.normalize = True
    # max_capacity_prompt --> base_capacity
    # init only once
    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = AdaKVCluster( 
            num_hidden_layers = self.config.num_hidden_layers,
            layer_idx = self.layer_idx,
            window_size = self.config.window_size, 
            max_capacity_prompt = self.config.max_capacity_prompt, 
            kernel_size = self.config.kernel_size,
            pooling = self.config.pooling,
            floor = self.config.floor,
            normalize = self.config.normalize
            )

def init_championkv(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 7
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'allocate_strategy'):
            self.config.allocate_strategy = 'pyramid'
        if not hasattr(self.config, 'beta'):
            self.config.beta = 20
        if not hasattr(self.config, 'min_head_budget_ratio'):
            self.config.min_head_budget_ratio = 0.2
        if not hasattr(self.config, 'abnormal_head_max_tau'):
            self.config.abnormal_head_max_tau = getattr(self.config, 'abnormal_head_sum_tau', 0.0001)
        if not hasattr(self.config, 'gqa_support'):
            self.config.gqa_support = False
        if not hasattr(self.config, 'inspect_attention_weight'):
            self.config.inspect_attention_weight = False
        if not hasattr(self.config, 'attention_log_file'):
            self.config.attention_log_file = None

    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = ChampionKVCluster(
            window_size = self.config.window_size,
            kernel_size = self.config.kernel_size,
            pooling = self.config.pooling,
            base_capacity = self.config.max_capacity_prompt,
            allocate_strategy = self.config.allocate_strategy,
            min_head_budget_ratio = self.config.min_head_budget_ratio,
            abnormal_head_max_tau = self.config.abnormal_head_max_tau,
            layer_idx = self.layer_idx,
            num_hidden_layers = self.config.num_hidden_layers,
            beta = self.config.beta,
            gqa_support = self.config.gqa_support,
            num_key_value_groups = getattr(self, 'num_key_value_groups', 1),
            gqa_func = getattr(self.config, 'gqa_func', None),
            inspect_attention_weight = self.config.inspect_attention_weight,
            attention_log_file = self.config.attention_log_file,
        )


