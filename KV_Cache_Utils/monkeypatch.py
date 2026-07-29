SUPPORTED_METHODS = {"fullkv", "pyramidkv", "snapkv", "adaptivekv", "adakv", "championkv"}


def _unsupported_method(model_type, method):
    supported = ", ".join(sorted(SUPPORTED_METHODS))
    raise ValueError(
        f"Unsupported {model_type} kv method in this build: {method}. "
        f"Supported methods: {supported}"
    )


def replace_llama(method, model_name=None):
    import transformers
    from KV_Cache_Utils.llama_model import (
        adaptive_LlamaModel_forward,
        llama_attn_forward_PyramidKV,
        llama_flash_attn2_forward_PyramidKV,
        llama_sdpa_attn_forward_PyramidKV,
        llama_attn_forward_SnapKV,
        llama_flash_attn2_forward_SnapKV,
        llama_sdpa_attn_forward_SnapKV,
        llama_flash_attn2_forward_AdaKV,
        prepare_inputs_for_generation_llama_new,
    )

    method = method.lower()

    if method == "pyramidkv":
        print("Using PyramidKV!")
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_PyramidKV
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_PyramidKV
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_PyramidKV
    elif method == "snapkv":
        print("Using SnapKV!")
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_SnapKV
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_SnapKV
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_SnapKV
    elif method in {"adaptivekv", "adakv", "championkv"}:
        print(f"Using {method.upper()}!")
        transformers.models.llama.modeling_llama.LlamaModel.forward = adaptive_LlamaModel_forward
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_flash_attn2_forward_AdaKV
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_AdaKV
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_flash_attn2_forward_AdaKV
    elif method != "fullkv":
        _unsupported_method("llama", method)

    if method != "fullkv":
        transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = (
            prepare_inputs_for_generation_llama_new
        )


def replace_mistral(method):
    import transformers
    from KV_Cache_Utils.mistral_model import (
        adaptive_MistralModel_forward,
        mistral_attn_forward_PyramidKV,
        mistral_flash_attn2_forward_PyramidKV,
        mistral_sdpa_attn_forward_PyramidKV,
        mistral_attn_forward_SnapKV,
        mistral_flash_attn2_forward_SnapKV,
        mistral_sdpa_attn_forward_SnapKV,
        mistral_flash_attn2_forward_AdaKV,
        prepare_inputs_for_generation_mistral_new,
    )

    method = method.lower()

    if method == "pyramidkv":
        print("Using PyramidKV!")
        transformers.models.mistral.modeling_mistral.MistralAttention.forward = mistral_attn_forward_PyramidKV
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_PyramidKV
        transformers.models.mistral.modeling_mistral.MistralSdpaAttention.forward = mistral_sdpa_attn_forward_PyramidKV
    elif method == "snapkv":
        print("Using SnapKV!")
        transformers.models.mistral.modeling_mistral.MistralAttention.forward = mistral_attn_forward_SnapKV
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_SnapKV
        transformers.models.mistral.modeling_mistral.MistralSdpaAttention.forward = mistral_sdpa_attn_forward_SnapKV
    elif method in {"adaptivekv", "adakv", "championkv"}:
        print(f"Using {method.upper()}!")
        transformers.models.mistral.modeling_mistral.MistralModel.forward = adaptive_MistralModel_forward
        transformers.models.mistral.modeling_mistral.MistralAttention.forward = mistral_flash_attn2_forward_AdaKV
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_AdaKV
        transformers.models.mistral.modeling_mistral.MistralSdpaAttention.forward = mistral_flash_attn2_forward_AdaKV
    elif method != "fullkv":
        _unsupported_method("mistral", method)

    if method != "fullkv":
        transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = (
            prepare_inputs_for_generation_mistral_new
        )
