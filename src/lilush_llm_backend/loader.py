import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, MambaForCausalLM
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from bltzr import Tokenizer
from peft import PeftModel
from .mixin import GenerationMixin

from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache_Q4,
    ExLlamaV2Tokenizer,
    ExLlamaV2Lora,
)
from exllamav2.generator import (
    ExLlamaV2StreamingGenerator,
    ExLlamaV2Sampler
)

def LoadExl2Model(model_dir, context_length=None, lora_dir=None):
    # Initialize model and cache
    config = ExLlamaV2Config()
    config.model_dir = model_dir
    config.prepare()
    if context_length is not None and context_length != 0:
        config.max_seq_len = context_length

    model = ExLlamaV2(config)
    print("Loading model: " + model_dir)
    model.load()
    tokenizer = ExLlamaV2Tokenizer(config)
    cache = ExLlamaV2Cache_Q4(model, lazy = not model.loaded)
    lora = None
    if lora_dir is not None:
        lora = ExLlamaV2Lora.from_directory(model, lora_dir)
    # Initialize generator
    generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)
    # Make sure CUDA is initialized so we can measure performance
    generator.warmup()
    return { "model": model, "generator": generator, "tokenizer": tokenizer, "cache": cache, "lora": lora, "type": "exl2" }

def LoadTfModel(model_dir, context_length=None, lora_dir=None, trust_remote_code=False):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map='auto', quantization_config=nf4_config, trust_remote_code=trust_remote_code, attn_implementation="flash_attention_2")
    print(model.generation_config)
    model.eval()
    if lora_dir is not None:
        model = PeftModel.from_pretrained(model, lora_dir)

    return { "model": model, "tokenizer": tokenizer, "type": "tf" }

class CustomModelForCausalLM(MambaForCausalLM, GenerationMixin):
    def _validate_model_kwargs(self, model_kwargs):
        # Skip validation for unsupported arguments
        supported_kwargs = [
            "max_length",
            "min_length",
            "do_sample",
            "early_stopping",
            "num_beams",
            "temperature",
            "top_k",
            "top_p",
            "repetition_penalty",
            "bad_words_ids",
            "bos_token_id",
            "pad_token_id",
            "eos_token_id",
            "length_penalty",
            "no_repeat_ngram_size",
            "encoder_no_repeat_ngram_size",
            "num_return_sequences",
            "max_time",
            "max_new_tokens",
            "decoder_start_token_id",
            "use_cache",
            "num_beam_groups",
            "diversity_penalty",
            "prefix_allowed_tokens_fn",
            "logits_processor",
            "renormalize_logits",
            "stopping_criteria",
            "constraints",
            "output_attentions",
            "output_hidden_states",
            "output_scores",
            "return_dict_in_generate",
            "forced_bos_token_id",
            "forced_eos_token_id",
            "remove_invalid_values",
            "synced_gpus",
            "exponential_decay_length_penalty",
            "suppress_tokens",
            "begin_suppress_tokens",
            "forced_decoder_ids",
        ]
        for key in list(model_kwargs):   # Making a copy of model_kwargs with `list` so we can remove elements from the original
            if key not in supported_kwargs:
                model_kwargs.pop(key)
        super()._validate_model_kwargs(model_kwargs)

def LoadMambaHfModel(model_dir):
    tokenizer = Tokenizer()
    model = CustomModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16).to('cuda')
    return { "model": model, "tokenizer": tokenizer, "type": "mamba" }

def LoadMambaModel(model_dir):
    tokenizer = Tokenizer()
    model = MambaLMHeadModel.from_pretrained(model_dir, device="cuda", dtype=torch.bfloat16)
    return { "model": model, "tokenizer": tokenizer, "type": "mamba" }
