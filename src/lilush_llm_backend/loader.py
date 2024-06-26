import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, MambaForCausalLM
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from bltzr import Tokenizer
from peft import PeftModel

from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache_Q4,
    ExLlamaV2Tokenizer,
    ExLlamaV2Lora,
)
from exllamav2.generator import (
    ExLlamaV2StreamingGenerator,
    ExLlamaV2DynamicGenerator,
    ExLlamaV2Sampler
)

def LoadExl2Model(model_dir, context_length=None, cache_size=None, dynamic=False, lora_dir=None):
    # Initialize model and cache
    config = ExLlamaV2Config(model_dir)
    model = ExLlamaV2(config)
    if context_length is not None and context_length != 0:
        config.max_seq_len = context_length

    if cache_size is None:
        c_size = config.max_seq_len
    else:
        c_size = cache_size
    cache = ExLlamaV2Cache_Q4(model, max_seq_len = c_size, lazy = True)
    print("Loading model: " + model_dir)
    model.load_autosplit(cache)
    tokenizer = ExLlamaV2Tokenizer(config)
    lora = None
    if lora_dir is not None:
        lora = ExLlamaV2Lora.from_directory(model, lora_dir)
    # Initialize generator
    if dynamic:
        generator = ExLlamaV2DynamicGenerator(model, cache, tokenizer)
    else:
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

def LoadMambaModel(model_dir):
    tokenizer = Tokenizer()
    model = MambaLMHeadModel.from_pretrained(model_dir, device="cuda", dtype=torch.bfloat16)
    return { "model": model, "tokenizer": tokenizer, "type": "mamba" }
