import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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

class CustomAutoModelForCausalLM(AutoModelForCausalLM, GenerationMixin):
    pass

def LoadMambaModel(model_dir):
    tokenizer = Tokenizer()
    model = CustomAutoModelForCausalLM.from_pretrained(model_dir)
    return { "model": model, "tokenizer": tokenizer, "type": "mamba" }
