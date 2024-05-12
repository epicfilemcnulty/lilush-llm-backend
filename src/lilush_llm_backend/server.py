import sys, os
import uuid
import time
import torch
import gc
import bottle
from bottle import Bottle, run, route, request, response
bottle.BaseRequest.MEMFILE_MAX = 1024 * 1024 * 10

from .loader import LoadExl2Model, LoadTfModel, LoadMambaModel
from .generation import Exl2Query, TfQuery, MambaQuery

models = {}
app = Bottle()

@app.route('/load', method='POST')
def load_model():
    data = request.json
    model_dir = data.get('model_dir')
    model_type = data.get('model_type')
    model_alias = data.get('model_alias')
    trust_remote_code = data.get('trust_remote_code', False)
    if not model_dir:
        response.status = 400
        return {"error": "model_dir is required"}
    if not model_type:
        response.status = 400
        return {"error": "model_type is required"}
    if not model_alias:
        response.status = 400
        return {"error": "model_alias is required"}
    context_length = data.get('context_length')
    lora_dir = data.get('lora_dir')
    if model_type == "exl2":
        models[model_alias] = LoadExl2Model(model_dir, context_length, lora_dir)
        return {"message": "model loaded"}
    if model_type == "tf":
        models[model_alias] = LoadTfModel(model_dir, context_length, lora_dir, trust_remote_code)
        return {"message": "model loaded"}
    if model_type == "mamba":
        models[model_alias] = LoadMambaModel(model_dir)
        return {"message": "model loaded"}

@app.route('/unload', method='DELETE')
def unload_model():
    data = request.json
    model_alias = data.get("model_alias")
    if model_alias is not None:
        if model_alias in models:
            del models[model_alias]
            gc.collect()
            with torch.no_grad():
                torch.cuda.empty_cache()
            return { "message": "model unloaded" }
    response.status = 404
    return { "error": "no such model" }

@app.route('/models', method='GET')
def loaded_models():
    return { "models": list(models.keys()) }

@app.route('/complete', method='POST')
def complete():
    data = request.json
    query = data.get('query')
    conversation_uuid = data.get('uuid', str(uuid.uuid4()))
    model_alias = data.get('model')
    if model_alias not in models:
        response.status = 404
        return { "error": "model not found"}

    model_type = models[model_alias]["type"]

    sampler = {
        "temperature": data.get("temperature", 0.5),
        "top_k": data.get("top_k", 40),
        "top_p": data.get("top_p", 0.75),
        "min_p": data.get("min_p", 0.0),
        "repetition_penalty": data.get("repetition_penalty", 1.05),
        "max_new_tokens": data.get("max_new_tokens", 512),
        "add_bos": data.get('add_bos', True),
        "add_eos": data.get('add_eos', False),
        "encode_special_tokens": data.get('encode_special_tokens', False),
        "stop_conditions": data.get('stop_conditions', []),
        "hide_special_tokens": data.get('hide_special_tokens', False),
    }

    start_time = time.time_ns()

    stop_reason = None
    if model_type == "exl2":
        new_text, prompt_tokens, generated_tokens, stop_reason = Exl2Query(query, sampler, models[model_alias]["tokenizer"], models[model_alias]["generator"], models[model_alias]["lora"])
    if model_type == "tf":
        new_text, prompt_tokens, generated_tokens = TfQuery(query, sampler, models[model_alias]["model"], models[model_alias]["tokenizer"])
    if model_type == "mamba":
        new_text, prompt_tokens, generated_tokens = MambaQuery(query, sampler, models[model_alias]["model"], models[model_alias]["tokenizer"])

    end_time = time.time_ns()
    secs = (end_time - start_time) / 1e9

    return {
        "uuid": conversation_uuid,
        "text": new_text,
        "tokens": generated_tokens,
        "rate": generated_tokens / secs,
        "model": model_alias,
        "backend" : model_type,
        "stop": stop_reason,
        "ctx" : prompt_tokens + generated_tokens
    }

def Serve(ip='127.0.0.1', port=8013): 
    run(app, host=ip, port=port)
