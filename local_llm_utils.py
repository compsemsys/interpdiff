"""Local causal LM loading and text generation (local_files_only).

Key functions:
- ``load_causal_lm``: load tokenizer/model with local-first settings and device placement.
- ``build_explain_prompt``: format instruction templates, optionally via chat templates.
- ``generate_completion``: run deterministic generation and return decoded new tokens.
- ``sanitize_model_slug``: normalize model path names for artifact directories.
"""

from __future__ import annotations

import importlib.util
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from token_embed_utils import pick_device


def sanitize_model_slug(path: str) -> str:
    # Slug becomes a stable directory name under outputs/{excerpts,responses}/.
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", os.path.basename(path.rstrip("/\\")))


def load_causal_lm(
    model_path: str,
    device: str | None = None,
    *,
    local_only: bool = True,
    trust_remote_code: bool = True,
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    if device is None:
        device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        local_files_only=local_only,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device == "cuda" else torch.float32
    has_accelerate = importlib.util.find_spec("accelerate") is not None
    load_kwargs = {
        "local_files_only": local_only,
        "torch_dtype": dtype,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if device == "cuda" and has_accelerate:
        load_kwargs["device_map"] = "cuda:0"
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=local_only,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
    model.eval()
    if not (device == "cuda" and has_accelerate):
        model.to(device)
    return tokenizer, model


def build_explain_prompt(
    tokenizer: AutoTokenizer,
    instruction: str,
    *,
    excerpt: str = "",
    title: str = "",
) -> str:
    """Format ``instruction`` with ``{excerpt}`` and/or ``{title}`` (unused keys are ok)."""
    body = instruction.format(excerpt=excerpt, title=title)
    if getattr(tokenizer, "chat_template", None):
        # Respect model-specific chat formatting when available.
        messages = [{"role": "user", "content": body}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return body + "\n\n"


def generate_completion(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompt: str,
    device: str,
    max_new_tokens: int,
) -> str:
    enc = tokenizer(prompt, return_tensors="pt", truncation=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    input_len = enc["input_ids"].shape[1]
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    with torch.no_grad():
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
        )
    new_tokens = gen[0, input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
