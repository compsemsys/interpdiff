"""Local causal LM loading and text generation (local_files_only).

Key functions:
- ``load_causal_lm``: load tokenizer/model with local-first settings and device placement.
- ``build_explain_prompt``: format instruction templates, optionally via chat templates.
- ``generate_completion``: run deterministic generation and return decoded new tokens.
- ``response_word_count`` / ``truncate_completion_to_max_generated_words``: whitespace word
  counting on the **generated completion** (hyphenated spellings stay one word) and hard caps.
- ``sanitize_model_slug``: normalize model path names for artifact directories.
"""

from __future__ import annotations

import importlib.util
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

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
    word_count: int | str | None = None,
) -> str:
    """Format ``instruction`` with ``{excerpt}``, ``{title}``, and/or ``{word_count}``."""
    body = instruction.format(excerpt=excerpt, title=title, word_count=word_count)
    if getattr(tokenizer, "chat_template", None):
        # Respect model-specific chat formatting when available.
        messages = [{"role": "user", "content": body}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return body + "\n\n"


def response_word_count(text: str) -> int:
    """Count whitespace-delimited words in generated text (after strip).

    ``str.split()`` is used so hyphenated forms like ``well-known`` or ``co-op``
    count as a single word; newlines and tabs are word separators.
    """
    s = text.strip()
    return len(s.split()) if s else 0


def truncate_completion_to_max_generated_words(text: str, max_generated_words: int) -> str:
    """Return completion text with at most ``max_generated_words`` whitespace-delimited words."""
    if max_generated_words < 1:
        return ""
    s = text.strip()
    if not s:
        return s
    parts = s.split()
    if len(parts) <= max_generated_words:
        return s
    return " ".join(parts[:max_generated_words])


class _MaxGeneratedWordsStoppingCriteria(StoppingCriteria):
    """Stop when decoded completion (prompt excluded) has ``>= max_generated_words`` words."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        prompt_token_len: int,
        max_generated_words: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.prompt_token_len = int(prompt_token_len)
        self.max_generated_words = int(max_generated_words)

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs: object,
    ) -> torch.BoolTensor:
        out = []
        for i in range(input_ids.shape[0]):
            new_ids = input_ids[i, self.prompt_token_len :]
            if new_ids.numel() == 0:
                out.append(False)
                continue
            piece = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            out.append(response_word_count(piece) >= self.max_generated_words)
        return torch.tensor(out, device=input_ids.device, dtype=torch.bool)


def generate_completion(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompt: str,
    device: str,
    max_new_tokens: int,
    *,
    max_generated_words: int | None = None,
) -> str:
    """Greedy-decode a completion (new tokens only); optional ``max_generated_words`` caps decoded reply length."""
    enc = tokenizer(prompt, return_tensors="pt", truncation=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    input_len = enc["input_ids"].shape[1]
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    stopping_criteria = None
    if max_generated_words is not None:
        stopping_criteria = StoppingCriteriaList(
            [
                _MaxGeneratedWordsStoppingCriteria(
                    tokenizer, input_len, max_generated_words
                )
            ]
        )
    gen_kw: dict[str, object] = {
        **enc,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": pad_id,
    }
    if stopping_criteria is not None:
        gen_kw["stopping_criteria"] = stopping_criteria
    with torch.no_grad():
        gen = model.generate(**gen_kw)
    new_tokens = gen[0, input_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if max_generated_words is not None:
        text = truncate_completion_to_max_generated_words(text, max_generated_words)
    return text
