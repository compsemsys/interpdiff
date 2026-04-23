"""
Shared tokenization + chunking helpers (factored from mainlad.py).

Each token meta dict includes doc fields plus optional keys from ``extra_meta``
(e.g. category, title, segment).

Key functions:
- ``chunk_token_ids_with_meta``: tokenize text and emit chunk tensors plus per-token metadata.
- ``model_max_length``: resolve safe model max sequence length from config/tokenizer.
- ``wrap_token_ids_with_special_tokens``: add model-specific special tokens to a chunk.
- ``pick_device``: choose CUDA when usable, else CPU.
"""

from __future__ import annotations

from typing import Any, Iterator

import torch
from transformers import PreTrainedTokenizerBase


def pick_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    try:
        # Probe with a tiny CUDA op so we gracefully fall back on partially broken setups.
        x = torch.zeros(1, device="cuda", dtype=torch.float32)
        x = x + 1
        torch.cuda.synchronize()
        return "cuda"
    except Exception:
        return "cpu"


def special_prefix_suffix_ids(tokenizer: PreTrainedTokenizerBase) -> tuple[list[int], list[int]]:
    for probe in ("a", "the", " hello", "x", "test"):
        inner = tokenizer(probe, add_special_tokens=False)["input_ids"]
        outer = tokenizer(probe, add_special_tokens=True)["input_ids"]
        if not inner:
            continue
        n = len(inner)
        for i in range(len(outer) - n + 1):
            if outer[i : i + n] == inner:
                return outer[:i], outer[i + n :]

    bos_id = getattr(tokenizer, "bos_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    add_bos = bool(getattr(tokenizer, "add_bos_token", True))
    add_eos = bool(getattr(tokenizer, "add_eos_token", False))
    pre = [bos_id] if (add_bos and bos_id is not None) else []
    suf = [eos_id] if (add_eos and eos_id is not None) else []
    return pre, suf


def num_special_tokens(tokenizer: PreTrainedTokenizerBase) -> int:
    if hasattr(tokenizer, "build_inputs_with_special_tokens"):
        return len(tokenizer.build_inputs_with_special_tokens([]))
    pre, suf = special_prefix_suffix_ids(tokenizer)
    return len(pre) + len(suf)


def wrap_token_ids_with_special_tokens(
    tokenizer: PreTrainedTokenizerBase, token_ids: list[int]
) -> list[int]:
    if hasattr(tokenizer, "build_inputs_with_special_tokens"):
        return tokenizer.build_inputs_with_special_tokens(token_ids)
    pre, suf = special_prefix_suffix_ids(tokenizer)
    return pre + list(token_ids) + suf


def model_max_length(tokenizer: PreTrainedTokenizerBase, config: Any) -> int:
    try:
        m = getattr(config, "max_position_embeddings", None)
        if m is None:
            m = getattr(tokenizer, "model_max_length", 512)
        return int(m)
    except Exception:
        return 512


def chunk_token_ids_with_meta(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    chunk_size: int,
    doc_id: int | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> Iterator[tuple[torch.Tensor, list[dict[str, Any]]]]:
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    token_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    num_special = num_special_tokens(tokenizer)
    # Reserve room for BOS/EOS/etc. so wrapped chunks stay model-safe.
    max_chunk = chunk_size - num_special
    if max_chunk < 1:
        max_chunk = 1

    extra = dict(extra_meta) if extra_meta else {}

    for i in range(0, len(token_ids), max_chunk):
        chunk_ids = token_ids[i : i + max_chunk]
        chunk_offsets = offsets[i : i + max_chunk]
        chunk_tokens = tokens[i : i + max_chunk]
        chunk_ids_with_special = wrap_token_ids_with_special_tokens(tokenizer, chunk_ids)
        chunk_tensor = torch.tensor(chunk_ids_with_special, dtype=torch.long)
        chunk_meta = []
        for j, (tok, off) in enumerate(zip(chunk_tokens, chunk_offsets)):
            word = text[off[0] : off[1]] if off[0] < off[1] else ""
            row = {
                "token_id": chunk_ids[j],
                "token_str": tok,
                "word": word,
                "span": off,
                "word_idx": j + i,
                "doc_id": doc_id,
                **extra,
            }
            chunk_meta.append(row)
        yield chunk_tensor, chunk_meta
