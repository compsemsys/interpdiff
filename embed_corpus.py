"""
Compute contextual token embeddings for a list of labeled texts (same logic as mainlad.py).

Uses ``transformers.AutoModel`` (not CausalLM) to match ``last_hidden_state`` behavior.

Key functions:
- ``tokenize_and_chunk_document``: chunk one document into model-safe token blocks + meta.
- ``embed_labeled_texts``: load encoder model, batch chunks by length, and emit flat token embeddings.
"""

from __future__ import annotations

import importlib.util
import time
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from token_embed_utils import (
    chunk_token_ids_with_meta,
    model_max_length,
    pick_device,
)


def tokenize_and_chunk_document(
    item: tuple[int, dict[str, Any]],
    tokenizer,
    chunk_size: int,
    model_max_len: int,
) -> list[tuple[torch.Tensor, list[dict[str, Any]], tuple[int, int]]]:
    doc_idx, doc = item
    text = doc["text"]
    extra = {k: v for k, v in doc.items() if k != "text"}
    if "doc_id" in extra:
        extra = {k: v for k, v in extra.items() if k != "doc_id"}
    chunk_tensors = []
    # Keep token metadata aligned with each emitted chunk tensor for later merges.
    for chunk_idx, (chunk_tensor, chunk_meta) in enumerate(
        chunk_token_ids_with_meta(
            text,
            tokenizer,
            chunk_size,
            doc_id=doc_idx,
            extra_meta=extra,
        ),
        1,
    ):
        input_len = chunk_tensor.shape[0]
        if input_len > model_max_len:
            raise ValueError(
                f"Input chunk length {input_len} exceeds model max length {model_max_len}"
            )
        chunk_tensors.append((chunk_tensor, chunk_meta, (doc_idx, chunk_idx)))
    return chunk_tensors


def embed_labeled_texts(
    docs: list[dict[str, Any]],
    model_path: str,
    *,
    device: str | None = None,
    local_files_only: bool = True,
    batch_size: int = 12,
    chunk_size: int = 1024,
    trust_remote_code: bool = True,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """
    Each doc must include ``text`` and ``doc_id``; optional keys are merged into token meta.

    Returns (embeddings [num_tokens, hidden], flat token_meta list).
    """
    if device is None:
        device = pick_device()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    has_accelerate = importlib.util.find_spec("accelerate") is not None
    load_kwargs = {
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if device == "cuda":
        load_kwargs["torch_dtype"] = torch.float16
        if has_accelerate:
            load_kwargs["device_map"] = "cuda:0"
    try:
        model = AutoModel.from_pretrained(model_path, **load_kwargs)
    except TypeError:
        model = AutoModel.from_pretrained(
            model_path,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
    model.eval()
    if not (device == "cuda" and has_accelerate):
        model.to(device)

    mxl = model_max_length(tokenizer, model.config)
    chunk_size = min(chunk_size, mxl)

    embedding_chunks: list[np.ndarray] = []
    all_token_meta: list[dict[str, Any]] = []
    pending_by_len: dict[int, list[tuple[torch.Tensor, list[dict[str, Any]]]]] = {}
    total_batches = 0
    total_chunks_processed = 0
    total_tokens = 0
    embedding_start = time.time()

    def flush_one_length(chunk_len: int, *, flush_all: bool = False) -> None:
        # Batch by equal sequence length to avoid unnecessary attention padding.
        nonlocal total_batches, total_chunks_processed, total_tokens
        queue = pending_by_len.get(chunk_len, [])
        if not queue:
            return
        while len(queue) >= batch_size or (flush_all and queue):
            batch_items = queue[:batch_size]
            del queue[:batch_size]
            batch_tensors = [item[0] for item in batch_items]
            batch_metas = [item[1] for item in batch_items]
            batch_tensor = torch.stack(batch_tensors, dim=0).to(device)
            with torch.no_grad():
                outputs = model(input_ids=batch_tensor)
            emb = outputs.last_hidden_state.detach().float().cpu().numpy()
            for i, single_meta in enumerate(batch_metas):
                token_count = min(len(single_meta), emb.shape[1])
                if token_count <= 0:
                    continue
                embedding_chunks.append(emb[i, :token_count, :])
                all_token_meta.extend(single_meta[:token_count])
                total_tokens += token_count
            total_batches += 1
            total_chunks_processed += len(batch_tensors)
            print(
                f"  Embedded batch {total_batches}: +{len(batch_tensors)} chunks, "
                f"total {total_chunks_processed} chunks ({total_tokens} tokens)",
                flush=True,
            )

    for doc in docs:
        doc_idx = int(doc["doc_id"])
        for chunk_tensor, chunk_meta, _key in tokenize_and_chunk_document(
            (doc_idx, doc), tokenizer, chunk_size, mxl
        ):
            chunk_len = int(chunk_tensor.shape[0])
            pending_by_len.setdefault(chunk_len, []).append((chunk_tensor, chunk_meta))
            flush_one_length(chunk_len)

    for chunk_len in list(pending_by_len.keys()):
        flush_one_length(chunk_len, flush_all=True)

    embedding_end = time.time()
    print(
        f"  Token embedding wall time: {embedding_end - embedding_start:.2f}s, "
        f"{total_tokens} tokens"
    )

    if not embedding_chunks:
        return np.zeros((0, 0), dtype=np.float32), []

    arr = np.concatenate(embedding_chunks, axis=0)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return arr, all_token_meta
