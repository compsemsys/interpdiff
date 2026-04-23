"""
Model-agnostic grouping of token embeddings into document-level vectors.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from word_merge_agnostic import flatten_token_meta


def _doc_fields_from_token(m: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k in ("category", "title", "segment"):
        if k in m:
            out[k] = m[k]
    return out


def merge_token_embeddings_to_docs(
    embeddings: np.ndarray,
    token_meta: list | list[list[dict[str, Any]]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """
    Return (doc_embeddings [n_docs, dim], doc_meta list).
    """
    token_meta = flatten_token_meta(token_meta)
    if embeddings.shape[0] != len(token_meta):
        raise ValueError(
            f"embeddings/token_meta length mismatch: {embeddings.shape[0]} vs {len(token_meta)}"
        )
    if embeddings.shape[0] == 0:
        dim = embeddings.shape[1] if embeddings.ndim == 2 else 0
        return np.zeros((0, dim), dtype=np.float32), []

    # Group all token rows by doc_id, then mean-pool to one vector per document.
    doc_to_indices: dict[int, list[int]] = defaultdict(list)
    doc_to_first_meta: dict[int, dict[str, Any]] = {}

    for i, m in enumerate(token_meta):
        doc_id = int(m["doc_id"])
        doc_to_indices[doc_id].append(i)
        doc_to_first_meta.setdefault(doc_id, m)

    doc_embeddings: list[np.ndarray] = []
    doc_meta: list[dict[str, Any]] = []
    for doc_id in sorted(doc_to_indices):
        indices = doc_to_indices[doc_id]
        pooled = embeddings[indices].mean(axis=0)
        base = _doc_fields_from_token(doc_to_first_meta[doc_id])
        doc_embeddings.append(pooled)
        doc_meta.append(
            {
                "doc_id": doc_id,
                "token_count": len(indices),
                "token_indices": indices,
                **base,
            }
        )

    return np.stack(doc_embeddings, axis=0), doc_meta
