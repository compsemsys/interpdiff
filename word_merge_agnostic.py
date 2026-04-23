"""
Model-agnostic grouping of token embeddings into word-level vectors (from aggregate_words_model_agnostic).

Key functions:
- ``group_tokens``: deterministic token grouping into candidate word units.
- ``canonical_word_and_span``: normalize grouped token text/span into one canonical word.
- ``merge_token_embeddings_to_words``: mean-pool token vectors to word vectors + metadata.
- ``token_starts_new_word``: tokenizer-agnostic boundary heuristic for word starts.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np


def flatten_token_meta(meta: list) -> list[dict[str, Any]]:
    if meta and isinstance(meta[0], list):
        return [item for chunk in meta for item in chunk]
    return meta


def token_starts_new_word(curr: dict, prev: dict | None) -> bool:
    if prev is None:
        return True
    if curr["doc_id"] != prev["doc_id"]:
        return True

    curr_start, _ = curr["span"]
    _, prev_end = prev["span"]

    curr_ts = curr.get("token_str", "")
    prev_ts = prev.get("token_str", "")
    prev_word = prev.get("word", "")

    if curr_ts.startswith("##"):
        return False
    if curr_ts.startswith("Ġ") or curr_ts.startswith("Ċ"):
        return True
    if curr_ts.startswith("▁"):
        return True
    if curr_start > prev_end:
        return True
    if prev_word and prev_word[-1].isspace():
        return True
    if prev_ts.startswith("Ċ"):
        return True
    return False


def group_tokens(meta: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    # Sort first so grouping is deterministic regardless of upstream chunk order.
    meta_sorted = sorted(meta, key=lambda m: (m["doc_id"], m["span"][0], m["span"][1]))
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    prev = None

    for m in meta_sorted:
        if token_starts_new_word(m, prev):
            if current:
                groups.append(current)
            current = [m]
        else:
            current.append(m)
        prev = m

    if current:
        groups.append(current)
    return groups


def canonical_word_and_span(group: list[dict[str, Any]]) -> tuple[str, Any]:
    pos_to_char: dict[int, str] = {}
    for m in group:
        start, end = m["span"]
        token_word = m.get("word", "")
        span_len = max(0, end - start)
        usable = token_word[:span_len]
        for i, ch in enumerate(usable):
            abs_pos = start + i
            if abs_pos not in pos_to_char:
                pos_to_char[abs_pos] = ch

    if not pos_to_char:
        return "", None

    # Keep alnum + a small punctuation subset to normalize tokenization artifacts.
    allowed = []
    for pos in sorted(pos_to_char):
        ch = pos_to_char[pos]
        if ch.isalnum() or ch in ["-", "'"]:
            allowed.append((pos, ch))

    if not allowed:
        return "", None

    word = "".join(ch for _, ch in allowed).lower()
    word = re.sub(r"'+$", "", word)
    if not re.search(r"[a-zA-Z0-9]", word):
        return "", None

    span = (allowed[0][0], allowed[-1][0] + 1)
    return word, span


def _doc_fields_from_token(m: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k in ("category", "title", "segment"):
        if k in m:
            out[k] = m[k]
    return out


def merge_token_embeddings_to_words(
    embeddings: np.ndarray,
    token_meta: list | list[list[dict[str, Any]]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """
    Return (word_embeddings [n_words, dim], word_meta list).
    """
    token_meta = flatten_token_meta(token_meta)
    meta_id_to_idx = {id(m): i for i, m in enumerate(token_meta)}
    groups = group_tokens(token_meta)

    word_embeddings: list[np.ndarray] = []
    word_meta: list[dict[str, Any]] = []

    for group in groups:
        word, span = canonical_word_and_span(group)
        if not word:
            continue

        # Mean pooling over subword rows gives one vector per canonical word.
        indices = [meta_id_to_idx[id(m)] for m in group]
        pooled = embeddings[indices].mean(axis=0)
        base = _doc_fields_from_token(group[0])
        word_embeddings.append(pooled)
        word_meta.append(
            {
                "doc_id": group[0]["doc_id"],
                "span": span,
                "word": word,
                "token_indices": indices,
                "subwords": [m.get("word", "") for m in group],
                "token_strs": [m.get("token_str", "") for m in group],
                **base,
            }
        )

    if not word_embeddings:
        return np.zeros((0, embeddings.shape[1]), dtype=np.float32), []
    return np.stack(word_embeddings, axis=0), word_meta
