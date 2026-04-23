"""Tests for word_merge_agnostic.merge_token_embeddings_to_words."""

from __future__ import annotations

import numpy as np

from word_merge_agnostic import merge_token_embeddings_to_words


def test_merge_propagates_category():
    emb = np.arange(12, dtype=np.float32).reshape(4, 3)
    meta = [
        {
            "doc_id": 1,
            "span": (0, 1),
            "token_str": "hel",
            "word": "hel",
            "token_id": 0,
            "word_idx": 0,
            "category": "CatA",
            "title": "t1",
            "segment": "excerpt",
        },
        {
            "doc_id": 1,
            "span": (1, 2),
            "token_str": "##lo",
            "word": "lo",
            "token_id": 1,
            "word_idx": 1,
            "category": "CatA",
            "title": "t1",
            "segment": "excerpt",
        },
        {
            "doc_id": 1,
            "span": (3, 4),
            "token_str": "x",
            "word": "x",
            "token_id": 2,
            "word_idx": 2,
            "category": "CatA",
            "title": "t1",
            "segment": "excerpt",
        },
        {
            "doc_id": 1,
            "span": (4, 5),
            "token_str": "y",
            "word": "y",
            "token_id": 3,
            "word_idx": 3,
            "category": "CatA",
            "title": "t1",
            "segment": "excerpt",
        },
    ]
    w, wm = merge_token_embeddings_to_words(emb, meta)
    assert w.shape[0] >= 1
    assert "category" in wm[0]
    assert wm[0]["category"] == "CatA"
    assert wm[0]["segment"] == "excerpt"
