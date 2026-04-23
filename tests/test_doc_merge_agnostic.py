"""Tests for doc_merge_agnostic.merge_token_embeddings_to_docs."""

from __future__ import annotations

import numpy as np

from doc_merge_agnostic import merge_token_embeddings_to_docs


def test_merge_docs_means_by_doc_id_and_propagates_meta():
    emb = np.array(
        [
            [1.0, 2.0, 3.0],
            [3.0, 4.0, 5.0],
            [10.0, 20.0, 30.0],
        ],
        dtype=np.float32,
    )
    meta = [
        {
            "doc_id": 2,
            "span": (0, 1),
            "category": "CatB",
            "title": "Doc 2",
            "segment": "response",
        },
        {
            "doc_id": 2,
            "span": (1, 2),
            "category": "CatB",
            "title": "Doc 2",
            "segment": "response",
        },
        {
            "doc_id": 5,
            "span": (0, 1),
            "category": "CatC",
            "title": "Doc 5",
            "segment": "response",
        },
    ]

    out, out_meta = merge_token_embeddings_to_docs(emb, meta)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out[0], np.array([2.0, 3.0, 4.0], dtype=np.float32))
    np.testing.assert_allclose(out[1], np.array([10.0, 20.0, 30.0], dtype=np.float32))
    assert out_meta[0]["doc_id"] == 2
    assert out_meta[0]["token_count"] == 2
    assert out_meta[0]["category"] == "CatB"
    assert out_meta[1]["doc_id"] == 5
    assert out_meta[1]["token_count"] == 1


def test_merge_docs_supports_nested_token_meta():
    emb = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    nested_meta = [
        [{"doc_id": 11, "span": (0, 1)}],
        [{"doc_id": 11, "span": (1, 2)}],
    ]
    out, out_meta = merge_token_embeddings_to_docs(emb, nested_meta)
    assert out.shape == (1, 2)
    np.testing.assert_allclose(out[0], np.array([1.0, 2.0], dtype=np.float32))
    assert out_meta[0]["doc_id"] == 11
    assert out_meta[0]["token_count"] == 2
