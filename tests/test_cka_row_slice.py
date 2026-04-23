"""Row slicing helpers for CKA (metadata filters on aligned word rows)."""

from __future__ import annotations

import numpy as np
import pytest

from cka_word_embeddings import (
    apply_row_indices,
    pairwise_linear_cka_from_paths,
    row_indices_meta_match,
    validate_aligned_meta,
)


def test_row_indices_category_and_doc_ids():
    meta = [
        {"doc_id": 1, "span": (0, 1), "word": "a", "category": "CatA"},
        {"doc_id": 1, "span": (1, 2), "word": "b", "category": "CatA"},
        {"doc_id": 2, "span": (0, 1), "word": "c", "category": "CatB"},
    ]
    idx = row_indices_meta_match(meta, filters=[("category", "CatA")], doc_ids=None)
    assert list(idx) == [0, 1]
    idx2 = row_indices_meta_match(meta, filters=[("category", "CatB")], doc_ids={1})
    assert list(idx2) == []


def test_apply_row_indices_keeps_alignment():
    X = np.arange(12, dtype=np.float32).reshape(4, 3)
    Y = np.arange(12, dtype=np.float32).reshape(4, 3) + 0.5
    meta_a = [
        {"doc_id": 0, "span": (i, i + 1), "word": f"w{i}"} for i in range(4)
    ]
    meta_b = [dict(m) for m in meta_a]
    idx = np.array([1, 3], dtype=np.intp)
    Xs, Ys, ma, mb = apply_row_indices(X, Y, meta_a, meta_b, idx)
    assert Xs.shape == (2, 3)
    validate_aligned_meta(ma, mb)
    assert [m["word"] for m in ma] == ["w1", "w3"]


def test_pairwise_linear_cka_from_paths_with_filter(tmp_path):
    meta = []
    for doc_id, cat in [(1, "A"), (1, "A"), (2, "B")]:
        meta.append({"doc_id": doc_id, "span": (0, 1), "word": "x", "category": cat})
    X = np.random.default_rng(0).standard_normal((3, 4)).astype(np.float32)
    Y = X + 0.01 * np.random.default_rng(1).standard_normal((3, 4)).astype(np.float32)
    pa = tmp_path / "a.npy"
    pb = tmp_path / "b.npy"
    np.save(str(pa), {"embeddings": X, "meta": meta}, allow_pickle=True)
    np.save(str(pb), {"embeddings": Y, "meta": list(meta)}, allow_pickle=True)
    out = pairwise_linear_cka_from_paths(
        str(pa),
        str(pb),
        chunk_rows=256,
        filters=[("category", "A")],
        doc_ids=None,
    )
    assert out["n_rows_total"] == 3
    assert out["n_rows"] == 2
    assert "row_slice" in out
    assert not np.isnan(out["linear_cka"])


def test_validate_aligned_meta_accepts_document_rows():
    meta_a = [
        {"doc_id": 1, "category": "A", "title": "Doc 1", "segment": "excerpt"},
        {"doc_id": 2, "category": "B", "title": "Doc 2", "segment": "excerpt"},
    ]
    meta_b = [dict(m) for m in meta_a]
    validate_aligned_meta(meta_a, meta_b)


def test_pairwise_linear_cka_from_paths_with_document_meta(tmp_path):
    meta = [
        {"doc_id": 10, "category": "A", "title": "A1", "segment": "excerpt"},
        {"doc_id": 11, "category": "B", "title": "B1", "segment": "excerpt"},
    ]
    X = np.random.default_rng(5).standard_normal((2, 6)).astype(np.float32)
    Y = X + 0.01 * np.random.default_rng(6).standard_normal((2, 6)).astype(np.float32)
    pa = tmp_path / "a_doc.npy"
    pb = tmp_path / "b_doc.npy"
    np.save(str(pa), {"embeddings": X, "meta": meta}, allow_pickle=True)
    np.save(str(pb), {"embeddings": Y, "meta": list(meta)}, allow_pickle=True)
    out = pairwise_linear_cka_from_paths(
        str(pa),
        str(pb),
        chunk_rows=256,
        filters=[("category", "A")],
        doc_ids=None,
    )
    assert out["n_rows_total"] == 2
    assert out["n_rows"] == 1
    assert "row_slice" in out
    assert np.isnan(out["linear_cka"])
