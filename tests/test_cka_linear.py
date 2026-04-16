"""
Pytest checks for linear CKA (Kornblith-style) used in `cka_word_embeddings.linear_cka`.

Run from repo root: pytest tests/test_cka_linear.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from cka_word_embeddings import linear_cka as cka


def _get_one():
    X = np.cos(0.1 * np.pi * np.arange(10)).reshape((-1, 1))
    Y = np.cos(2 + 0.07 * np.pi * np.arange(10)).reshape((-1, 1))
    return X, Y


def _get_multi():
    X = np.cos(
        0.1
        * np.pi
        * np.arange(10).reshape((-1, 1))
        * np.linspace(0.5, 1.5, num=3).reshape((1, -1))
    )
    Y = np.cos(
        0.5
        + 0.07
        * np.pi
        * np.arange(10).reshape((-1, 1))
        * np.linspace(0.7, 1.3, num=4).reshape((1, -1))
    )
    return X, Y


def test_identity_lenient():
    """A matrix is perfectly aligned with itself (linear CKA = 1)."""
    X, _ = _get_multi()
    np.testing.assert_allclose(cka(X, X), 1.0)


def test_column_swaps():
    """Same Gram matrix under column permutation: CKA(X_a, X_b) == 1."""
    X, _ = _get_multi()
    c = cka(X[:, [0, 1]], X[:, [1, 0]])
    np.testing.assert_allclose(c, 1.0)


def test_centering():
    """Column-wise constant offset is removed by centering: CKA stays 1."""
    X, _ = _get_multi()
    Xp = X.copy()
    Xp[:, 1] += 1.0

    c = cka(X, Xp)
    np.testing.assert_allclose(c, 1.0)


def test_pure():
    """linear_cka must not mutate input arrays."""
    X, _ = _get_multi()
    Xp = X.copy()
    Xp[:, 1] += 1.0

    Xp_original = Xp.copy()
    _ = cka(X, Xp)
    np.testing.assert_allclose(Xp_original[:, 1], Xp[:, 1])


def test_corr():
    """For two 1D columns, linear CKA equals squared Pearson correlation."""
    X, Y = _get_one()
    c1 = cka(X, Y)
    c2 = np.corrcoef(X.squeeze(), Y.squeeze())[0, 1] ** 2
    np.testing.assert_allclose(c1, c2)


def test_isoscaling():
    """CKA is invariant to per-matrix scalar scaling (including sign flip)."""
    X, Y = _get_multi()
    c1 = cka(X, Y)
    c2 = cka(2.0 * X, -1 * Y)
    np.testing.assert_allclose(c1, c2)


def test_rotation():
    """CKA is invariant to orthogonal rotations of columns (same n, same Y)."""
    X, Y = _get_multi()
    X0 = X[:, :2]
    X0p = X0 @ np.array([[1, -1], [1, 1]], dtype=float) / np.sqrt(2)
    c1 = cka(X0, Y)
    c2 = cka(X0p, Y)
    np.testing.assert_allclose(c1, c2)


def test_no_iso():
    """CKA changes under non-isotropic column scaling (not orthogonal)."""
    X, Y = _get_multi()
    X0 = X[:, :2]
    X0p = X0 @ np.array([[1, 1], [10, 1]], dtype=float)
    c1 = cka(X0, Y)
    c2 = cka(X0p, Y)
    assert abs(c1 - c2) > 0.001
