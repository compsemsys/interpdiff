"""
Linear CKA between two aligned word-embedding matrices.

Uses the linear (inner-product) CKA from Kornblith et al.,
"Similarity of Neural Network Representations Revisited" (ICML 2019).
Rows must correspond to the same word instances (same meta order).

Example:
  python cka_word_embeddings.py \\
    --a F:/quantas/outputs/.../word_embeddings_merged_agnostic.npy \\
    --b F:/quantas/outputs/.../word_embeddings_merged_agnostic.npy
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from typing import Any

import numpy as np


def _limit_threads(n: int) -> None:
    """Cap BLAS/OpenMP threads so one run does not pin every CPU core and thrash RAM."""
    s = str(max(1, int(n)))
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(key, s)


def load_embeddings_and_meta(
    path: str, dtype=np.float32
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    data = np.load(path, allow_pickle=True).item()
    emb = np.asarray(data["embeddings"], dtype=dtype)
    meta = data["meta"]
    return emb, meta


def meta_key(m: dict[str, Any]) -> tuple:
    return (m["doc_id"], tuple(m["span"]), m["word"])


def validate_aligned_meta(meta_a: list, meta_b: list) -> None:
    if len(meta_a) != len(meta_b):
        raise ValueError(
            f"Row count mismatch: {len(meta_a)} vs {len(meta_b)}"
        )
    for i, (a, b) in enumerate(zip(meta_a, meta_b)):
        if meta_key(a) != meta_key(b):
            raise ValueError(
                f"Metadata mismatch at row {i}: {meta_key(a)!r} vs {meta_key(b)!r}"
            )


def center_columns(X: np.ndarray) -> np.ndarray:
    return X - X.mean(axis=0, keepdims=True)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear CKA between two matrices with the same number of rows (samples).
    X: (n, d1), Y: (n, d2). Column-centered, then:
      CKA = ||Xc.T @ Yc||_F^2 / (||Xc.T @ Xc||_F * ||Yc.T @ Yc||_F)
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"Same n required: {X.shape[0]} vs {Y.shape[0]}")
    n = X.shape[0]
    if n < 2:
        return float("nan")

    Xc = center_columns(X)
    Yc = center_columns(Y)

    cross = Xc.T @ Yc
    num = float(np.linalg.norm(cross, ord="fro") ** 2)

    xx = Xc.T @ Xc
    yy = Yc.T @ Yc
    denom = float(np.linalg.norm(xx, ord="fro") * np.linalg.norm(yy, ord="fro"))
    if denom <= 0:
        return float("nan")
    return num / denom


def linear_cka_chunked(
    X: np.ndarray, Y: np.ndarray, chunk_rows: int
) -> float:
    """
    Same linear CKA as `linear_cka`, but accumulates statistics in row blocks.

    Uses:
      Xc.T @ Yc = X.T @ Y - n * mx @ my.T
      Xc.T @ Xc = X.T @ X - n * mx @ mx.T
      Yc.T @ Yc = Y.T @ Y - n * my @ my.T

    Peak memory stays near one chunk of rows (plus d1*d2 accumulators) instead of
    relying on one huge GEMM over all rows.
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"Same n required: {X.shape[0]} vs {Y.shape[0]}")
    n = X.shape[0]
    if n < 2:
        return float("nan")

    d1, d2 = X.shape[1], Y.shape[1]
    chunk_rows = max(1, int(chunk_rows))

    # Accumulate in float64 for stability; inputs may be float32
    sum_x = np.zeros(d1, dtype=np.float64)
    sum_y = np.zeros(d2, dtype=np.float64)
    xt_y = np.zeros((d1, d2), dtype=np.float64)
    xt_x = np.zeros((d1, d1), dtype=np.float64)
    yt_y = np.zeros((d2, d2), dtype=np.float64)

    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        xb = np.asarray(X[start:end], dtype=np.float64)
        yb = np.asarray(Y[start:end], dtype=np.float64)
        sum_x += xb.sum(axis=0)
        sum_y += yb.sum(axis=0)
        xt_y += xb.T @ yb
        xt_x += xb.T @ xb
        yt_y += yb.T @ yb

    mx = sum_x / n
    my = sum_y / n
    xc_y = xt_y - np.outer(mx, my) * n
    xc_x = xt_x - np.outer(mx, mx) * n
    yc_y = yt_y - np.outer(my, my) * n

    num = float(np.linalg.norm(xc_y, ord="fro") ** 2)
    denom = float(np.linalg.norm(xc_x, ord="fro") * np.linalg.norm(yc_y, ord="fro"))
    if denom <= 0:
        return float("nan")
    return num / denom


def subsample_rows(
    X: np.ndarray, Y: np.ndarray, meta: list, rng: np.random.Generator, k: int
) -> tuple[np.ndarray, np.ndarray, list]:
    n = X.shape[0]
    if k >= n:
        return X, Y, meta
    idx = rng.choice(n, size=k, replace=False)
    return X[idx], Y[idx], [meta[i] for i in idx]


def per_doc_indices(meta: list) -> dict[Any, list[int]]:
    by_doc: dict[Any, list[int]] = defaultdict(list)
    for i, m in enumerate(meta):
        by_doc[m["doc_id"]].append(i)
    return dict(by_doc)


def run_bootstrap(
    X: np.ndarray,
    Y: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[float, float, float, float]:
    n = X.shape[0]
    scores = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        scores[b] = linear_cka(X[idx], Y[idx])
    q = np.percentile(scores, [2.5, 97.5])
    return float(np.mean(scores)), float(np.std(scores)), float(q[0]), float(q[1])


def main() -> None:
    p = argparse.ArgumentParser(
        description="Linear CKA between two aligned word embedding .npy files"
    )
    p.add_argument("--a", required=True, help="First word_embeddings_merged_agnostic.npy")
    p.add_argument("--b", required=True, help="Second word_embeddings_merged_agnostic.npy")
    p.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Random subset size (for quick runs)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--per_doc",
        action="store_true",
        help="Also compute CKA per doc_id",
    )
    p.add_argument(
        "--min_doc_rows",
        type=int,
        default=32,
        help="Skip docs with fewer rows for per-doc CKA",
    )
    p.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="Number of bootstrap resamples (0 = off)",
    )
    p.add_argument(
        "--json_out",
        type=str,
        default=None,
        help="Write JSON report to this path",
    )
    p.add_argument(
        "--chunk_rows",
        type=int,
        default=8192,
        help="Process CKA in row blocks of this size (lower = less RAM spike). 0 = one dense pass (higher peak RAM).",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Max BLAS/OpenMP threads (lower = less CPU contention; default 4).",
    )
    p.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Array dtype when loading embeddings (float32 halves RAM vs float64).",
    )
    args = p.parse_args()

    _limit_threads(args.threads)

    rng = np.random.default_rng(args.seed)

    dtype = np.float64 if args.dtype == "float64" else np.float32

    t0 = time.perf_counter()
    X, meta_a = load_embeddings_and_meta(args.a, dtype=dtype)
    Y, meta_b = load_embeddings_and_meta(args.b, dtype=dtype)
    load_s = time.perf_counter() - t0

    validate_aligned_meta(meta_a, meta_b)

    if args.sample_size is not None:
        X, Y, meta_a = subsample_rows(X, Y, meta_a, rng, args.sample_size)
        meta_b = meta_a

    n, d1 = X.shape
    _, d2 = Y.shape

    t1 = time.perf_counter()
    if args.chunk_rows and args.chunk_rows > 0:
        global_cka = linear_cka_chunked(X, Y, args.chunk_rows)
    else:
        global_cka = linear_cka(X, Y)
    cka_s = time.perf_counter() - t1

    report: dict[str, Any] = {
        "file_a": args.a,
        "file_b": args.b,
        "n_rows": n,
        "d_a": d1,
        "d_b": d2,
        "linear_cka": global_cka,
        "load_seconds": load_s,
        "cka_compute_seconds": cka_s,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "chunk_rows": args.chunk_rows,
        "threads": args.threads,
        "dtype": args.dtype,
    }

    print(f"n={n}, d_a={d1}, d_b={d2}, dtype={args.dtype}, chunk_rows={args.chunk_rows}, threads={args.threads}")
    print(f"Linear CKA: {global_cka:.6f}")
    print(f"Load time: {load_s:.2f}s, CKA compute: {cka_s:.2f}s")

    if args.bootstrap > 0:
        t2 = time.perf_counter()
        mean_b, std_b, lo, hi = run_bootstrap(X, Y, rng, args.bootstrap)
        boot_s = time.perf_counter() - t2
        report["bootstrap"] = {
            "n": args.bootstrap,
            "mean": mean_b,
            "std": std_b,
            "ci95_low": lo,
            "ci95_high": hi,
            "seconds": boot_s,
        }
        print(
            f"Bootstrap (n={args.bootstrap}): mean={mean_b:.6f} std={std_b:.6f} "
            f"95% CI [{lo:.6f}, {hi:.6f}] ({boot_s:.2f}s)"
        )

    if args.per_doc:
        by_doc = per_doc_indices(meta_a)
        doc_ckas: list[float] = []
        skipped = 0
        t3 = time.perf_counter()
        for doc_id, idxs in sorted(by_doc.items(), key=lambda x: x[0]):
            if len(idxs) < args.min_doc_rows:
                skipped += 1
                continue
            i = np.array(idxs, dtype=np.intp)
            c = linear_cka(X[i], Y[i])
            if not np.isnan(c):
                doc_ckas.append(c)
        per_doc_s = time.perf_counter() - t3
        if doc_ckas:
            arr = np.array(doc_ckas)
            report["per_doc"] = {
                "n_docs_computed": len(doc_ckas),
                "n_docs_skipped_small": skipped,
                "min_doc_rows": args.min_doc_rows,
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "seconds": per_doc_s,
            }
            print(
                f"Per-doc CKA: docs={len(doc_ckas)} skipped_small={skipped} "
                f"mean={arr.mean():.6f} std={arr.std():.6f} "
                f"min={arr.min():.6f} max={arr.max():.6f} ({per_doc_s:.2f}s)"
            )
        else:
            report["per_doc"] = {"error": "no docs with enough rows"}
            print("Per-doc CKA: no documents met min_doc_rows")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
