"""
Linear CKA between two aligned embedding matrices.

Uses the linear (inner-product) CKA from Kornblith et al.,
"Similarity of Neural Network Representations Revisited" (ICML 2019).
Rows must correspond to the same instances (same meta order), e.g.
word-level rows or document-level rows.

Key functions:
- ``pairwise_linear_cka_from_paths``: load aligned npy files, apply optional row slicing, run CKA.
- ``row_indices_meta_match``: resolve metadata/doc-id row filters to index arrays.
- ``linear_cka_chunked``: memory-aware CKA accumulation across row blocks.
- ``validate_aligned_meta``: enforce row identity alignment before any comparison.
- ``run_leave_k_out_cka``: delete-k row stability (drop random rows, recompute CKA).
- ``main``: CLI entry for quick comparisons, filtering, bootstrap, leave-k-out, and JSON reporting.

Example:
  python cka_word_embeddings.py \\
    --a F:/quantas/outputs/.../word_embeddings_merged_agnostic.npy \\
    --b F:/quantas/outputs/.../word_embeddings_merged_agnostic.npy
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
    meta = list(data["meta"])
    # Newer runs include top-level doc_ids[i] == meta[i]["doc_id"]; older files omit it.
    if "doc_ids" in data:
        doc_ids = np.asarray(data["doc_ids"])
        if len(doc_ids) != emb.shape[0]:
            raise ValueError(
                f"doc_ids/embeddings length mismatch for {path}: "
                f"{len(doc_ids)} vs {emb.shape[0]}"
            )
        for i, (did, m) in enumerate(zip(doc_ids, meta)):
            if int(m["doc_id"]) != int(did):
                raise ValueError(
                    f"doc_ids/meta mismatch for {path} at row {i}: "
                    f"doc_ids[{i}]={int(did)} meta.doc_id={int(m['doc_id'])}"
                )
    return emb, meta


def load_doc_ids(path: str) -> np.ndarray:
    """
    Explicit per-row document IDs for an embeddings ``.npy``.

    Prefers the top-level ``doc_ids`` array written by newer pipeline runs;
    falls back to ``meta[i]['doc_id']`` for older files.
    """
    data = np.load(path, allow_pickle=True).item()
    if "doc_ids" in data:
        return np.asarray(data["doc_ids"], dtype=np.int64)
    meta_key = "meta" if "meta" in data else "token_meta"
    meta = data[meta_key]
    if meta and isinstance(meta[0], list):
        meta = [item for chunk in meta for item in chunk]
    return np.asarray([int(m["doc_id"]) for m in meta], dtype=np.int64)


def meta_key(m: dict[str, Any]) -> tuple:
    """Comparable row identity for word-level and document-level metadata."""
    if "span" in m and "word" in m:
        return ("word", int(m["doc_id"]), tuple(m["span"]), m["word"])
    if "doc_id" in m:
        return (
            "document",
            int(m["doc_id"]),
            m.get("segment", ""),
            m.get("category", ""),
            m.get("title", ""),
        )
    # Fallback for unexpected row metadata schemas.
    return ("generic", json.dumps(m, sort_keys=True, default=str))


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


def parse_meta_filter_arg(spec: str) -> tuple[str, str]:
    """
    Parse ``KEY=VALUE`` for row-wise metadata equality (embedding ``meta`` rows).

    The first ``=`` separates key and value; values may contain ``=``.
    """
    k, sep, v = spec.partition("=")
    if not sep:
        raise ValueError(f"Filter must be KEY=VALUE, got {spec!r}")
    key, val = k.strip(), v.strip()
    if not key:
        raise ValueError(f"Empty filter key in {spec!r}")
    return key, val


def parse_doc_ids_arg(s: str | None) -> set[int] | None:
    """Comma-separated doc ids; empty or None means no doc-id restriction."""
    if not s or not str(s).strip():
        return None
    out: set[int] = set()
    for part in str(s).split(","):
        part = part.strip()
        if part:
            out.add(int(part))
    return out if out else None


def slice_spec_dict(
    filters: list[tuple[str, str]],
    doc_ids: set[int] | None,
) -> dict[str, Any]:
    return {
        "filters": [[a, b] for a, b in filters],
        "doc_ids": sorted(doc_ids) if doc_ids else None,
    }


def row_indices_meta_match(
    meta: list[dict[str, Any]],
    *,
    filters: list[tuple[str, str]] | None = None,
    doc_ids: set[int] | None = None,
) -> np.ndarray:
    """
    Indices of rows whose metadata passes all equality ``filters`` and optional ``doc_ids``.

    With no filters and no ``doc_ids``, returns ``arange(len(meta))``.
    """
    filters = filters or []
    if not filters and doc_ids is None:
        return np.arange(len(meta), dtype=np.intp)
    # Keep explicit Python filtering to support mixed meta schemas robustly.
    idxs: list[int] = []
    for i, m in enumerate(meta):
        if doc_ids is not None and int(m.get("doc_id", -1)) not in doc_ids:
            continue
        ok = True
        for key, val in filters:
            if key == "doc_id":
                if int(m.get("doc_id", -(10**18))) != int(val):
                    ok = False
                    break
            else:
                cur = m.get(key, "")
                if cur is None:
                    cur = ""
                if str(cur) != val:
                    ok = False
                    break
        if ok:
            idxs.append(i)
    return np.asarray(idxs, dtype=np.intp)


def apply_row_indices(
    X: np.ndarray,
    Y: np.ndarray,
    meta_a: list,
    meta_b: list,
    idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list, list]:
    """Subset aligned matrices and meta to the given row indices."""
    if len(idx) == 0:
        raise ValueError("No rows selected")
    Xs = np.asarray(X[idx], dtype=X.dtype)
    Ys = np.asarray(Y[idx], dtype=Y.dtype)
    sma = [meta_a[i] for i in idx]
    smb = [meta_b[i] for i in idx]
    validate_aligned_meta(sma, smb)
    return Xs, Ys, sma, smb


def cka_row_slice_file_suffix(
    filters: list[tuple[str, str]],
    doc_ids: set[int] | None,
    label: str | None,
) -> str:
    """
    Filename suffix for CKA JSON outputs when row slicing is active.

    Empty string means use the default unsliced basename.
    """
    if not filters and not doc_ids:
        return ""
    if label and label.strip():
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", label.strip())[:64].strip("_")
        if safe:
            return "__" + safe
    payload = json.dumps(
        {"f": filters, "d": sorted(doc_ids) if doc_ids else []},
        sort_keys=True,
    ).encode()
    return "__slice_" + hashlib.md5(payload, usedforsecurity=False).hexdigest()[:8]


def pairwise_linear_cka_from_paths(
    path_a: str,
    path_b: str,
    *,
    chunk_rows: int,
    filters: list[tuple[str, str]] | None = None,
    doc_ids: set[int] | None = None,
) -> dict[str, Any]:
    """
    Load two aligned embedding npy files, optionally slice rows by metadata, run chunked linear CKA.

    Returns a dict suitable for JSON (includes ``n_rows``, ``n_rows_total``, optional ``row_slice``).
    """
    filters = list(filters or [])
    xa, ma = load_embeddings_and_meta(path_a)
    xb, mb = load_embeddings_and_meta(path_b)
    validate_aligned_meta(ma, mb)
    idx = row_indices_meta_match(ma, filters=filters, doc_ids=doc_ids)
    n_total = len(ma)
    has_slice = bool(filters) or bool(doc_ids)
    if len(idx) == 0:
        return {
            "linear_cka": float("nan"),
            "path_a": path_a,
            "path_b": path_b,
            "n_rows": 0,
            "n_rows_total": n_total,
            "row_slice": slice_spec_dict(filters, doc_ids) if has_slice else None,
            "error": "no rows matched filters",
        }
    if len(idx) < len(ma):
        xa_s, xb_s, _, _ = apply_row_indices(xa, xb, ma, mb, idx)
    else:
        xa_s, xb_s = xa, xb
    val = float(linear_cka_chunked(xa_s, xb_s, chunk_rows))
    out: dict[str, Any] = {
        "linear_cka": val,
        "path_a": path_a,
        "path_b": path_b,
        "n_rows": int(xa_s.shape[0]),
        "n_rows_total": n_total,
    }
    if has_slice:
        out["row_slice"] = slice_spec_dict(filters, doc_ids)
    return out


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

    # Stream row blocks to reduce peak RAM compared with one monolithic multiply.
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


def run_leave_k_out_cka(
    X: np.ndarray,
    Y: np.ndarray,
    rng: np.random.Generator,
    *,
    drop_k: int = 10,
    n_reps: int = 100,
) -> tuple[float, float, np.ndarray]:
    """
    Leave-k-out (delete-k) CKA stability: drop ``drop_k`` random rows from both
    matrices, compute linear CKA on the remainder, repeat ``n_reps`` times.

    Returns ``(mean, std, scores)`` where ``scores`` has length ``n_reps``.
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"Same n required: {X.shape[0]} vs {Y.shape[0]}")
    n = int(X.shape[0])
    drop_k = int(drop_k)
    n_reps = int(n_reps)
    if n_reps < 1:
        raise ValueError(f"n_reps must be >= 1, got {n_reps}")
    if drop_k < 1 or drop_k >= n:
        raise ValueError(f"drop_k must satisfy 1 <= drop_k < n ({n}), got {drop_k}")
    if n - drop_k < 2:
        raise ValueError(
            f"Need at least 2 rows after dropping: n={n}, drop_k={drop_k}"
        )

    scores = np.empty(n_reps, dtype=np.float64)
    for r in range(n_reps):
        drop = rng.choice(n, size=drop_k, replace=False)
        mask = np.ones(n, dtype=bool)
        mask[drop] = False
        scores[r] = linear_cka(X[mask], Y[mask])
    return float(np.mean(scores)), float(np.std(scores)), scores


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
        "--leave_k_out",
        type=int,
        default=0,
        metavar="N",
        help="Leave-k-out reps: drop --drop_k random rows each time (0 = off)",
    )
    p.add_argument(
        "--drop_k",
        type=int,
        default=10,
        help="Rows to drop per leave-k-out rep (default 10)",
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
        "--cka_filter",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Keep rows whose word meta matches all filters (AND). Example: --cka_filter category=Category:Statistics",
    )
    p.add_argument(
        "--cka_doc_ids",
        default=None,
        metavar="IDS",
        help="Comma-separated doc_id list (intersection with --cka_filter).",
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

    row_filters: list[tuple[str, str]] = []
    if args.cka_filter:
        try:
            for spec in args.cka_filter:
                row_filters.append(parse_meta_filter_arg(spec))
        except ValueError as e:
            p.error(str(e))
    row_doc_ids = parse_doc_ids_arg(args.cka_doc_ids)

    _limit_threads(args.threads)

    rng = np.random.default_rng(args.seed)

    dtype = np.float64 if args.dtype == "float64" else np.float32

    t0 = time.perf_counter()
    X, meta_a = load_embeddings_and_meta(args.a, dtype=dtype)
    Y, meta_b = load_embeddings_and_meta(args.b, dtype=dtype)
    load_s = time.perf_counter() - t0

    validate_aligned_meta(meta_a, meta_b)

    if row_filters or row_doc_ids:
        idx = row_indices_meta_match(meta_a, filters=row_filters, doc_ids=row_doc_ids)
        if len(idx) == 0:
            p.error("No rows matched --cka_filter / --cka_doc_ids")
        X, Y, meta_a, meta_b = apply_row_indices(X, Y, meta_a, meta_b, idx)

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
        "row_slice": slice_spec_dict(row_filters, row_doc_ids)
        if (row_filters or row_doc_ids)
        else None,
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

    if args.leave_k_out > 0:
        t_lko = time.perf_counter()
        mean_l, std_l, scores_l = run_leave_k_out_cka(
            X, Y, rng, drop_k=args.drop_k, n_reps=args.leave_k_out
        )
        lko_s = time.perf_counter() - t_lko
        report["leave_k_out"] = {
            "n_reps": args.leave_k_out,
            "drop_k": args.drop_k,
            "n_rows_kept": n - args.drop_k,
            "mean": mean_l,
            "std": std_l,
            "scores": scores_l.tolist(),
            "seconds": lko_s,
        }
        print(
            f"Leave-k-out (drop_k={args.drop_k}, n_reps={args.leave_k_out}): "
            f"mean={mean_l:.6f} std={std_l:.6f} ({lko_s:.2f}s)"
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
