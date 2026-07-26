"""
Leave-k-out CKA stability on focused pooled document comparisons.

Reads ``<out_dir>/cka/document_cka_by_category.json``, takes the focused pooled
pairs (within-model different segments + cross-model same segment), reloads
aligned embeddings, and for each pair drops ``drop_k`` random rows ``n_reps``
times, recording mean/std of linear CKA.

Example:
  python scripts/cka_leave_k_out.py --out_dir outputs/my_run11
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cka_word_embeddings import (  # noqa: E402
    linear_cka,
    load_embeddings_and_meta,
    run_leave_k_out_cka,
)


def _label(model: str, segment: str) -> str:
    return f"{model} ({segment})"


def _fmt(v: float | None) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return f"{v:.6f}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def align_embeddings_by_doc_id(
    emb_a: np.ndarray,
    meta_a: list[dict[str, Any]],
    emb_b: np.ndarray,
    meta_b: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Align document embedding rows by shared ``doc_id`` (sorted)."""
    idx_a: dict[int, int] = {}
    idx_b: dict[int, int] = {}
    for i, m in enumerate(meta_a):
        idx_a[int(m["doc_id"])] = i
    for i, m in enumerate(meta_b):
        idx_b[int(m["doc_id"])] = i
    shared = sorted(set(idx_a).intersection(idx_b))
    if not shared:
        return (
            np.zeros((0, emb_a.shape[1]), dtype=emb_a.dtype),
            np.zeros((0, emb_b.shape[1]), dtype=emb_b.dtype),
            [],
        )
    ia = [idx_a[d] for d in shared]
    ib = [idx_b[d] for d in shared]
    return emb_a[ia], emb_b[ib], shared


def _focus_bucket_rows(
    focused: dict[str, Any], bucket: str
) -> list[dict[str, Any]]:
    rows = focused.get(bucket)
    if not rows and bucket == "within_model_different_segments":
        rows = focused.get("within_model_excerpt_vs_response")
    if not isinstance(rows, list):
        return []
    return [r for r in rows if isinstance(r, dict)]


def focused_pooled_pairs(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Nine (typical) summary pairs from aggregated_focused."""
    agf = doc.get("aggregated_focused") or {}
    if not isinstance(agf, dict):
        return []
    within = _focus_bucket_rows(agf, "within_model_different_segments")
    cross = _focus_bucket_rows(agf, "cross_model_same_segment")
    return within + cross


def main() -> None:
    p = argparse.ArgumentParser(
        description="Leave-k-out CKA on focused pooled document pairs from a run dir"
    )
    p.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Pipeline run directory (contains cka/document_cka_by_category.json)",
    )
    p.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to document_cka_by_category.json (default: <out_dir>/cka/...)",
    )
    p.add_argument("--drop_k", type=int, default=10, help="Rows to drop per rep")
    p.add_argument("--n_reps", type=int, default=100, help="Number of leave-k-out reps")
    p.add_argument("--seed", type=int, default=0, help="RNG seed")
    p.add_argument(
        "--json_out",
        type=str,
        default=None,
        help="JSON output path (default: <out_dir>/cka/leave_k_out_pooled.json)",
    )
    p.add_argument(
        "--md_out",
        type=str,
        default=None,
        help="Markdown output path (default: <out_dir>/cka/leave_k_out_pooled.md)",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir).resolve()
    in_path = (
        Path(args.input).resolve()
        if args.input
        else out_dir / "cka" / "document_cka_by_category.json"
    )
    if not in_path.is_file():
        raise SystemExit(f"Missing input: {in_path}")

    with open(in_path, encoding="utf-8") as f:
        doc = json.load(f)

    pairs = focused_pooled_pairs(doc)
    if not pairs:
        raise SystemExit(
            f"No aggregated_focused pooled pairs in {in_path}; "
            "re-run the cka stage with a current pipeline."
        )

    rng = np.random.default_rng(args.seed)
    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    for rec in pairs:
        path_a = str(rec["path_a"])
        path_b = str(rec["path_b"])
        ea, meta_a = load_embeddings_and_meta(path_a)
        eb, meta_b = load_embeddings_and_meta(path_b)
        xa, xb, shared_doc_ids = align_embeddings_by_doc_id(ea, meta_a, eb, meta_b)
        n = int(xa.shape[0])
        ma, mb = str(rec["model_a"]), str(rec["model_b"])
        sa, sb = str(rec["segment_a"]), str(rec["segment_b"])

        out_rec: dict[str, Any] = {
            "model_a": ma,
            "model_b": mb,
            "segment_a": sa,
            "segment_b": sb,
            "path_a": path_a,
            "path_b": path_b,
            "n_rows": n,
            "doc_ids": shared_doc_ids,
            "full_linear_cka": rec.get("linear_cka"),
        }

        if n < 2:
            out_rec["linear_cka"] = float("nan")
            out_rec["error"] = "fewer than 2 aligned docs"
            results.append(out_rec)
            print(f"[skip] {_label(ma, sa)} vs {_label(mb, sb)}: n={n}")
            continue

        full_cka = float(linear_cka(xa, xb))
        out_rec["linear_cka"] = full_cka

        try:
            mean_l, std_l, scores = run_leave_k_out_cka(
                xa, xb, rng, drop_k=args.drop_k, n_reps=args.n_reps
            )
        except ValueError as e:
            out_rec["error"] = str(e)
            results.append(out_rec)
            print(f"[error] {_label(ma, sa)} vs {_label(mb, sb)}: {e}")
            continue

        out_rec["leave_k_out"] = {
            "drop_k": int(args.drop_k),
            "n_reps": int(args.n_reps),
            "n_rows_kept": n - int(args.drop_k),
            "mean": mean_l,
            "std": std_l,
            "scores": scores.tolist(),
        }
        results.append(out_rec)
        print(
            f"{_label(ma, sa)} vs {_label(mb, sb)}: "
            f"cka={full_cka:.6f} leave-k-out mean={mean_l:.6f} std={std_l:.6f} "
            f"(n={n}, drop_k={args.drop_k}, reps={args.n_reps})"
        )

    elapsed = time.perf_counter() - t0
    report: dict[str, Any] = {
        "analysis": "leave_k_out_pooled",
        "out_dir": str(out_dir),
        "source": str(in_path),
        "drop_k": int(args.drop_k),
        "n_reps": int(args.n_reps),
        "seed": int(args.seed),
        "n_pairs": len(results),
        "seconds": elapsed,
        "results": results,
    }

    json_out = (
        Path(args.json_out).resolve()
        if args.json_out
        else out_dir / "cka" / "leave_k_out_pooled.json"
    )
    md_out = (
        Path(args.md_out).resolve()
        if args.md_out
        else out_dir / "cka" / "leave_k_out_pooled.md"
    )
    json_out.parent.mkdir(parents=True, exist_ok=True)

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_rows: list[list[str]] = []
    for r in results:
        lko = r.get("leave_k_out") or {}
        mean_s = _fmt(lko.get("mean") if isinstance(lko, dict) else None)
        std_s = _fmt(lko.get("std") if isinstance(lko, dict) else None)
        md_rows.append(
            [
                _label(str(r["model_a"]), str(r["segment_a"])),
                _label(str(r["model_b"]), str(r["segment_b"])),
                _fmt(r.get("linear_cka")),  # type: ignore[arg-type]
                mean_s,
                std_s,
                str(r.get("n_rows", "")),
            ]
        )

    lines = [
        "# Leave-k-out CKA (focused pooled)",
        "",
        f"- Source: `{in_path}`",
        f"- drop_k={args.drop_k}, n_reps={args.n_reps}, seed={args.seed}",
        f"- Pairs: {len(results)}",
        f"- Elapsed: {elapsed:.2f}s",
        "",
        _markdown_table(
            ["A", "B", "Full CKA", "Leave-k-out mean", "Leave-k-out std", "Rows"],
            md_rows,
        ),
        "",
    ]
    md_out.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {json_out}")
    print(f"Wrote {md_out}")


if __name__ == "__main__":
    main()
