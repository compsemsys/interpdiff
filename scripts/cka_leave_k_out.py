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


def _side_label(
    *,
    model: str,
    segment: str,
    embedder: str | None = None,
    text_source: str | None = None,
) -> str:
    emb = (embedder or "").strip()
    text = (text_source or "").strip()
    if emb or text:
        emb_s = emb or model
        text_s = text or model
        # Use "/" not "|": pipes break GitHub/Cursor markdown tables.
        return f"emb={emb_s} / text={text_s} ({segment})"
    return _label(model, segment)


def _pair_side_labels(r: dict[str, Any]) -> tuple[str, str]:
    ma = str(r.get("model_a", ""))
    mb = str(r.get("model_b", ""))
    sa = str(r.get("segment_a", ""))
    sb = str(r.get("segment_b", ""))
    ea = r.get("embedder_a")
    eb = r.get("embedder_b")
    shared_ts = r.get("text_source")
    tsa = r.get("text_source_a")
    tsb = r.get("text_source_b")
    if tsa is None or (isinstance(tsa, str) and not tsa.strip()):
        tsa = shared_ts
    if tsb is None or (isinstance(tsb, str) and not tsb.strip()):
        tsb = shared_ts
    has_cross = any(
        x is not None and str(x).strip() != ""
        for x in (ea, eb, tsa, tsb, r.get("comparison"))
    )
    if not has_cross:
        return _label(ma, sa), _label(mb, sb)
    return (
        _side_label(
            model=ma,
            segment=sa,
            embedder=str(ea) if ea is not None else None,
            text_source=str(tsa) if tsa is not None else None,
        ),
        _side_label(
            model=mb,
            segment=sb,
            embedder=str(eb) if eb is not None else None,
            text_source=str(tsb) if tsb is not None else None,
        ),
    )


_CROSS_EMBED_BUCKETS: tuple[str, ...] = (
    "same_text_cross_embedder",
    "same_embedder_excerpt_vs_foreign_text",
    "same_embedder_own_vs_foreign_text",
)


def _fmt(v: float | None) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return f"{v:.6f}"


def _md_cell(s: str) -> str:
    """Escape characters that break pipe tables."""
    return str(s).replace("|", "\\|")


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out = [
        "| " + " | ".join(_md_cell(h) for h in headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        out.append("| " + " | ".join(_md_cell(c) for c in r) + " |")
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
    """Own-embed focused pooled pairs, plus cross-embed pooled pairs when present."""
    agf = doc.get("aggregated_focused") or {}
    if not isinstance(agf, dict):
        agf = {}
    within = _focus_bucket_rows(agf, "within_model_different_segments")
    cross = _focus_bucket_rows(agf, "cross_model_same_segment")
    pairs = within + cross

    cross_embed = doc.get("cross_embed") or {}
    if isinstance(cross_embed, dict):
        cef = cross_embed.get("aggregated_focused") or {}
        if isinstance(cef, dict):
            for bucket in _CROSS_EMBED_BUCKETS:
                pairs.extend(_focus_bucket_rows(cef, bucket))
    return pairs


def _copy_cross_fields(src: dict[str, Any], dest: dict[str, Any]) -> None:
    for key in (
        "comparison",
        "text_source",
        "text_source_a",
        "text_source_b",
        "embedder_a",
        "embedder_b",
    ):
        if key in src:
            dest[key] = src[key]


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
        label_a, label_b = _pair_side_labels(rec)

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
            "label_a": label_a,
            "label_b": label_b,
        }
        _copy_cross_fields(rec, out_rec)

        if n < 2:
            out_rec["linear_cka"] = float("nan")
            out_rec["error"] = "fewer than 2 aligned docs"
            results.append(out_rec)
            print(f"[skip] {label_a} vs {label_b}: n={n}")
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
            print(f"[error] {label_a} vs {label_b}: {e}")
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
            f"{label_a} vs {label_b}: "
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
        la = str(r.get("label_a") or _label(str(r["model_a"]), str(r["segment_a"])))
        lb = str(r.get("label_b") or _label(str(r["model_b"]), str(r["segment_b"])))
        md_rows.append(
            [
                str(r.get("comparison") or "own_embed"),
                la,
                lb,
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
            [
                "Comparison",
                "A",
                "B",
                "Full CKA",
                "Leave-k-out mean",
                "Leave-k-out std",
                "Rows",
            ],
            md_rows,
        ),
        "",
    ]
    md_out.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {json_out}")
    print(f"Wrote {md_out}")


if __name__ == "__main__":
    main()
