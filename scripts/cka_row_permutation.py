"""
Topic-correspondence row-permutation CKA null on focused pooled document pairs.

Reads ``<out_dir>/cka/document_cka_by_category.json``, takes the focused pooled
pairs (within-model different segments + cross-model same segment + cross-embed),
reloads aligned embeddings, and for each pair randomly permutes rows of one
representation (Y) ``n_reps`` times, recording mean/std of linear CKA.

Example:
  python scripts/cka_row_permutation.py --out_dir outputs/my_run12
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

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from cka_leave_k_out import (  # noqa: E402
    _copy_cross_fields,
    _fmt,
    _label,
    _markdown_table,
    _pair_side_labels,
    align_embeddings_by_doc_id,
    focused_pooled_pairs,
)
from cka_word_embeddings import (  # noqa: E402
    linear_cka,
    load_embeddings_and_meta,
    run_row_permutation_cka,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Row-permutation CKA null on focused pooled document pairs from a run dir"
        )
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
    p.add_argument(
        "--n_reps",
        type=int,
        default=100,
        help="Number of row-permutation reps (default 100)",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")
    p.add_argument(
        "--json_out",
        type=str,
        default=None,
        help="JSON output path (default: <out_dir>/cka/row_permutation_pooled.json)",
    )
    p.add_argument(
        "--md_out",
        type=str,
        default=None,
        help="Markdown output path (default: <out_dir>/cka/row_permutation_pooled.md)",
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
            mean_p, std_p, scores = run_row_permutation_cka(
                xa, xb, rng, n_reps=args.n_reps
            )
        except ValueError as e:
            out_rec["error"] = str(e)
            results.append(out_rec)
            print(f"[error] {label_a} vs {label_b}: {e}")
            continue

        out_rec["row_permutation"] = {
            "n_reps": int(args.n_reps),
            "permute": "Y",
            "mean": mean_p,
            "std": std_p,
            "scores": scores.tolist(),
        }
        results.append(out_rec)
        print(
            f"{label_a} vs {label_b}: "
            f"cka={full_cka:.6f} row-perm mean={mean_p:.6f} std={std_p:.6f} "
            f"(n={n}, reps={args.n_reps})"
        )

    elapsed = time.perf_counter() - t0
    report: dict[str, Any] = {
        "analysis": "row_permutation_pooled",
        "out_dir": str(out_dir),
        "source": str(in_path),
        "n_reps": int(args.n_reps),
        "seed": int(args.seed),
        "permute": "Y",
        "n_pairs": len(results),
        "seconds": elapsed,
        "results": results,
    }

    json_out = (
        Path(args.json_out).resolve()
        if args.json_out
        else out_dir / "cka" / "row_permutation_pooled.json"
    )
    md_out = (
        Path(args.md_out).resolve()
        if args.md_out
        else out_dir / "cka" / "row_permutation_pooled.md"
    )
    json_out.parent.mkdir(parents=True, exist_ok=True)

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_rows: list[list[str]] = []
    for r in results:
        perm = r.get("row_permutation") or {}
        mean_s = _fmt(perm.get("mean") if isinstance(perm, dict) else None)
        std_s = _fmt(perm.get("std") if isinstance(perm, dict) else None)
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
        "# Row-permutation CKA null (focused pooled)",
        "",
        f"- Source: `{in_path}`",
        f"- permute=Y, n_reps={args.n_reps}, seed={args.seed}",
        f"- Pairs: {len(results)}",
        f"- Elapsed: {elapsed:.2f}s",
        "",
        _markdown_table(
            [
                "Comparison",
                "A",
                "B",
                "Full CKA",
                "Row-perm mean",
                "Row-perm std",
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
