"""
Render a readable markdown report from document_cka_by_category.json.

Input:
  outputs/<run>/cka/document_cka_by_category.json

Output (default):
  outputs/<run>/cka/document_cka_by_category.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _label(model: str, segment: str) -> str:
    return f"{model} ({segment})"


def _fmt(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.6f}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        default=None,
        help=(
            "Path to document_cka_by_category.json "
            "(default: <out_dir>/cka/document_cka_by_category.json)"
        ),
    )
    p.add_argument(
        "--out_dir",
        default=None,
        help="Run directory, used when --input is omitted",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output markdown path (default: alongside input with .md)",
    )
    args = p.parse_args()

    if args.input:
        in_path = Path(args.input).resolve()
    else:
        if not args.out_dir:
            raise SystemExit("Provide either --input or --out_dir.")
        in_path = Path(args.out_dir).resolve() / "cka" / "document_cka_by_category.json"

    if not in_path.is_file():
        raise SystemExit(f"Input JSON not found: {in_path}")

    data = json.loads(in_path.read_text(encoding="utf-8"))
    out_path = (
        Path(args.output).resolve()
        if args.output
        else in_path.with_suffix(".md")
    )

    results = data.get("results", [])
    categories = sorted(data.get("categories", []))
    models = sorted(data.get("models", []))
    segments = sorted(data.get("segments", []))
    if not categories:
        categories = sorted({r.get("category", "") for r in results})
    if not models:
        models = sorted(
            {
                str(r.get("model_a", ""))
                for r in results
            }
            | {
                str(r.get("model_b", ""))
                for r in results
            }
        )
    if not segments:
        segments = sorted(
            {
                str(r.get("segment_a", ""))
                for r in results
            }
            | {
                str(r.get("segment_b", ""))
                for r in results
            }
        )

    # Build lookup for directed access.
    lookup = {}
    for r in results:
        key = (
            r["category"],
            r["model_a"],
            r["segment_a"],
            r["model_b"],
            r["segment_b"],
        )
        lookup[key] = r

    lines: list[str] = []
    lines.append("# Document CKA By Category")
    lines.append("")
    lines.append(f"- Source: `{in_path}`")
    lines.append(f"- Models: {', '.join(models) if models else '(none)'}")
    lines.append(f"- Segments: {', '.join(segments) if segments else '(none)'}")
    lines.append(f"- Categories: {', '.join(categories) if categories else '(none)'}")
    lines.append(f"- Comparisons: {len(results)}")
    lines.append("")

    focused = data.get("focused", {})
    for bucket in (
        "cross_model_same_segment",
        "within_model_excerpt_vs_response",
        "cross_model_cross_segment",
    ):
        bucket_rows = focused.get(bucket, [])
        lines.append(f"## {bucket.replace('_', ' ').title()}")
        lines.append("")
        if not bucket_rows:
            lines.append("_No rows_")
            lines.append("")
            continue
        rows = []
        for r in bucket_rows:
            rows.append(
                [
                    r["category"],
                    _label(r["model_a"], r["segment_a"]),
                    _label(r["model_b"], r["segment_b"]),
                    _fmt(r.get("linear_cka")),
                    str(r.get("n_rows", 0)),
                ]
            )
        lines.append(
            _markdown_table(
                ["Category", "A", "B", "CKA", "Rows"],
                rows,
            )
        )
        lines.append("")

    # Category matrices
    lines.append("## Per-Category Matrices")
    lines.append("")
    axis = [(m, s) for m in models for s in segments]
    for cat in categories:
        lines.append(f"### {cat}")
        lines.append("")
        headers = ["A \\ B"] + [_label(m, s) for (m, s) in axis]
        table_rows = []
        for a_m, a_s in axis:
            row = [_label(a_m, a_s)]
            for b_m, b_s in axis:
                if (a_m, a_s) == (b_m, b_s):
                    row.append("1.000000")
                    continue
                r = lookup.get((cat, a_m, a_s, b_m, b_s))
                if r is None:
                    r = lookup.get((cat, b_m, b_s, a_m, a_s))
                row.append(_fmt(None if r is None else r.get("linear_cka")))
            table_rows.append(row)
        lines.append(_markdown_table(headers, table_rows))
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
