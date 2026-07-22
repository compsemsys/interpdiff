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


def _focus_bucket_markdown(
    bucket_rows: list[dict[str, object]],
    *,
    show_category: bool,
) -> str:
    if not bucket_rows:
        return "_No rows_"
    rows: list[list[str]] = []
    for r in bucket_rows:
        if show_category:
            rows.append(
                [
                    str(r.get("category", "")),
                    _label(str(r["model_a"]), str(r["segment_a"])),
                    _label(str(r["model_b"]), str(r["segment_b"])),
                    _fmt(r.get("linear_cka")),  # type: ignore[arg-type]
                    str(r.get("n_rows", 0)),
                ]
            )
        else:
            rows.append(
                [
                    _label(str(r["model_a"]), str(r["segment_a"])),
                    _label(str(r["model_b"]), str(r["segment_b"])),
                    _fmt(r.get("linear_cka")),  # type: ignore[arg-type]
                    str(r.get("n_rows", 0)),
                ]
            )
    headers = (
        ["Category", "A", "B", "CKA", "Rows"]
        if show_category
        else ["A", "B", "CKA", "Rows"]
    )
    return _markdown_table(headers, rows)


def _models_segments_match_row(
    r: dict[str, object],
    ma: str,
    mb: str,
    sa: str,
    sb: str,
) -> bool:
    if str(r.get("segment_a")) != sa or str(r.get("segment_b")) != sb:
        return False
    ra, rb = str(r.get("model_a", "")), str(r.get("model_b", ""))
    if (ra, rb) == (ma, mb):
        return True
    if sa == sb and (ra, rb) == (mb, ma):
        return True
    return False


def _cka_value_for_row(
    recs: list[dict[str, object]],
    category: str | None,
    ma: str,
    mb: str,
    sa: str,
    sb: str,
) -> float | None:
    for r in recs:
        if category is not None and r.get("category") != category:
            continue
        if not _models_segments_match_row(r, ma, mb, sa, sb):
            continue
        v = r.get("linear_cka")
        if isinstance(v, (int, float)) and v == v:  # not NaN
            return float(v)
    return None


def _n_rows_for_category_sample(
    results: list[dict[str, object]],
    category: str,
) -> int | None:
    for r in results:
        if r.get("category") == category:
            n = r.get("n_rows")
            if isinstance(n, int):
                return n
    return None


def _first_n_rows_aggregated(aggregated: list[dict[str, object]]) -> int | None:
    for r in aggregated:
        n = r.get("n_rows")
        if isinstance(n, int):
            return n
    return None


def _category_by_keyword(categories: list[str], keyword: str) -> str | None:
    k = keyword.lower()
    for c in categories:
        if k in c.lower():
            return c
    return None


_SEGMENT_ORDER = {"excerpt": 0, "response": 1, "summary": 2}


def _sort_segments(segments: list[str]) -> list[str]:
    return sorted(segments, key=lambda s: (_SEGMENT_ORDER.get(s, 99), s))


def _summary_row_specs(
    models: list[str], segments: list[str]
) -> list[tuple[str, str, str, str]]:
    """Highlight rows: within-model different segments, then cross-model same segment."""
    segs = _sort_segments(segments)
    specs: list[tuple[str, str, str, str]] = []
    for m in models:
        for i, sa in enumerate(segs):
            for sb in segs[i + 1 :]:
                specs.append((m, m, sa, sb))
    if len(models) == 2:
        m0, m1 = models[0], models[1]
        for s in segs:
            specs.append((m0, m1, s, s))
    return specs


def _focus_bucket_rows(
    focused: dict[str, object], bucket: str
) -> list[dict[str, object]]:
    """Read a focused bucket; accept legacy within-model key name."""
    rows = focused.get(bucket)
    if not rows and bucket == "within_model_different_segments":
        rows = focused.get("within_model_excerpt_vs_response")
    if not isinstance(rows, list):
        return []
    return [r for r in rows if isinstance(r, dict)]


def _summary_table_markdown(
    models: list[str],
    results: list[dict[str, object]],
    aggregated: list[dict[str, object]],
    categories: list[str],
    segments: list[str],
) -> str:
    if len(models) != 2:
        return f"_No summary table: need exactly two models; found {len(models)}._"
    if not segments:
        return "_No summary table: no segments found in the JSON._"

    n_pool = _first_n_rows_aggregated(aggregated)
    cat_science = _category_by_keyword(categories, "science")
    cat_culture = _category_by_keyword(categories, "culture")
    n_s = _n_rows_for_category_sample(results, cat_science) if cat_science else None
    n_c = _n_rows_for_category_sample(results, cat_culture) if cat_culture else None

    h_pool = f"Pooled CKA ({n_pool})" if n_pool is not None else "Pooled CKA"
    h_s = f"Science CKA ({n_s})" if n_s is not None and cat_science else "Science CKA"
    h_c = f"Culture CKA ({n_c})" if n_c is not None and cat_culture else "Culture CKA"

    out_rows: list[list[str]] = []
    for a, b, sa, sb in _summary_row_specs(models, segments):
        ps = _fmt(_cka_value_for_row(aggregated, None, a, b, sa, sb))
        if cat_science:
            ss = _fmt(_cka_value_for_row(results, cat_science, a, b, sa, sb))
        else:
            ss = "—"
        if cat_culture:
            cc = _fmt(_cka_value_for_row(results, cat_culture, a, b, sa, sb))
        else:
            cc = "—"
        out_rows.append(
            [
                _label(a, sa),
                _label(b, sb),
                ps,
                ss,
                cc,
            ]
        )

    lines = _markdown_table(
        ["A", "B", h_pool, h_s, h_c],
        out_rows,
    )
    seg_note = ", ".join(f"**{s}**" for s in _sort_segments(segments))
    if cat_science or cat_culture:
        note = (
            f"_A / B: model and segment for each side "
            f"(within-model different segments, then cross-model same segment). "
            f"Segments: {seg_note}. "
            f"Science column: `{cat_science}`; "
            f"Culture column: `{cat_culture}` (matched from category strings in the JSON)._"
        )
    else:
        note = (
            f"_A / B: model and segment for each side "
            f"(within-model different segments, then cross-model same segment). "
            f"Segments: {seg_note}. "
            "No column matched the keywords *science* or *culture* in your category names._"
        )
    return lines + "\n" + note


def _cka_matrix_markdown(
    cat: str,
    axis: list[tuple[str, str]],
    lookup: dict[tuple[object, str, str, str, str], dict[str, object]],
) -> str:
    headers = ["A \\ B"] + [_label(m, s) for (m, s) in axis]
    table_rows: list[list[str]] = []
    for a_m, a_s in axis:
        row = [_label(a_m, a_s)]
        for b_m, b_s in axis:
            if (a_m, a_s) == (b_m, b_s):
                row.append("1.000000")
                continue
            r = lookup.get((cat, a_m, a_s, b_m, b_s))
            if r is None:
                r = lookup.get((cat, b_m, b_s, a_m, a_s))
            row.append(_fmt(None if r is None else r.get("linear_cka")))  # type: ignore[arg-type]
        table_rows.append(row)
    return _markdown_table(headers, table_rows)


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
    aggregated = data.get("aggregated", [])
    categories = sorted(data.get("categories", []))
    models = sorted(data.get("models", []))
    segments = sorted(data.get("segments", []))
    if not categories:
        categories = sorted(
            {r.get("category", "") for r in results if r.get("category") is not None}
        )
    if not models:
        _recs = list(results) + list(aggregated)
        models = sorted(
            {str(r.get("model_a", "")) for r in _recs}
            | {str(r.get("model_b", "")) for r in _recs}
        )
    if not segments:
        _recs = list(results) + list(aggregated)
        segments = sorted(
            {str(r.get("segment_a", "")) for r in _recs}
            | {str(r.get("segment_b", "")) for r in _recs}
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
    lines.append("# Document CKA (by category and pooled)")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        _summary_table_markdown(
            list(models), results, aggregated, list(categories), list(segments)
        )
    )
    lines.append("")
    lines.append(f"- Source: `{in_path}`")
    lines.append(f"- Models: {', '.join(models) if models else '(none)'}")
    lines.append(f"- Segments: {', '.join(segments) if segments else '(none)'}")
    lines.append(f"- Categories: {', '.join(categories) if categories else '(none)'}")
    lines.append(f"- Per-category pair comparisons: {len(results)}")
    lines.append(f"- Pooled pair comparisons (all doc_id aligned rows): {len(aggregated)}")

    lines.append("")

    pool_key = "(all documents)"
    lookup_pooled: dict[
        tuple[object, str, str, str, str],
        dict[str, object],
    ] = {}
    for r in aggregated:
        rk = (
            r.get("category", pool_key),
            r["model_a"],
            r["segment_a"],
            r["model_b"],
            r["segment_b"],
        )
        lookup_pooled[rk] = r

    lines.append("")
    lines.append("## Pooled (all documents)")
    lines.append("")
    lines.append(
        "One CKA per model/segment **pair** using the full set of document embeddings aligned by `doc_id` "
        "(not restricted to a single category)."
    )
    lines.append("")

    agg_focused = data.get("aggregated_focused", {}) or {}
    if not isinstance(agg_focused, dict):
        agg_focused = {}
    for section_title, bucket in (
        ("Cross model, same segment", "cross_model_same_segment"),
        ("Within model, different segments", "within_model_different_segments"),
    ):
        lines.append(f"### {section_title}")
        lines.append("")
        lines.append(
            _focus_bucket_markdown(
                _focus_bucket_rows(agg_focused, bucket),
                show_category=False,
            )
        )
        lines.append("")

    lines.append("### Full matrix (pooled)")
    lines.append("")
    if not aggregated:
        lines.append(
            "_No pooled data in this file; re-run the `cka` stage with a current `run_categorized_corpus.py` "
            "to write `aggregated` into `document_cka_by_category.json`._"
        )
        lines.append("")
    else:
        axis = [(m, s) for m in models for s in _sort_segments(list(segments))]
        lines.append(_cka_matrix_markdown(pool_key, axis, lookup_pooled))
        lines.append("")

    lines.append("## Per-category: highlighted comparisons")
    lines.append("")

    focused = data.get("focused", {}) or {}
    if not isinstance(focused, dict):
        focused = {}
    for section_title, bucket in (
        ("Cross model, same segment", "cross_model_same_segment"),
        ("Within model, different segments", "within_model_different_segments"),
    ):
        lines.append(f"### {section_title}")
        lines.append("")
        lines.append(
            _focus_bucket_markdown(
                _focus_bucket_rows(focused, bucket),
                show_category=True,
            )
        )
        lines.append("")

    lines.append("## Per-Category Matrices")
    lines.append("")
    axis = [(m, s) for m in models for s in _sort_segments(list(segments))]
    for cat in categories:
        lines.append(f"### {cat}")
        lines.append("")
        lines.append(_cka_matrix_markdown(cat, axis, lookup))
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
