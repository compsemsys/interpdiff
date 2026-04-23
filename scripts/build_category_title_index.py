from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

# Allow running as "python scripts/build_category_title_index.py" from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wiki_tree_random_articles import normalize_category_title


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _iter_records(path: Path) -> Iterable[dict[str, Any]]:
    with _open_text(path) as f:
        first_nonempty: str | None = None
        buffered_lines: list[str] = []
        for line in f:
            s = line.strip()
            if not s:
                continue
            first_nonempty = s
            buffered_lines.append(line)
            break

        if first_nonempty is None:
            return

        if first_nonempty.startswith("[") or first_nonempty.startswith("{"):
            # JSON array/object mode: parse whole file.
            rest = f.read()
            payload = "".join(buffered_lines) + rest
            obj = json.loads(payload)
            if isinstance(obj, list):
                for x in obj:
                    if isinstance(x, dict):
                        yield x
            elif isinstance(obj, dict):
                # common wrappers: {"items":[...]} or {"articles":[...]}
                for key in ("items", "articles", "data", "results"):
                    arr = obj.get(key)
                    if isinstance(arr, list):
                        for x in arr:
                            if isinstance(x, dict):
                                yield x
                        return
                # single-record json object
                yield obj
            return

        # NDJSON mode (line-delimited objects)
        for line in buffered_lines:
            s = line.strip()
            if s:
                x = json.loads(s)
                if isinstance(x, dict):
                    yield x
        for line in f:
            s = line.strip()
            if not s:
                continue
            x = json.loads(s)
            if isinstance(x, dict):
                yield x


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Build local category->title index from Wikimedia Enterprise snapshot files "
            "(JSON/JSONL, optionally .gz)."
        )
    )
    p.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        metavar="PATH",
        help="Snapshot file(s): .json, .jsonl/.ndjson, and/or .gz variants.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/wiki_snapshot_category_index.json"),
        help="Output index JSON path.",
    )
    p.add_argument(
        "--project",
        default="enwiki",
        help="Project identifier filter (e.g. enwiki). Empty means all projects.",
    )
    p.add_argument(
        "--namespace-id",
        type=int,
        default=0,
        help="Only index records with this namespace identifier (default: 0, articles).",
    )
    p.add_argument(
        "--max-titles-per-category",
        type=int,
        default=0,
        help="Optional cap per category in output (0 = unlimited).",
    )
    args = p.parse_args()

    project_filter = (args.project or "").strip()
    by_category: dict[str, set[str]] = defaultdict(set)

    n_records = 0
    n_kept = 0
    for ipath in args.inputs:
        path = Path(ipath)
        if not path.exists():
            raise SystemExit(f"Missing input: {path}")
        print(f"Indexing: {path}")
        for rec in _iter_records(path):
            n_records += 1
            title = str(rec.get("name", "") or "").strip()
            if not title:
                continue
            ns = int((rec.get("namespace") or {}).get("identifier", -1))
            if ns != args.namespace_id:
                continue
            project = str((rec.get("is_part_of") or {}).get("identifier", "") or "").strip()
            if project_filter and project != project_filter:
                continue
            cats = rec.get("categories") or []
            if not isinstance(cats, list):
                continue
            any_cat = False
            for c in cats:
                if not isinstance(c, dict):
                    continue
                cname = str(c.get("name", "") or "").strip()
                if not cname:
                    continue
                cname_n = normalize_category_title(cname)
                by_category[cname_n].add(title)
                any_cat = True
            if any_cat:
                n_kept += 1

    capped: dict[str, list[str]] = {}
    cap = int(args.max_titles_per_category)
    for cat in sorted(by_category):
        titles = sorted(by_category[cat])
        if cap > 0:
            titles = titles[:cap]
        capped[cat] = titles

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "project": project_filter or None,
        "namespace_id": args.namespace_id,
        "n_records_seen": n_records,
        "n_records_indexed": n_kept,
        "n_categories": len(capped),
        "category_title_index": capped,
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote index: {args.out} "
        f"(categories={len(capped)}, records_seen={n_records}, records_indexed={n_kept})"
    )


if __name__ == "__main__":
    main()
