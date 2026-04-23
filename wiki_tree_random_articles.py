"""
Sample random Wikipedia articles using only categories (and page counts) stored in
`wiki_category_library.json`: the root entry plus its `direct_subcategories` list.
No API tree expansion.

Weights: each category C contributes P = stored `stats.pages`; pick C with probability
P / (sum of P over that pool). Then draw n uniformly from {1, …, P} and take the n-th
mainspace page in stable `categorymembers` order (live API only for listing that
category and for extracts).

Writes JSONL with `wiki_fetch.page_extracts`.

Example:
  python wiki_tree_random_articles.py --per-group 10 --out data/wiki_tree_random.jsonl --seed 42

Key functions:
- ``pages_pool_from_library_entry``: build weighted category pool from stored library stats.
- ``sample_articles_for_root``: weighted sampling loop for article picks per root category.
- ``nth_mainspace_article_in_category``: locate the n-th mainspace page via categorymembers paging.
- ``weighted_random_category``: draw categories proportional to stored page counts.
- ``main``: CLI orchestration from library load through extract fetch and JSONL output.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import requests

from wiki_fetch import (
    DEFAULT_API,
    DEFAULT_UA,
    DEFAULT_WIKIMEDIA_API_KEY_ENV,
    api_get,
    build_wiki_session,
    page_extracts,
    resolve_api_url,
)


def normalize_category_title(title: str) -> str:
    t = title.strip()
    if not t.startswith("Category:"):
        t = f"Category:{t}"
    return t.replace(" ", "_")


def _int_pages(v: Any) -> int:
    if v is None:
        return 0
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def pages_pool_from_library_entry(entry: dict[str, Any]) -> dict[str, int]:
    """Category title -> stored page count (root + direct_subcategories only)."""
    # Sampling pool intentionally stays shallow: root + direct subcategories only.
    pool: dict[str, int] = {}
    rt = entry.get("title")
    if not rt:
        return pool
    root_n = normalize_category_title(rt)
    pool[root_n] = _int_pages(entry.get("stats", {}).get("pages"))
    for sub in entry.get("direct_subcategories") or []:
        st = sub.get("title")
        if not st:
            continue
        key = normalize_category_title(st)
        pool[key] = _int_pages(sub.get("stats", {}).get("pages"))
    return pool


def nth_mainspace_article_in_category(
    session: requests.Session,
    api_url: str,
    category: str,
    n: int,
    delay_sec: float,
) -> str | None:
    """1-indexed n among mainspace (ns=0) pages in category; None if fewer than n."""
    if n < 1:
        return None
    cat = normalize_category_title(category)
    seen = 0
    cmcontinue = None
    while True:
        params: dict[str, Any] = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": cat,
            "cmlimit": "max",
            "cmtype": "page",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        data = api_get(session, api_url, params)
        if delay_sec:
            time.sleep(delay_sec)
        for m in data.get("query", {}).get("categorymembers", []):
            if m.get("ns") != 0:
                continue
            t = m.get("title")
            if not t:
                continue
            seen += 1
            if seen == n:
                return t
        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            return None


def weighted_random_category(
    rng: random.Random,
    pages_by_cat: dict[str, int],
) -> str | None:
    eligible = sorted(
        ((c, p) for c, p in pages_by_cat.items() if p > 0),
        key=lambda x: x[0],
    )
    if not eligible:
        return None
    total = sum(p for _, p in eligible)
    r = rng.randint(1, total)
    acc = 0
    for c, p in eligible:
        acc += p
        if r <= acc:
            return c
    return eligible[-1][0]


def sample_articles_for_root(
    session: requests.Session,
    api_url: str,
    root: str,
    *,
    count: int,
    rng: random.Random,
    pages_by_cat: dict[str, int],
    delay_sec: float,
    max_sample_attempts: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    used_titles: set[str] = set()
    attempts = 0
    # Resample until target count or attempt budget to handle duplicates/stale counts.
    while len(rows) < count and attempts < max_sample_attempts:
        attempts += 1
        cat = weighted_random_category(rng, pages_by_cat)
        if not cat:
            break
        p = pages_by_cat.get(cat, 0)
        if p < 1:
            continue
        n = rng.randint(1, p)
        title = nth_mainspace_article_in_category(session, api_url, cat, n, delay_sec)
        if not title or title in used_titles:
            continue
        used_titles.add(title)
        rows.append(
            {
                "root_category": normalize_category_title(root),
                "picked_category": cat,
                "page_index_in_category": n,
                "pages_count_library": p,
                "title": title,
            }
        )
    return rows


def load_library_entries(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("entries", []))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--library",
        type=Path,
        default=Path(__file__).with_name("wiki_category_library.json"),
        help="wiki_category_library.json (required). Pool = root + direct_subcategories only.",
    )
    p.add_argument(
        "--roots",
        nargs="+",
        default=None,
        metavar="TITLE",
        help='Process only these roots, e.g. "Science" "Culture" (must match library entry titles)',
    )
    p.add_argument("--per-group", type=int, default=10, metavar="N", help="Articles per library entry")
    p.add_argument("--out", type=Path, default=Path("data/wiki_tree_random.jsonl"))
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--delay_sec", type=float, default=0.15, help="Pause between API calls")
    p.add_argument(
        "--api",
        default=None,
        help="API endpoint URL (default: env WIKIMEDIA_API_URL or enwiki action API).",
    )
    p.add_argument("--user_agent", default=DEFAULT_UA)
    p.add_argument(
        "--api_key",
        default=None,
        help=f"Optional API key (default from env {DEFAULT_WIKIMEDIA_API_KEY_ENV}).",
    )
    p.add_argument(
        "--max-sample-attempts",
        type=int,
        default=500,
        help="Upper bound on resamples per root (collisions / stale counts)",
    )
    args = p.parse_args()

    if not args.library.is_file():
        p.error(f"Library not found: {args.library}")

    entries = load_library_entries(args.library)
    if args.roots:
        want = {normalize_category_title(x) for x in args.roots}
        entries = [
            e
            for e in entries
            if e.get("title") and normalize_category_title(e["title"]) in want
        ]
        if not entries:
            p.error(f"No library entries matched --roots {args.roots!r}")

    rng = random.Random(args.seed)
    api_url = resolve_api_url(args.api)
    session = build_wiki_session(args.user_agent, args.api_key)

    all_rows: list[dict[str, Any]] = []
    for e in entries:
        root = e.get("title")
        if not root:
            continue
        root_n = normalize_category_title(root)
        pages_by_cat = pages_pool_from_library_entry(e)
        total_w = sum(pages_by_cat.values())
        pos_w = sum(p for p in pages_by_cat.values() if p > 0)
        n_cats = len(pages_by_cat)
        print(f"{root_n}: pool = root + {max(0, n_cats - 1)} subcats (stored), total pages weight {pos_w} (raw sum {total_w})")
        if pos_w < 1:
            print(f"  skip: no positive page counts in library for this entry")
            continue
        picked = sample_articles_for_root(
            session,
            api_url,
            root_n,
            count=args.per_group,
            rng=rng,
            pages_by_cat=pages_by_cat,
            delay_sec=args.delay_sec,
            max_sample_attempts=args.max_sample_attempts,
        )
        print(f"  sampled articles: {len(picked)} / {args.per_group}")
        all_rows.extend(picked)

    titles = [r["title"] for r in all_rows]
    if not titles:
        print("No articles sampled; nothing written.")
        return
    print("Fetching extracts …")
    extracts, resolved_titles = page_extracts(session, api_url, titles)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    doc_id = 0
    with args.out.open("w", encoding="utf-8") as f:
        for r in all_rows:
            doc_id += 1
            text = extracts.get(r["title"], "").strip()
            if not text:
                text = f"(empty extract for {r['title']})"
            canonical = resolved_titles.get(r["title"], r["title"])
            rec = {
                "doc_id": doc_id,
                "root_category": r["root_category"],
                "picked_category": r["picked_category"],
                "page_index_in_category": r["page_index_in_category"],
                "pages_count_library": r["pages_count_library"],
                "title": r["title"],
                "text": text,
            }
            if canonical != r["title"]:
                rec["resolved_title"] = canonical
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(all_rows)} records to {args.out}")


if __name__ == "__main__":
    main()
