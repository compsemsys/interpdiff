from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path
from typing import Any

# Allow running as "python scripts/expand_wiki_tree_random.py" from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wiki_fetch import (
    DEFAULT_UA,
    DEFAULT_WIKIMEDIA_API_KEY_ENV,
    build_wiki_session,
    category_titles,
    page_extracts,
    resolve_api_url,
)
from wiki_tree_random_articles import (
    load_library_entries,
    normalize_category_title,
    pages_pool_from_library_entry,
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _load_title_pool_cache(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    pools = obj.get("category_title_pools", {})
    if not isinstance(pools, dict):
        return {}
    out: dict[str, list[str]] = {}
    for k, v in pools.items():
        if isinstance(k, str) and isinstance(v, list):
            out[k] = [str(x) for x in v if str(x).strip()]
    return out


def _save_title_pool_cache(path: Path, pools: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"category_title_pools": pools}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_category_index(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    idx = obj.get("category_title_index", {})
    if not isinstance(idx, dict):
        return {}
    out: dict[str, list[str]] = {}
    for k, v in idx.items():
        if isinstance(k, str) and isinstance(v, list):
            out[k] = [str(x) for x in v if str(x).strip()]
    return out


def _validate_library_entries(entries: list[dict[str, Any]], source_path: Path) -> None:
    """
    Validate wiki_category_library.json shape used by this script:
    {"entries":[{"title":..., "stats":{"pages":...}, "direct_subcategories":[...]}]}
    """
    if not isinstance(entries, list) or not entries:
        raise SystemExit(f"{source_path}: expected non-empty 'entries' list.")
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise SystemExit(f"{source_path}: entries[{i}] must be an object.")
        title = entry.get("title")
        if not isinstance(title, str) or not title.strip():
            raise SystemExit(f"{source_path}: entries[{i}].title must be a non-empty string.")
        dsubs = entry.get("direct_subcategories", [])
        if dsubs is None:
            continue
        if not isinstance(dsubs, list):
            raise SystemExit(f"{source_path}: entries[{i}].direct_subcategories must be a list.")
        for j, sub in enumerate(dsubs):
            if not isinstance(sub, dict):
                raise SystemExit(
                    f"{source_path}: entries[{i}].direct_subcategories[{j}] must be an object."
                )
            sub_title = sub.get("title")
            if not isinstance(sub_title, str) or not sub_title.strip():
                raise SystemExit(
                    f"{source_path}: entries[{i}].direct_subcategories[{j}].title "
                    "must be a non-empty string."
                )


def _weighted_sample_without_replacement(
    items: list[tuple[str, int]],
    k: int,
    rng: random.Random,
) -> list[tuple[str, int]]:
    """
    Weighted sample without replacement using Efraimidis-Spirakis keys.
    """
    if k <= 0 or not items:
        return []
    scored: list[tuple[float, str, int]] = []
    for cat, weight in items:
        w = max(0, int(weight))
        if w <= 0:
            continue
        u = max(rng.random(), 1e-12)
        key = u ** (1.0 / float(w))
        scored.append((key, cat, w))
    scored.sort(reverse=True)
    return [(cat, w) for _key, cat, w in scored[: min(k, len(scored))]]


def _allocate_quota(total: int, weights: list[int]) -> list[int]:
    if total <= 0 or not weights:
        return [0] * len(weights)
    s = float(sum(max(0, w) for w in weights))
    if s <= 0:
        return [0] * len(weights)
    raw = [total * (max(0, w) / s) for w in weights]
    base = [int(x) for x in raw]
    remain = total - sum(base)
    frac_idx = sorted(
        range(len(raw)),
        key=lambda i: (raw[i] - base[i]),
        reverse=True,
    )
    for i in frac_idx[:remain]:
        base[i] += 1
    return base


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Append additional random wiki docs per root category using "
            "wiki_category_library root + direct subcategories."
        )
    )
    p.add_argument("--jsonl", type=Path, default=Path("data/wiki_tree_random.jsonl"))
    p.add_argument("--library", type=Path, default=Path("wiki_category_library.json"))
    p.add_argument(
        "--title-pool-cache",
        type=Path,
        default=Path("data/wiki_category_title_pool_cache.json"),
        help="Local cache of category->title pools to reduce repeated API list calls.",
    )
    p.add_argument(
        "--category-index",
        type=Path,
        default=Path("data/wiki_snapshot_category_index.json"),
        help="Local category->title index built from snapshot data.",
    )
    p.add_argument(
        "--use-category-index",
        action="store_true",
        help="Use local snapshot category index first (falls back to API list calls).",
    )
    p.add_argument(
        "--index-only",
        action="store_true",
        help="Require local category index only (no API category list fallbacks).",
    )
    p.add_argument(
        "--no-cache-read",
        action="store_true",
        help="Ignore existing title pool cache when sampling.",
    )
    p.add_argument(
        "--no-cache-write",
        action="store_true",
        help="Do not write updated title pools to cache after sampling.",
    )
    p.add_argument("--add-per-root", type=int, default=90)
    p.add_argument("--seed", type=int, default=20260423)
    p.add_argument("--delay-sec", type=float, default=0.10)
    p.add_argument(
        "--max-categories-per-root",
        type=int,
        default=24,
        help="Bound category list calls per root (lower = fewer API calls).",
    )
    p.add_argument(
        "--oversample-ratio",
        type=float,
        default=1.5,
        help="Collect this multiple of N candidate titles before final local sampling.",
    )
    p.add_argument(
        "--pool-multiplier",
        type=float,
        default=4.0,
        help="Per-category member pool ~= quota * multiplier (single category list call).",
    )
    p.add_argument(
        "--pool-min",
        type=int,
        default=30,
        help="Minimum per-category member_pool size when listing titles.",
    )
    p.add_argument(
        "--pool-max",
        type=int,
        default=500,
        help="Maximum per-category member_pool size for one list call.",
    )
    p.add_argument(
        "--api",
        default=None,
        help="API endpoint URL (default: env WIKIMEDIA_API_URL or enwiki action API).",
    )
    p.add_argument(
        "--user-agent",
        default=DEFAULT_UA,
        help="User-Agent header value.",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help=f"Optional API key (default from env {DEFAULT_WIKIMEDIA_API_KEY_ENV}).",
    )
    args = p.parse_args()

    if not args.jsonl.is_file():
        raise SystemExit(f"Missing JSONL: {args.jsonl}")
    if not args.library.is_file():
        raise SystemExit(f"Missing library: {args.library}")

    rows = _load_jsonl(args.jsonl)
    roots = sorted(
        {
            normalize_category_title(r.get("root_category", ""))
            for r in rows
            if r.get("root_category")
        }
    )
    if not roots:
        raise SystemExit("No root categories found in input JSONL.")

    existing_titles_by_root: dict[str, set[str]] = {rt: set() for rt in roots}
    for r in rows:
        rt = normalize_category_title(r.get("root_category", ""))
        t = r.get("title")
        if rt in existing_titles_by_root and t:
            existing_titles_by_root[rt].add(t)

    entries = load_library_entries(args.library)
    _validate_library_entries(entries, args.library)
    entry_by_root = {
        normalize_category_title(e.get("title", "")): e for e in entries if e.get("title")
    }

    api_url = resolve_api_url(args.api)
    rng = random.Random(args.seed)
    session = build_wiki_session(args.user_agent, args.api_key)
    title_pool_cache = (
        {}
        if args.no_cache_read
        else _load_title_pool_cache(args.title_pool_cache)
    )
    category_index = (
        _load_category_index(args.category_index) if args.use_category_index or args.index_only else {}
    )
    if args.index_only and not category_index:
        raise SystemExit(
            f"--index-only was set, but index is missing/empty: {args.category_index}"
        )

    new_rows: list[dict[str, Any]] = []
    for rt in roots:
        e = entry_by_root.get(rt)
        if not e:
            raise SystemExit(f"No matching library entry for root {rt}")
        pages_by_cat = pages_pool_from_library_entry(e)  # includes direct subcategories
        used_titles = set(existing_titles_by_root.get(rt, set()))
        target = int(args.add_per_root)
        need_candidates = max(target, int(round(target * float(args.oversample_ratio))))

        eligible = [(cat, int(p)) for cat, p in pages_by_cat.items() if int(p) > 0]
        if not eligible:
            raise SystemExit(f"No eligible categories for root {rt}")

        selected = _weighted_sample_without_replacement(
            eligible,
            k=min(int(args.max_categories_per_root), len(eligible)),
            rng=rng,
        )
        if not selected:
            raise SystemExit(f"Failed to select categories for root {rt}")

        quotas = _allocate_quota(need_candidates, [w for _c, w in selected])
        candidate_rows: list[dict[str, Any]] = []
        category_api_calls = 0

        for (cat, weight), quota in zip(selected, quotas):
            if quota <= 0:
                continue
            pool_limit = int(
                max(
                    int(args.pool_min),
                    min(
                        int(args.pool_max),
                        round(quota * float(args.pool_multiplier)),
                    ),
                )
            )
            cached = title_pool_cache.get(cat, [])
            indexed = category_index.get(cat, [])
            if indexed:
                title_pool = indexed[:pool_limit]
            elif len(cached) >= pool_limit:
                title_pool = cached[:pool_limit]
            else:
                if args.index_only:
                    title_pool = []
                    continue
                title_pool = category_titles(session, api_url, cat, pool_limit)
                title_pool_cache[cat] = list(title_pool)
                category_api_calls += 1
                if args.delay_sec:
                    import time

                    time.sleep(args.delay_sec)

            filtered = [t for t in title_pool if t and t not in used_titles]
            if not filtered:
                continue
            take = min(quota, len(filtered))
            chosen = rng.sample(filtered, take) if len(filtered) > take else filtered
            for title in chosen:
                used_titles.add(title)
                candidate_rows.append(
                    {
                        "root_category": normalize_category_title(rt),
                        "picked_category": cat,
                        "page_index_in_category": None,
                        "pages_count_library": weight,
                        "title": title,
                    }
                )

        if len(candidate_rows) < target:
            raise SystemExit(
                f"Could not gather enough unique candidates for {rt}: "
                f"{len(candidate_rows)}/{target}. "
                f"Try raising --max-categories-per-root or --pool-max."
            )

        gathered = (
            rng.sample(candidate_rows, target)
            if len(candidate_rows) > target
            else candidate_rows
        )

        print(
            f"{rt}: selected {len(selected)} categories, "
            f"category_list_calls={category_api_calls}, "
            f"candidates={len(candidate_rows)}, final={len(gathered)}"
        )

        titles = [g["title"] for g in gathered]
        extracts, resolved_titles = page_extracts(session, api_url, titles)
        for g in gathered:
            rec = dict(g)
            text = (extracts.get(g["title"]) or "").strip()
            rec["text"] = text if text else f"(empty extract for {g['title']})"
            canonical = resolved_titles.get(g["title"], g["title"])
            if canonical != g["title"]:
                rec["resolved_title"] = canonical
            new_rows.append(rec)

    max_doc_id = max((int(r.get("doc_id", 0)) for r in rows), default=0)
    for i, r in enumerate(new_rows, 1):
        r["doc_id"] = max_doc_id + i

    with args.jsonl.open("a", encoding="utf-8") as f:
        for r in new_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if not args.no_cache_write:
        _save_title_pool_cache(args.title_pool_cache, title_pool_cache)
        print(f"Updated title pool cache: {args.title_pool_cache}")

    print(f"Appended {len(new_rows)} rows to {args.jsonl}")


if __name__ == "__main__":
    main()
