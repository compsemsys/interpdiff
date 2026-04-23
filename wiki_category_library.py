"""
Maintain wiki_category_library.json using the MediaWiki API.

For each library category we store:
  - Header counts via prop=categoryinfo (same numbers as the Wikipedia category box).
  - Every direct subcategory: name + counts via generator=categorymembers + prop=categoryinfo
    in the same HTTP response (one call per parent if ≤500 subcats; otherwise continue).

Example:
  python wiki_category_library.py --refresh
  python wiki_category_library.py --refresh --no-subcategories

Key functions:
- ``refresh_library``: refresh each library entry's stats and optional direct subcategories.
- ``fetch_categoryinfo``: batched categoryinfo fetch for root categories.
- ``fetch_direct_subcategories_with_info``: list direct subcategories + stats via generator query.
- ``main``: CLI entrypoint for refresh options and API configuration.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from wiki_fetch import (
    DEFAULT_API,
    DEFAULT_UA,
    DEFAULT_WIKIMEDIA_API_KEY_ENV,
    api_get,
    build_wiki_session,
    resolve_api_url,
)

_CI_BATCH = 50


def normalize_category_title(title: str) -> str:
    t = title.strip()
    if not t.startswith("Category:"):
        t = f"Category:{t}"
    return t.replace(" ", "_")


def fetch_categoryinfo(
    session: requests.Session,
    api_url: str,
    titles: list[str],
) -> dict[str, dict[str, Any]]:
    """title -> categoryinfo dict (may be empty if invalid)."""
    out: dict[str, dict[str, Any]] = {}
    for i in range(0, len(titles), _CI_BATCH):
        batch = titles[i : i + _CI_BATCH]
        params = {
            "action": "query",
            "format": "json",
            "prop": "categoryinfo",
            "titles": "|".join(batch),
        }
        data = api_get(session, api_url, params)
        for _pid, page in data.get("query", {}).get("pages", {}).items():
            t = page.get("title")
            if not t:
                continue
            ci = page.get("categoryinfo")
            if isinstance(ci, dict):
                out[t] = dict(ci)
    return out


def _stats_from_ci(ci: dict[str, Any] | None, updated_iso: str) -> dict[str, Any]:
    if not ci:
        return {"updated_iso": updated_iso}
    return {
        "size": ci.get("size"),
        "pages": ci.get("pages"),
        "subcats": ci.get("subcats"),
        "files": ci.get("files"),
        "updated_iso": updated_iso,
    }


def fetch_direct_subcategories_with_info(
    session: requests.Session,
    api_url: str,
    parent_category: str,
    *,
    max_subcategories: int | None,
    updated_iso: str,
) -> list[dict[str, Any]]:
    """
    Direct subcategories with categoryinfo in one query pattern:
    generator=categorymembers (gcmtype=subcat) + prop=categoryinfo.
    """
    cat = normalize_category_title(parent_category)
    gcmcontinue = None
    rows: list[dict[str, Any]] = []

    while max_subcategories is None or len(rows) < max_subcategories:
        if max_subcategories is None:
            gcmlimit: str | int = "max"
        else:
            need = max_subcategories - len(rows)
            gcmlimit = min(500, need) if need < 500 else "max"
        params: dict[str, Any] = {
            "action": "query",
            "format": "json",
            "generator": "categorymembers",
            "gcmtitle": cat,
            "gcmtype": "subcat",
            "gcmlimit": gcmlimit,
            "prop": "categoryinfo",
        }
        if gcmcontinue:
            params["gcmcontinue"] = gcmcontinue
        data = api_get(session, api_url, params)
        pages = data.get("query", {}).get("pages", {})
        for _pid, page in pages.items():
            t = page.get("title")
            if not t or page.get("ns") != 14:
                continue
            ci = page.get("categoryinfo")
            child_ci = dict(ci) if isinstance(ci, dict) else None
            rows.append({"title": t, "stats": _stats_from_ci(child_ci, updated_iso)})
            if max_subcategories is not None and len(rows) >= max_subcategories:
                break

        if max_subcategories is not None and len(rows) >= max_subcategories:
            break
        gcmcontinue = data.get("continue", {}).get("gcmcontinue")
        if not gcmcontinue:
            break

    rows.sort(key=lambda r: r["title"])
    if max_subcategories is not None and len(rows) > max_subcategories:
        rows = rows[:max_subcategories]
    return rows


def refresh_library(
    path: Path,
    api_url: str,
    user_agent: str,
    *,
    api_key: str | None,
    include_subcategories: bool,
    max_subcategories: int | None,
) -> None:
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries: list[dict[str, Any]] = raw.get("entries", [])
    titles = [normalize_category_title(e["title"]) for e in entries if e.get("title")]
    if not titles:
        print("No entries with titles; nothing to refresh.")
        return

    session = build_wiki_session(user_agent, api_key)
    info_by_title = fetch_categoryinfo(session, api_url, titles)
    now = datetime.now(timezone.utc).isoformat()

    # Refresh each root category and optionally replace its direct-subcategory snapshot.
    for e in entries:
        t = normalize_category_title(e.get("title", ""))
        if not t:
            continue
        ci = info_by_title.get(t)
        st = e.setdefault("stats", {})
        if not ci:
            print(f"Warning: no categoryinfo for {t} (missing or invalid title?)")
            st["updated_iso"] = now
            continue
        st["size"] = ci.get("size")
        st["pages"] = ci.get("pages")
        st["subcats"] = ci.get("subcats")
        st["files"] = ci.get("files")
        st["updated_iso"] = now

        want_sub = include_subcategories and e.get("include_subcategories", True) is not False
        if not want_sub:
            continue

        e["direct_subcategories"] = fetch_direct_subcategories_with_info(
            session,
            api_url,
            t,
            max_subcategories=max_subcategories,
            updated_iso=now,
        )

    path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Updated {len(entries)} entries in {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--library",
        type=Path,
        default=Path(__file__).with_name("wiki_category_library.json"),
        help="Path to wiki_category_library.json",
    )
    p.add_argument("--refresh", action="store_true", help="Fetch categoryinfo and rewrite library")
    p.add_argument(
        "--no-subcategories",
        action="store_true",
        help="Do not list direct subcategories or fetch their stats",
    )
    p.add_argument(
        "--max-subcategories",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N direct subcategories per entry (default: all)",
    )
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
    args = p.parse_args()
    api_url = resolve_api_url(args.api)

    if args.refresh:
        refresh_library(
            args.library,
            api_url,
            args.user_agent,
            api_key=args.api_key,
            include_subcategories=not args.no_subcategories,
            max_subcategories=args.max_subcategories,
        )
    else:
        p.print_help()


if __name__ == "__main__":
    main()
