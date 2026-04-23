"""
Fetch Wikipedia articles by category via the MediaWiki Action API and save JSONL
compatible with run_categorized_corpus.py.

Pipeline (no separate "probe" queries — only what you need for the corpus):

1. **List titles** — `list=categorymembers` (batched, up to 500 per HTTP call on enwiki).
2. **Choose locally** — optional oversample + `random.sample` (no API; default: take
   the first `per_category` titles in API order).
3. **Fetch text** — `prop=extracts` for the chosen titles (batched, ~20 per call; redirects resolved).

Example:
  python wiki_fetch.py --categories "Category:Machine_learning" "Category:Statistics" \\
    --per_category 10 --out data/corpus_wiki.jsonl

  # Fetch up to 300 names in few list calls, pick 100 at random, then extract those only:
  python wiki_fetch.py --categories Science --member_pool 300 --per_category 100 \\
    --seed 1 --out data/wiki_science.jsonl

Uses requests; set a descriptive User-Agent (WMF policy).

Key functions:
- ``category_titles``: list mainspace titles from categorymembers with continuation.
- ``page_extracts``: fetch intro extracts in batches and map redirected titles back.
- ``api_get``: retrying MediaWiki request wrapper with backoff/logging.
- ``build_wiki_session``: build requests session with user-agent and optional API key headers.
- ``main``: CLI workflow for category sampling and JSONL corpus writing.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

# en.wikipedia.org allows up to 500 category members per categorymembers call.
# TextExtracts: multi-title requests only return one *full-article* extract unless
# exintro=1; with exintro, enwiki caps at ~20 extracts per request (see limits.extracts).
_CM_LIMIT = 500
_EXTRACT_BATCH = 20

DEFAULT_API = "https://en.wikipedia.org/w/api.php"
DEFAULT_UA = "IndependentStudyBot/0.1 (local research; python requests)"
DEFAULT_WIKIMEDIA_API_KEY_ENV = "WIKIMEDIA_API_KEY"
DEFAULT_WIKIMEDIA_API_URL_ENV = "WIKIMEDIA_API_URL"
_API_CALL_SEQ = 0

# Auto-load repo-local .env for API settings when available.
load_dotenv(dotenv_path=Path(__file__).resolve().with_name(".env"), override=False)


def resolve_api_url(cli_api_url: str | None = None) -> str:
    if cli_api_url and str(cli_api_url).strip():
        return str(cli_api_url).strip()
    env_api = os.getenv(DEFAULT_WIKIMEDIA_API_URL_ENV, "").strip()
    if env_api:
        return env_api
    return DEFAULT_API


def resolve_api_key(cli_api_key: str | None = None) -> str | None:
    if cli_api_key and str(cli_api_key).strip():
        return str(cli_api_key).strip()
    env_key = os.getenv(DEFAULT_WIKIMEDIA_API_KEY_ENV, "").strip()
    return env_key or None


def build_wiki_session(user_agent: str, api_key: str | None = None) -> requests.Session:
    session = requests.Session()
    headers = {"User-Agent": user_agent}
    key = resolve_api_key(api_key)
    if key:
        # Some gateways expect Bearer auth, some accept explicit API key header.
        headers["Authorization"] = f"Bearer {key}"
        headers["Api-Key"] = key
    session.headers.update(headers)
    return session


def _api_params_summary(params: dict[str, Any]) -> str:
    parts = []
    for k in ("action", "list", "prop", "generator", "cmtitle", "gcmtitle"):
        v = params.get(k)
        if v:
            parts.append(f"{k}={v}")
    titles = params.get("titles")
    if isinstance(titles, str) and titles.strip():
        parts.append(f"titles_count={len([t for t in titles.split('|') if t])}")
    cmcontinue = params.get("cmcontinue") or params.get("gcmcontinue")
    if cmcontinue:
        parts.append("continued=1")
    return ", ".join(parts) if parts else "no-params"


def api_get(session: requests.Session, api_url: str, params: dict[str, Any]) -> dict:
    global _API_CALL_SEQ
    backoff = 1.0
    last_err: Exception | None = None
    for attempt in range(1, 8):
        _API_CALL_SEQ += 1
        call_id = _API_CALL_SEQ
        t0 = time.perf_counter()
        try:
            r = session.get(api_url, params=params, timeout=60)
            dt_ms = int((time.perf_counter() - t0) * 1000)
            print(
                f"[wiki_api] call={call_id} attempt={attempt} status={r.status_code} "
                f"duration_ms={dt_ms} {_api_params_summary(params)}"
            )
            if r.status_code in (429, 503):
                last_err = requests.HTTPError(
                    f"{r.status_code} from API for params={params!r}", response=r
                )
                retry_after = r.headers.get("Retry-After")
                if retry_after:
                    try:
                        sleep_s = max(float(retry_after), backoff)
                    except ValueError:
                        sleep_s = backoff
                else:
                    sleep_s = backoff
                time.sleep(sleep_s)
                backoff = min(backoff * 2.0, 60.0)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last_err = e
            dt_ms = int((time.perf_counter() - t0) * 1000)
            print(
                f"[wiki_api] call={call_id} attempt={attempt} error={type(e).__name__} "
                f"duration_ms={dt_ms} {_api_params_summary(params)}"
            )
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 60.0)
    if last_err is not None:
        raise last_err
    raise RuntimeError("api_get failed without an exception")


def category_titles(
    session: requests.Session,
    api_url: str,
    category: str,
    limit: int,
) -> list[str]:
    titles: list[str] = []
    cmcontinue = None
    # Walk categorymembers continuation until we hit limit or exhaust the category.
    while len(titles) < limit:
        params: dict[str, Any] = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": category if category.startswith("Category:") else f"Category:{category}",
            "cmlimit": min(_CM_LIMIT, limit - len(titles)),
            "cmtype": "page",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        data = api_get(session, api_url, params)
        q = data.get("query", {})
        members = q.get("categorymembers", [])
        for m in members:
            t = m.get("title")
            if t and m.get("ns") == 0:
                titles.append(t)
                if len(titles) >= limit:
                    break
        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break
    return titles[:limit]


def _titles_equal(a: str, b: str) -> bool:
    return a == b or a.replace("_", " ") == b.replace("_", " ")


def _resolve_to_canonical_title(
    title: str,
    normalized: list[Any],
    redirects: list[Any],
) -> str:
    """Apply query.normalized then follow query.redirects to the article title keys use in pages."""
    t = title
    for item in normalized or []:
        if not isinstance(item, dict):
            continue
        fr, to = item.get("from"), item.get("to")
        if fr and to and _titles_equal(fr, t):
            t = to
    mp = {
        r["from"]: r["to"]
        for r in (redirects or [])
        if isinstance(r, dict) and r.get("from") and r.get("to")
    }
    seen: set[str] = set()
    while t in mp and t not in seen:
        seen.add(t)
        t = mp[t]
    return t


def page_extracts(
    session: requests.Session,
    api_url: str,
    titles: list[str],
) -> tuple[dict[str, str], dict[str, str]]:
    """For each requested title, lead-section extract after redirects.

    Returns (extract_by_requested_title, canonical_title_by_requested_title).
    With ``redirects=1``, the API keys extracts by the target page; we map back so
    ``out["G.I. Generation"]`` matches the extract for ``Greatest Generation``.
    """
    out: dict[str, str] = {}
    resolved: dict[str, str] = {}
    for i in range(0, len(titles), _EXTRACT_BATCH):
        # Batch title lookups to stay within extract API limits and avoid huge requests.
        batch = titles[i : i + _EXTRACT_BATCH]
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "explaintext": 1,
            "exintro": 1,
            "exsectionformat": "plain",
            "exlimit": "max",
            "redirects": 1,
            "titles": "|".join(batch),
        }
        data = api_get(session, api_url, params)
        q = data.get("query", {})
        pages = q.get("pages", {})
        extract_by_canonical: dict[str, str] = {}
        for _pid, page in pages.items():
            t = page.get("title")
            if not t:
                continue
            extract_by_canonical[t] = page.get("extract", "") or ""
        normalized = q.get("normalized") or []
        redirects = q.get("redirects") or []
        for orig in batch:
            canon = _resolve_to_canonical_title(orig, normalized, redirects)
            resolved[orig] = canon
            out[orig] = extract_by_canonical.get(canon, "") or ""
    return out, resolved


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--categories",
        nargs="+",
        required=True,
        help='e.g. "Machine_learning" or "Category:Physics" (Category: prefix optional)',
    )
    p.add_argument("--per_category", type=int, default=10)
    p.add_argument(
        "--member_pool",
        type=int,
        default=None,
        metavar="N",
        help="List at most N category member titles (few batched API calls), then pick "
        "--per_category locally. Default: same as --per_category (no oversampling).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed when member_pool > per_category (random.sample).",
    )
    p.add_argument("--out", default="data/corpus_wiki.jsonl")
    p.add_argument(
        "--api",
        default=None,
        help=(
            "API endpoint URL. Default: --api, else env WIKIMEDIA_API_URL, "
            "else https://en.wikipedia.org/w/api.php."
        ),
    )
    p.add_argument("--user_agent", default=DEFAULT_UA)
    p.add_argument(
        "--api_key",
        default=None,
        help=f"Optional API key (default from env {DEFAULT_WIKIMEDIA_API_KEY_ENV}).",
    )
    p.add_argument("--delay_sec", type=float, default=0.5, help="Pause between API calls")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    api_url = resolve_api_url(args.api)
    session = build_wiki_session(args.user_agent, args.api_key)

    doc_id = 0
    rows: list[dict[str, Any]] = []

    rng = random.Random(args.seed)

    for cat in args.categories:
        pool_limit = args.member_pool if args.member_pool is not None else args.per_category
        pool_limit = max(pool_limit, args.per_category)
        title_pool = category_titles(session, api_url, cat, pool_limit)
        time.sleep(args.delay_sec)
        if not title_pool:
            print(f"Warning: no titles for {cat}")
            continue
        if len(title_pool) > args.per_category:
            titles = rng.sample(title_pool, args.per_category)
        else:
            titles = title_pool
        extracts, _resolved = page_extracts(session, api_url, titles)
        time.sleep(args.delay_sec)
        cat_label = cat if cat.startswith("Category:") else f"Category:{cat}"
        for title in titles:
            doc_id += 1
            text = extracts.get(title, "").strip()
            if not text:
                text = f"(empty extract for {title})"
            rows.append(
                {
                    "doc_id": doc_id,
                    "category": cat_label,
                    "title": title,
                    "text": text,
                }
            )

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} records to {args.out}")


if __name__ == "__main__":
    main()
