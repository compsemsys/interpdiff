"""
Wikimedia Enterprise auth helper.

Supports:
1) Username/password login -> access + refresh token
2) Refresh token exchange -> new access token

Optionally writes tokens into a local .env file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

# Load repo-local .env for credentials/tokens if present.
load_dotenv(dotenv_path=Path(__file__).resolve().with_name(".env"), override=False)

DEFAULT_LOGIN_URL = "https://auth.enterprise.wikimedia.com/v1/login"
DEFAULT_REFRESH_URL = "https://auth.enterprise.wikimedia.com/v1/refresh"
DEFAULT_TIMEOUT = 60

ENV_API_KEY = "WIKIMEDIA_API_KEY"
ENV_USERNAME = "WIKIMEDIA_ENTERPRISE_USERNAME"
ENV_PASSWORD = "WIKIMEDIA_ENTERPRISE_PASSWORD"
ENV_REFRESH_TOKEN = "WIKIMEDIA_ENTERPRISE_REFRESH_TOKEN"
ENV_ID_TOKEN = "WIKIMEDIA_ENTERPRISE_ID_TOKEN"
ENV_EXPIRES_IN = "WIKIMEDIA_ENTERPRISE_ACCESS_EXPIRES_IN"
ENV_OBTAINED_AT = "WIKIMEDIA_ENTERPRISE_ACCESS_OBTAINED_AT"


def _token_preview(value: str | None) -> str:
    if not value:
        return "(missing)"
    if len(value) <= 12:
        return "*" * len(value)
    return f"{value[:6]}...{value[-4:]}"


def _post_json(url: str, payload: dict[str, Any], timeout_sec: int) -> dict[str, Any]:
    r = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=timeout_sec,
    )
    if r.status_code >= 400:
        body_preview = (r.text or "").strip()
        if len(body_preview) > 500:
            body_preview = body_preview[:500] + "...(truncated)"
        raise requests.HTTPError(
            f"HTTP {r.status_code} for {url}\nResponse: {body_preview}",
            response=r,
        )
    try:
        return dict(r.json())
    except Exception as e:  # pragma: no cover - defensive parse path
        raise RuntimeError(f"Non-JSON response from {url}: {e}") from e


def login_with_password(
    *,
    username: str,
    password: str,
    login_url: str,
    timeout_sec: int,
) -> dict[str, Any]:
    return _post_json(
        login_url,
        {"username": username, "password": password},
        timeout_sec,
    )


def refresh_access_token(
    *,
    refresh_token: str,
    refresh_url: str,
    timeout_sec: int,
) -> dict[str, Any]:
    return _post_json(
        refresh_url,
        {"refresh_token": refresh_token},
        timeout_sec,
    )


def _write_env_file(path: Path, updates: dict[str, str]) -> None:
    existing_lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    key_to_line_idx: dict[str, int] = {}
    key_re = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=")
    for i, line in enumerate(existing_lines):
        m = key_re.match(line)
        if m:
            key_to_line_idx[m.group(1)] = i

    for key, val in updates.items():
        new_line = f"{key}={val}"
        if key in key_to_line_idx:
            existing_lines[key_to_line_idx[key]] = new_line
        else:
            existing_lines.append(new_line)

    path.write_text("\n".join(existing_lines).rstrip() + "\n", encoding="utf-8")


def _resolve_or_env(cli_value: str | None, env_name: str) -> str | None:
    if cli_value and cli_value.strip():
        return cli_value.strip()
    val = os.getenv(env_name, "").strip()
    return val or None


def _env_updates_from_response(resp: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    access_token = str(resp.get("access_token", "") or "")
    if access_token:
        out[ENV_API_KEY] = access_token
    refresh_token = str(resp.get("refresh_token", "") or "")
    if refresh_token:
        out[ENV_REFRESH_TOKEN] = refresh_token
    id_token = str(resp.get("id_token", "") or "")
    if id_token:
        out[ENV_ID_TOKEN] = id_token
    expires_in = resp.get("expires_in")
    if expires_in is not None:
        out[ENV_EXPIRES_IN] = str(expires_in)
    out[ENV_OBTAINED_AT] = datetime.now(timezone.utc).isoformat()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Wikimedia Enterprise token helper.")
    sub = p.add_subparsers(dest="command", required=True)

    p_login = sub.add_parser("login", help="Get tokens from username/password.")
    p_login.add_argument("--username", default=None, help=f"Defaults to env {ENV_USERNAME}.")
    p_login.add_argument("--password", default=None, help=f"Defaults to env {ENV_PASSWORD}.")
    p_login.add_argument("--login-url", default=DEFAULT_LOGIN_URL)
    p_login.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    p_login.add_argument("--env-file", default=".env")
    p_login.add_argument("--write-env", action="store_true", help="Persist tokens into .env file.")
    p_login.add_argument(
        "--print-json",
        action="store_true",
        help="Print raw JSON response (contains sensitive tokens).",
    )

    p_refresh = sub.add_parser("refresh", help="Get a new access token from refresh token.")
    p_refresh.add_argument(
        "--refresh-token",
        default=None,
        help=f"Defaults to env {ENV_REFRESH_TOKEN}.",
    )
    p_refresh.add_argument("--refresh-url", default=DEFAULT_REFRESH_URL)
    p_refresh.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    p_refresh.add_argument("--env-file", default=".env")
    p_refresh.add_argument("--write-env", action="store_true", help="Persist tokens into .env file.")
    p_refresh.add_argument(
        "--print-json",
        action="store_true",
        help="Print raw JSON response (contains sensitive tokens).",
    )

    args = p.parse_args()

    if args.command == "login":
        username = _resolve_or_env(args.username, ENV_USERNAME)
        password = _resolve_or_env(args.password, ENV_PASSWORD)
        if not username or not password:
            raise SystemExit(
                f"Missing credentials. Set --username/--password or env {ENV_USERNAME}/{ENV_PASSWORD}."
            )
        resp = login_with_password(
            username=username,
            password=password,
            login_url=args.login_url,
            timeout_sec=args.timeout,
        )
    else:
        refresh_token = _resolve_or_env(args.refresh_token, ENV_REFRESH_TOKEN)
        if not refresh_token:
            raise SystemExit(
                f"Missing refresh token. Set --refresh-token or env {ENV_REFRESH_TOKEN}."
            )
        resp = refresh_access_token(
            refresh_token=refresh_token,
            refresh_url=args.refresh_url,
            timeout_sec=args.timeout,
        )

    updates = _env_updates_from_response(resp)
    if args.write_env:
        _write_env_file(Path(args.env_file), updates)
        print(f"Wrote token fields to {args.env_file}")

    if args.print_json:
        print(json.dumps(resp, indent=2))
        return

    print("Token response received.")
    print(f"  access_token: {_token_preview(str(resp.get('access_token', '') or ''))}")
    print(f"  refresh_token: {_token_preview(str(resp.get('refresh_token', '') or ''))}")
    print(f"  id_token: {_token_preview(str(resp.get('id_token', '') or ''))}")
    print(f"  expires_in: {resp.get('expires_in')}")
    if args.write_env:
        print(f"  {ENV_API_KEY} updated in {args.env_file}")


if __name__ == "__main__":
    main()
