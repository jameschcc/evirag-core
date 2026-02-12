# ==============================================================================
# EviRAG — Enterprise-grade High-performance Offline Knowledge Base with
# Traceable Retrieval and Generation Architecture
#         DOI: 10.3969/j.issn.1671-3370.2026.02.032 (S.Chang) https://d.wanfangdata.com.cn/periodical/zgxxj202602032
# ==============================================================================

"""Call OpenCode's OpenAI-compatible inference API with the required headers.

Everything that makes the request acceptable upstream lives in one place:

* ``build_headers()`` — ``Authorization: Bearer <token>``, ``Content-Type``,
  the ``x-opencode-session`` header OpenCode Go rejects requests without
  (``400 MissingSessionID``), and an explicit ``User-Agent`` (Cloudflare
  answers ``403 error code: 1010`` to Python's default urllib agent).
* ``opencode_chat()`` — one function that takes messages, resolves the token
  and posts ``/chat/completions``; every script that needs the endpoint calls
  this instead of re-assembling headers by hand.

Defaults follow the calls DSH makes successfully: base URL
``https://opencode.ai/zen/go/v1``, model ``deepseek-v4.1-flash``, and the
pi-ai User-Agent.

Usage from code::

    from opencode_chat import opencode_chat
    reply = opencode_chat([{"role": "user", "content": "ping"}], model="deepseek-v4.1-flash")
    print(reply["choices"][0]["message"]["content"])

Usage from the shell::

    python backend/scripts/opencode_chat.py "你好" --model deepseek-v4.1-flash

Token lookup order: ``--api-key`` / ``--api-key-file`` -> ``OPENCODE_API_KEY``
-> ``OPENCODE_GO_API_KEY`` -> ``opencode``/``opencode-go`` entry of the local
OpenCode ``auth.json``. The token is never printed.

Notes on failures (verified upstream behaviour, not guesses):
- ``403`` with ``error code: 1010`` comes from Cloudflare's client-signature
  check, before the API sees the request; an explicit User-Agent fixes it.
- ``403`` on free-tier chat models is expected from non-OpenCode clients;
  use a paid model or an OpenCode Go subscription.
- ``402`` means the pay-as-you-go route has no credits; ``429`` with
  ``GoUsageLimitError`` means the Go monthly quota is spent.
- ``400 MissingSessionID`` means the request reached OpenCode Go without the
  ``x-opencode-session`` header; that is exactly what this module prevents.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import platform
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

DEFAULT_BASE_URL = "https://opencode.ai/zen/go/v1"
DEFAULT_MODEL = "deepseek-v4.1-flash"
DEFAULT_TIMEOUT = 120.0

# OpenCode Go routes requests by this header; a stable value per machine keeps
# the routing stable across runs instead of starting a new session each time.
SESSION_HEADER = "x-opencode-session"

# opencode.ai sits behind Cloudflare, which answers `403 error code: 1010` to
# clients whose signature it dislikes — Python's default `Python-urllib/3.x`
# User-Agent is blocked before the API ever sees the request. An explicit
# User-Agent is therefore part of "the required headers", not cosmetics.
# pi-ai (the client DSH uses) sends `pi (<platform> <release>; <arch>)`; this
# module mirrors it, and `--user-agent` / OPENCODE_USER_AGENT can override it.
USER_AGENT_ENV = "OPENCODE_USER_AGENT"
_PLATFORM_NAMES = {"win32": "win32", "cygwin": "win32", "linux": "linux", "darwin": "darwin"}
_ARCH_NAMES = {
    "amd64": "x64",
    "x86_64": "x64",
    "x64": "x64",
    "arm64": "arm64",
    "aarch64": "arm64",
}

_TOKEN_ENV_ORDER = ("OPENCODE_API_KEY", "OPENCODE_GO_API_KEY")
_AUTH_ENTRY_ORDER = ("opencode-go", "opencode")


class OpencodeError(RuntimeError):
    """Raised when the OpenCode API answers with a non-2xx status."""

    def __init__(self, status: int, body: str, hint: str = "") -> None:
        self.status = status
        self.body = body
        self.hint = hint
        parts = [f"OpenCode API returned HTTP {status}"]
        if hint:
            parts.append(hint)
        if body.strip():
            parts.append(body.strip()[:500])
        super().__init__(": ".join(parts))


def default_user_agent() -> str:
    """Mirror pi-ai's User-Agent: ``pi (<platform> <release>; <arch>)``."""
    plat = sys.platform
    name = _PLATFORM_NAMES.get(plat, plat)
    machine = platform.machine()
    arch = _ARCH_NAMES.get(machine.lower(), machine.lower())
    return f"pi ({name} {platform.release()}; {arch})"



def default_session_id() -> str:
    """Return a stable session id for this machine and user."""
    host = socket.gethostname().split(".")[0] or "localhost"
    try:
        user = getpass.getuser()
    except Exception:  # noqa: BLE001 - getuser can fail in odd environments
        user = "user"
    return f"evirag-opencode-chat-{user}-{host}"


def _read_auth_file(path: Path | None = None) -> str | None:
    """Read the OpenCode Go token from OpenCode's own auth file, if present."""
    if path is None:
        override = os.environ.get("OPENCODE_AUTH_FILE")
        path = Path(override) if override else Path.home() / ".local" / "share" / "opencode" / "auth.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    for entry in _AUTH_ENTRY_ORDER:
        value = data.get(entry)
        if isinstance(value, dict):
            key = value.get("key")
            if isinstance(key, str) and key.strip():
                return key.strip()
    return None


def resolve_api_key(api_key: str | None = None, auth_file: Path | None = None) -> str:
    """Resolve the bearer token without ever echoing it."""
    if api_key:
        return api_key.strip()
    for name in _TOKEN_ENV_ORDER:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    token = _read_auth_file(auth_file)
    if token:
        return token
    raise OpencodeError(
        401,
        "",
        "no token found; pass --api-key, set OPENCODE_API_KEY, or log in to OpenCode so "
        f"{Path.home() / '.local' / 'share' / 'opencode' / 'auth.json'} holds an opencode-go key",
    )


def build_headers(
    api_key: str,
    session_id: str | None = None,
    user_agent: str | None = None,
) -> dict[str, str]:
    """Return the full header set required by the OpenCode inference endpoint."""
    if not api_key or not api_key.strip():
        raise OpencodeError(401, "", "empty API token")
    return {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
        SESSION_HEADER: session_id or os.environ.get("OPENCODE_SESSION_ID") or default_session_id(),
        "User-Agent": user_agent or os.environ.get(USER_AGENT_ENV) or default_user_agent(),
    }


def _hint_for_status(status: int, body: str) -> str:
    if status == 401:
        return "token missing, invalid or expired; refresh the OpenCode Go credential"
    if status == 403 and "1010" in body:
        return (
            "Cloudflare rejected the client signature; send an explicit User-Agent "
            "(Python's default urllib User-Agent is blocked with error code 1010)"
        )
    if status == 403:
        return (
            "free-tier chat models are only available from inside OpenCode; "
            "use a paid model or an OpenCode Go subscription from other clients"
        )
    if status == 402:
        return "no OpenCode Models credits for this pay-as-you-go route; top up or use the Go route"
    if status == 429 and "GoUsageLimit" in body:
        return "OpenCode Go monthly quota is exhausted; wait for the reset or enable Extra Usage"
    if status == 400 and "MissingSessionID" in body:
        return f"request reached OpenCode Go without the required {SESSION_HEADER} header"
    return ""


def opencode_chat(
    messages: Sequence[Mapping[str, str]],
    *,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str | None = None,
    session_id: str | None = None,
    user_agent: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    extra_headers: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """POST a chat completion to OpenCode and return the decoded JSON reply."""
    headers = build_headers(resolve_api_key(api_key), session_id, user_agent)
    if extra_headers:
        headers.update({str(k): str(v) for k, v in extra_headers.items()})

    payload: dict[str, Any] = {
        "model": model,
        "messages": [dict(message) for message in messages],
        "stream": False,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    url = base_url.rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - body may already be consumed
            body = ""
        raise OpencodeError(exc.code, body, _hint_for_status(exc.code, body)) from exc
    except urllib.error.URLError as exc:
        raise OpencodeError(0, str(exc.reason), f"cannot reach {url}") from exc

    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise OpencodeError(200, raw[:500], "response is not JSON") from exc
    if not isinstance(data, dict):
        raise OpencodeError(200, raw[:500], "unexpected response shape")
    return data


def _first_message_text(reply: Mapping[str, Any]) -> str:
    choices = reply.get("choices")
    if not isinstance(choices, list) or not choices:
        return json.dumps(reply, ensure_ascii=False, indent=2)
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    return json.dumps(choice, ensure_ascii=False, indent=2)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Call the OpenCode chat completions API with the required headers.")
    parser.add_argument("prompt", nargs="*", help="user prompt; omit to read it from stdin")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=None, help="token; defaults to env or OpenCode auth.json")
    parser.add_argument("--api-key-file", default=None, help="read the token from this file")
    parser.add_argument(
        "--session-id",
        default=None,
        help=f"value for {SESSION_HEADER}; defaults to a stable per-machine id",
    )
    parser.add_argument(
        "--user-agent",
        default=None,
        help="User-Agent header; defaults to the pi-ai style agent (Cloudflare rejects urllib's default)",
    )
    parser.add_argument("--system", default=None, help="optional system prompt")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--json", action="store_true", help="print the raw JSON reply instead of the message text")
    args = parser.parse_args(argv)

    prompt = " ".join(args.prompt).strip() or sys.stdin.read().strip()
    if not prompt:
        parser.error("empty prompt")

    messages: list[MutableMapping[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": prompt})

    api_key = args.api_key
    if not api_key and args.api_key_file:
        api_key = Path(args.api_key_file).read_text(encoding="utf-8").strip()

    try:
        reply = opencode_chat(
            messages,
            model=args.model,
            base_url=args.base_url,
            api_key=api_key,
            session_id=args.session_id,
            user_agent=args.user_agent,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
    except OpencodeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(reply, ensure_ascii=False, indent=2) if args.json else _first_message_text(reply))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
