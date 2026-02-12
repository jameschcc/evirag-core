# ==============================================================================
# EviRAG — Enterprise-grade High-performance Offline Knowledge Base with
# Traceable Retrieval and Generation Architecture
#         DOI: 10.3969/j.issn.1671-3370.2026.02.032 (S.Chang) https://d.wanfangdata.com.cn/periodical/zgxxj202602032
# ==============================================================================

from __future__ import annotations

import json
import logging
import os
import re
import socket
from typing import Any
from urllib.parse import urlsplit

import httpx
from pydantic import ValidationError


logger = logging.getLogger(__name__)

_STRUCTURED_MODES = {"tool", "json"}
_TOOL_DESCRIPTION = (
    "Return the structured result for the current request. "
    "Call this function exactly once and put the complete result in its arguments."
)
# HTTP statuses that mean the provider rejected the request shape (for example
# tools or strict schemas are unsupported), not that the model failed.
SHAPE_REJECTED_STATUSES = {400, 404, 405, 415, 422}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _tool_name(schema: dict[str, Any]) -> str:
    title = str(schema.get("title") or "result")
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", title).strip("_").lower() or "result"
    return f"submit_{slug}"[:64]


class StructuredChatClient:
    """Small protocol adapter for Ollama and OpenAI-compatible chat endpoints.

    ``structured`` selects how the JSON contract reaches the model:

    * ``tool``      — OpenAI-style function tool with a JSON Schema and
      ``strict`` enabled where the provider supports it (DeepSeek does, as a
      beta), which constrains the arguments instead of only asking for JSON.
    * ``tool_loose``— same tool without ``strict`` (fallback when a provider
      rejects the strict flag).
    * ``json``      — legacy ``response_format: json_object`` (no schema).
    * ``format``    — Ollama native grammar-constrained output.
    """

    def __init__(
        self,
        provider: str,
        base_url: str,
        api_key: str,
        model: str,
        structured_mode: str | None = None,
        tool_choice: str | None = None,
        tool_strict: bool | None = None,
    ) -> None:
        self.provider = provider.strip().lower()
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key.strip()
        self.model = model
        mode = (structured_mode or os.getenv("KB_STRUCTURED_MODE") or "tool").strip().lower()
        self.structured_mode = mode if mode in _STRUCTURED_MODES else "tool"
        self.tool_choice = (tool_choice or os.getenv("KB_TOOL_CHOICE") or "auto").strip().lower()
        if self.tool_choice not in {"auto", "required"}:
            self.tool_choice = "auto"
        self.tool_strict = _env_bool("KB_TOOL_STRICT", True) if tool_strict is None else tool_strict

    @property
    def available(self) -> bool:
        return bool(
            self.base_url
            and self.model
            and self.provider in {"ollama", "openai", "openai-compatible", "openai_compatible"}
        )

    @property
    def is_openai_style(self) -> bool:
        return self.provider in {"openai", "openai-compatible", "openai_compatible"}

    @property
    def is_opencode(self) -> bool:
        """True when the endpoint is OpenCode's inference or Go route."""
        host = (urlsplit(self.base_url).hostname or "").lower()
        return host == "opencode.ai" or host.endswith(".opencode.ai")

    def _opencode_session_id(self) -> str:
        override = os.getenv("KB_OPENCODE_SESSION_ID")
        if override and override.strip():
            return override.strip()
        host = socket.gethostname().split(".")[0] or "evirag"
        return f"evirag-{host}"

    @property
    def endpoint(self) -> str:
        if self.provider == "ollama":
            return f"{self.base_url}/api/chat"
        if self.is_openai_style:
            return f"{self.base_url}/chat/completions"
        raise ValueError(f"不支持的 AI Provider：{self.provider}")

    @property
    def headers(self) -> dict[str, str]:
        if self.api_key and self.provider != "ollama":
            headers = {"Authorization": f"Bearer {self.api_key}"}
        else:
            headers = {}
        if self.is_opencode:
            # OpenCode Go answers 400 MissingSessionID without this routing
            # header; inference/pay-as-you-go routes accept it too.
            headers["x-opencode-session"] = self._opencode_session_id()
        return headers

    def effective_max_tokens(self, requested: int | None) -> int | None:
        """Leave DeepSeek output sizing to the provider, including thinking calls."""
        if self.is_openai_style and self.model.strip().lower().startswith("deepseek"):
            return None
        return requested

    def structured_variants(self) -> list[str]:
        """Ordered request shapes to try for one structured call."""
        if self.provider == "ollama":
            return ["format"]
        if self.structured_mode == "json":
            return ["json"]
        return ["tool", "tool_loose", "json"] if self.tool_strict else ["tool_loose", "json"]

    def payload(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        temperature: float,
        max_tokens: int | None,
        stream: bool = False,
        structured: str | None = None,
    ) -> dict[str, Any]:
        shape = structured or ("format" if self.provider == "ollama" else (
            "tool" if self.structured_mode == "tool" else "json"
        ))
        if self.provider == "ollama":
            options = {"temperature": temperature}
            if max_tokens is not None:
                options["num_predict"] = max_tokens
            return {
                "model": self.model,
                "stream": stream,
                "think": False,
                "format": schema,
                "options": options,
                "messages": messages,
            }
        if self.is_openai_style:
            payload: dict[str, Any] = {
                "model": self.model,
                "stream": stream,
                "temperature": temperature,
                "messages": messages,
            }
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
            if shape in {"tool", "tool_loose"}:
                function: dict[str, Any] = {
                    "name": _tool_name(schema),
                    "description": _TOOL_DESCRIPTION,
                    "parameters": schema,
                }
                if shape == "tool" and self.tool_strict:
                    function["strict"] = True
                payload["tools"] = [{"type": "function", "function": function}]
                payload["tool_choice"] = self.tool_choice
                if (
                    self.tool_choice != "auto"
                    and self.model.strip().lower().startswith("deepseek")
                ):
                    # DeepSeek rejects forced tool choice while thinking is on.
                    payload["thinking"] = {"type": "disabled"}
            else:
                payload["response_format"] = {"type": "json_object"}
            return payload
        raise ValueError(f"不支持的 AI Provider：{self.provider}")

    async def post(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        temperature: float,
        max_tokens: int | None,
        timeout: float,
        stream: bool = False,
    ) -> httpx.Response:
        """POST one structured call, degrading the request shape on rejection."""
        variants = self.structured_variants()
        last_error: httpx.HTTPError | None = None
        for index, shape in enumerate(variants):
            payload = self.payload(messages, schema, temperature, max_tokens, stream=stream, structured=shape)
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(self.endpoint, json=payload, headers=self.headers)
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as exc:
                last_error = exc
                if exc.response.status_code in SHAPE_REJECTED_STATUSES and index < len(variants) - 1:
                    logger.warning(
                        "structured request shape %s rejected with HTTP %s; trying %s",
                        shape,
                        exc.response.status_code,
                        variants[index + 1],
                    )
                    continue
                raise
        raise last_error or RuntimeError("structured request failed")

    def extract_content(self, body: dict[str, Any]) -> str:
        if self.provider == "ollama":
            message = body.get("message") or {}
            arguments = message.get("tool_calls")
            if isinstance(arguments, list) and arguments:
                return _tool_arguments(arguments[0])
            content = message.get("content", "")
        elif self.is_openai_style:
            choices = body.get("choices")
            message = choices[0].get("message", {}) if choices else {}
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                return _tool_arguments(tool_calls[0])
            content = message.get("content", "")
        else:
            raise ValueError(f"不支持的 AI Provider：{self.provider}")
        if isinstance(content, list):
            return "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return content if isinstance(content, str) else ""


def _tool_arguments(tool_call: object) -> str:
    """Return a tool call's arguments as JSON text, regardless of provider shape."""
    if not isinstance(tool_call, dict):
        return ""
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return ""
    arguments = function.get("arguments")
    if isinstance(arguments, dict):
        try:
            return json.dumps(arguments, ensure_ascii=False)
        except (TypeError, ValueError):
            return ""
    return arguments if isinstance(arguments, str) else ""


def safe_chat_failure_reason(exc: Exception) -> str:
    """Return a useful failure category without exposing URLs, bodies, or secrets."""
    if isinstance(exc, httpx.HTTPStatusError):
        return f"上游返回 HTTP {exc.response.status_code}"
    if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
        return "上游请求超时"
    if isinstance(exc, httpx.HTTPError):
        return "上游网络请求失败"
    if isinstance(exc, json.JSONDecodeError):
        return "模型返回空内容或无效 JSON" if not exc.doc.strip() else "模型返回的 JSON 无效"
    if isinstance(exc, ValidationError):
        errors = exc.errors(include_input=False)
        if any(error.get("type") == "json_invalid" for error in errors):
            return "模型返回空内容或无效 JSON"
        return "模型返回的 JSON 结构不符合预期"
    if isinstance(exc, ValueError) and str(exc) == "invalid candidate ids":
        return "模型返回的候选 ID 与请求不一致"
    if isinstance(exc, ValueError) and str(exc) in {
        "missing items/evidence list",
        "model response JSON is not an object",
    }:
        return "模型返回的 JSON 结构不符合预期"
    if isinstance(exc, (KeyError, TypeError)):
        return "上游响应字段不符合预期"
    if isinstance(exc, OSError):
        return "本地网络请求失败"
    return "模型响应校验失败"
