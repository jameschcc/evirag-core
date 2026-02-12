# ==============================================================================
# EviRAG — Enterprise-grade High-performance Offline Knowledge Base with
# Traceable Retrieval and Generation Architecture
#         DOI: 10.3969/j.issn.1671-3370.2026.02.032 (S.Chang) https://d.wanfangdata.com.cn/periodical/zgxxj202602032
# ==============================================================================

from __future__ import annotations

from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _canonical_provider(value: str) -> str:
    normalized = value.strip().lower()
    return "openai" if normalized in {"openai", "openai-compatible", "openai_compatible"} else normalized


class ProviderSettingsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["ollama", "openai"]
    base_url: str = Field(min_length=1, max_length=2048)
    model: str = Field(min_length=1, max_length=200)
    api_key: str | None = Field(default=None, max_length=4096)
    clear_api_key: bool = False

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        parsed = urlsplit(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("接口地址必须是完整的 http:// 或 https:// 地址")
        try:
            _ = parsed.port
        except ValueError as exc:
            raise ValueError("接口地址中的端口无效") from exc
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("接口地址不能包含账号、密码、查询参数或片段")
        return normalized

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("模型名称不能为空")
        return normalized

    @field_validator("api_key")
    @classmethod
    def normalize_api_key(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None


class ProviderPlanCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=80)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("方案名称不能为空")
        return normalized


class StructuredOutputUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["tool", "json"]


def provider_settings_storage(settings) -> dict[str, str]:
    return {
        "ai_provider": _canonical_provider(settings.effective_ai_provider),
        "ai_base_url": settings.effective_ai_base_url,
        "ai_model": settings.effective_ai_model,
        "ai_api_key": settings.ai_api_key,
        "embedding_provider": _canonical_provider(settings.effective_embedding_provider),
        "embedding_base_url": settings.effective_embedding_base_url,
        "embedding_model": settings.embedding_model,
        "embedding_api_key": settings.embedding_api_key,
        "structured_mode": settings.structured_mode,
        "tool_choice": settings.tool_choice,
    }


def provider_settings_view(settings) -> dict[str, dict[str, str | bool]]:
    return {
        "ai": {
            "provider": _canonical_provider(settings.effective_ai_provider),
            "base_url": settings.effective_ai_base_url,
            "model": settings.effective_ai_model,
            "api_key_configured": bool(settings.ai_api_key),
            "structured_mode": settings.structured_mode,
            "tool_choice": settings.tool_choice,
        },
        "embedding": {
            "provider": _canonical_provider(settings.effective_embedding_provider),
            "base_url": settings.effective_embedding_base_url,
            "model": settings.embedding_model,
            "api_key_configured": bool(settings.embedding_api_key),
        },
    }
