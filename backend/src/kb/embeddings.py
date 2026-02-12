# ==============================================================================
# EviRAG — Enterprise-grade High-performance Offline Knowledge Base with
# Traceable Retrieval and Generation Architecture
# Personal Research Project · Author: S.Chang (常帅)
# Paper: 常帅. 企业级高性能离网知识库可溯源检索与生成架构[J].
#        China Information World, 2026(2).
#        DOI: 10.3969/j.issn.1671-3370.2026.02.032
#        https://d.wanfangdata.com.cn/periodical/zgxxj202602032
# ==============================================================================

from __future__ import annotations

import hashlib
import math

import httpx


class EmbeddingError(RuntimeError):
    """Raised when the embedding service cannot produce a usable vector."""


class EmbeddingGateway:
    """Real embeddings from Ollama or an OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: str = "",
        model: str = "bge-m3",
        timeout: float = 30.0,
        batch_size: int = 16,
        provider: str = "ollama",
        api_key: str = "",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.batch_size = max(1, batch_size)
        self.provider = provider.strip().lower()
        self.api_key = api_key.strip()

    @property
    def revision(self) -> str:
        identity = f"{self.provider}\0{self.base_url}\0{self.model}".encode()
        digest = hashlib.sha256(identity).hexdigest()[:16]
        return f"{self.provider}-embed-v2:{self.model}:{digest}"

    @property
    def available(self) -> bool:
        return bool(self.base_url and self.model and self.provider in {"ollama", "openai", "openai-compatible", "openai_compatible"})

    @property
    def revision_id(self) -> str:
        return self.revision

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self.embed_texts([text])
        if not vectors:
            raise EmbeddingError("embedding response was empty")
        return vectors[0]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.available:
            raise EmbeddingError("embedding service is not configured")
        if not texts:
            return []
        prepared = [text if text.strip() else " " for text in texts]
        vectors: list[list[float]] = []
        if self.provider not in {"ollama", "openai", "openai-compatible", "openai_compatible"}:
            raise EmbeddingError(f"unsupported embedding provider: {self.provider}")
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for start in range(0, len(prepared), self.batch_size):
                batch = prepared[start : start + self.batch_size]
                headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key and self.provider != "ollama" else {}
                if self.provider == "ollama":
                    url = f"{self.base_url}/api/embed"
                    payload = {"model": self.model, "input": batch}
                else:
                    url = f"{self.base_url}/embeddings"
                    payload = {"model": self.model, "input": batch}
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                body = response.json()
                if self.provider == "ollama":
                    embeddings = body.get("embeddings")
                else:
                    data = body.get("data")
                    if isinstance(data, list):
                        data = sorted(data, key=lambda item: item.get("index", 0))
                        embeddings = [item.get("embedding") for item in data]
                    else:
                        embeddings = None
                if not isinstance(embeddings, list) or len(embeddings) != len(batch) or any(not isinstance(v, list) for v in embeddings):
                    raise EmbeddingError("unexpected embedding response shape")
                vectors.extend(_normalize(vector) for vector in embeddings)
        return vectors


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(value) * float(value) for value in vector))
    if not norm:
        return [float(value) for value in vector]
    return [float(value) / norm for value in vector]
