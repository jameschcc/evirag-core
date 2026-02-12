# ==============================================================================
# EviRAG — Enterprise-grade High-performance Offline Knowledge Base with
# Traceable Retrieval and Generation Architecture
# Personal Research Project · Author: S.Chang (常帅)
# Paper: 常帅. 企业级高性能离网知识库可溯源检索与生成架构[J].
#        China Information World, 2026(2).
#        DOI: 10.3969/j.issn.1671-3370.2026.02.032
#        https://d.wanfangdata.com.cn/periodical/zgxxj202602032
# ==============================================================================

"""HTTP client for an optional query-document Cross-Encoder reranker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from .contracts import Evidence


@dataclass(frozen=True)
class CrossEncoderResult:
    evidence: list[Evidence]
    trace: list[dict[str, Any]]


def parse_cross_encoder_results(body: Any, evidence: list[Evidence]) -> CrossEncoderResult:
    """Validate an Infinity ``/rerank`` response and order evidence by score."""
    if not isinstance(body, dict) or not isinstance(body.get("results"), list):
        raise ValueError("cross-encoder response is missing results")

    scores: dict[int, float] = {}
    for item in body["results"]:
        if not isinstance(item, dict):
            raise ValueError("cross-encoder result item is not an object")
        index = item.get("index")
        score = item.get("relevance_score")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("cross-encoder result index is invalid")
        if index < 0 or index >= len(evidence) or index in scores:
            raise ValueError("cross-encoder result index is out of range")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("cross-encoder relevance score is invalid")
        scores[index] = float(score)

    if set(scores) != set(range(len(evidence))):
        raise ValueError("cross-encoder response does not score every candidate")

    ordered = sorted(enumerate(evidence), key=lambda item: (-scores[item[0]], item[0]))
    trace = [
        {
            "evidence_id": item.evidence_id,
            "cross_encoder_score": scores[index],
            "cross_encoder_rank": rank,
        }
        for rank, (index, item) in enumerate(ordered, start=1)
    ]
    return CrossEncoderResult(evidence=[item for _, item in ordered], trace=trace)


class HttpCrossEncoder:
    """Small fail-open client for an Infinity-compatible ``/rerank`` API."""

    def __init__(self, base_url: str, model: str, timeout: float = 15.0):
        self.base_url = base_url.rstrip("/")
        self.model = model.strip()
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.base_url)

    async def rerank(
        self,
        question: str,
        evidence: list[Evidence],
        text_limit: int = 12000,
    ) -> CrossEncoderResult:
        if not self.available:
            raise RuntimeError("cross-encoder is not configured")
        documents = [item.chunk.text[:text_limit] for item in evidence]
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/rerank",
                json={"query": question, "documents": documents},
            )
            response.raise_for_status()
            return parse_cross_encoder_results(response.json(), evidence)
