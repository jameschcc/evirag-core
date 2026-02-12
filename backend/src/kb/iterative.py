# ==============================================================================
# EviRAG — Enterprise-grade High-performance Offline Knowledge Base with
# Traceable Retrieval and Generation Architecture
#         DOI: 10.3969/j.issn.1671-3370.2026.02.032 (S.Chang) https://d.wanfangdata.com.cn/periodical/zgxxj202602032
# ==============================================================================

from __future__ import annotations

from .contracts import Chunk, Evidence, RetrievalResult, SearchScope


class IterativeEvidenceService:
    """Bounded loop: neighbor expansion + evidence-grounded supplementary retrieval.

    The evidence session keeps every discovered chunk until the AI explicitly
    marks it irrelevant (recorded by stable chunk id); no numeric cap silently
    drops evidence between rounds or before generation.
    """

    def __init__(
        self,
        retriever,
        reranker,
        embeddings,
        enabled: bool = True,
        max_rounds: int = 3,
        neighbor_max_count: int = 3,
        supplementary_keywords_max: int = 5,
        dense_provider=None,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.embeddings = embeddings
        self.enabled = enabled
        self.max_rounds = max(0, max_rounds)
        self.neighbor_max_count = max(0, neighbor_max_count)
        self.supplementary_keywords_max = max(0, supplementary_keywords_max)
        # 可选的外部稠密召回后端（pgvector）。为 None 或调用失败时退回
        # retriever 的内存精确计算，检索不会因为数据库侧故障中断。
        self.dense_provider = dense_provider

    async def _dense_arguments(self, query_embeddings, scope, source_kinds):
        if self.dense_provider is None or not query_embeddings:
            return {}
        try:
            response = await self.dense_provider(query_embeddings, scope, source_kinds)
        except Exception:  # noqa: BLE001 - 外部后端故障必须降级，不能中断检索
            return {}
        if response is None:
            return {}
        candidates, available, missing = response
        return {
            "dense_candidates": candidates,
            "dense_available_count": available,
            "dense_missing_count": missing,
        }

    async def run(
        self,
        question: str,
        scope: SearchScope,
        limit: int,
        lexical_queries: list[str],
        query_embeddings: list[list[float]],
        enabled: bool | None = None,
        source_kinds: list[str] | None = None,
        on_progress=None,
    ) -> RetrievalResult:
        rounds_enabled = self.enabled if enabled is None else enabled
        allowed_kinds = {str(kind) for kind in (source_kinds or ["document", "code"])}
        # ``limit`` is only the initial display batch. Every discovered piece of
        # evidence is preserved for generation unless the AI explicitly rules it
        # irrelevant; there is no numeric final cap.
        initial_limit = max(6, limit)
        _report(on_progress, "retrieval", "首轮混合检索中")
        result = self.retriever.search(
            question, scope, initial_limit,
            lexical_queries=lexical_queries, query_embeddings=query_embeddings,
            source_kinds=list(allowed_kinds),
            **await self._dense_arguments(query_embeddings, scope, list(allowed_kinds)),
        )
        _report(on_progress, "rerank", "首批混合证据已到达", retrieval=result)
        result = await self.reranker.rank(question, result)
        _report(on_progress, "rerank", "首轮证据重排完成", retrieval=result)
        trace: list[dict] = []
        next_index = _next_evidence_index(result.evidence)
        tried_neighbours: set[tuple[str, str, int]] = set()
        tried_keywords: set[str] = set()
        # Stable exclusion set: chunk ids the AI judged irrelevant. They are
        # kept out of every later round instead of being silently re-added or
        # re-judged; the audit trail stays in rerank_trace.
        excluded_chunk_ids: set[str] = set()
        _collect_irrelevant(result, excluded_chunk_ids)

        for round_index in range(1, self.max_rounds + 1):
            if not rounds_enabled:
                break
            plan = result.retrieval_plan or {}
            if not plan:
                _report(on_progress, "rerank", "没有可用补证计划，保留现有证据并继续回答")
                break
            if plan.get("status") == "unavailable":
                _report(on_progress, "rerank", "重排计划不可用，保留候选证据并继续回答")
                break
            if plan.get("sufficient") is True:
                _report(on_progress, "rerank", "AI 判断现有证据已充分，停止扩展")
                break
            _report(on_progress, "expansion", f"扩展检索：第 {round_index} 轮", round=round_index)
            requested_neighbours = []
            evidence_ids = {item.evidence_id for item in result.evidence}
            for request in plan.get("neighbour_requests") or []:
                if not isinstance(request, dict):
                    continue
                evidence_id = request.get("id")
                direction = request.get("direction")
                count = request.get("count", 1)
                if evidence_id not in evidence_ids or direction not in {"prev", "next", "nearby"}:
                    continue
                if not isinstance(count, int) or isinstance(count, bool) or count < 1 or self.neighbor_max_count < 1:
                    continue
                normalized = {"id": evidence_id, "direction": direction, "count": min(count, self.neighbor_max_count)}
                key = (normalized["id"], normalized["direction"], normalized["count"])
                if key not in tried_neighbours and normalized not in requested_neighbours:
                    requested_neighbours.append(normalized)
                if len(requested_neighbours) == 3:
                    break
            requested_keywords = [
                item for item in (plan.get("supplementary_keywords") or [])
                if item.strip().lower() not in tried_keywords
            ]
            keywords = _novel_keywords(requested_keywords, result.evidence)[: self.supplementary_keywords_max]
            tried_neighbours.update((item["id"], item["direction"], item["count"]) for item in requested_neighbours)
            tried_keywords.update(item.strip().lower() for item in requested_keywords)
            additions = self._neighbour_additions(result, requested_neighbours, scope, allowed_kinds)
            if keywords:
                additions.extend(await self._supplementary(question, scope, limit, lexical_queries, query_embeddings, keywords, allowed_kinds))
            merged, next_index, added = _merge_evidence(
                result.evidence, additions, next_index, excluded_chunk_ids
            )
            trace.append({
                "round": round_index,
                "neighbour_requests": requested_neighbours,
                "supplementary_keywords": keywords,
                "added": added,
                "reason": plan.get("reason", ""),
            })
            expanded_result = result.model_copy(update={"evidence": merged, "retrieval_plan": {}, "iterative_trace": trace})
            _report(
                on_progress,
                "expansion",
                f"第 {round_index} 轮补入 {added} 段证据",
                retrieval=expanded_result,
                round=round_index,
                added=added,
            )
            if added == 0:
                break
            result = expanded_result
            result = await self.reranker.rank(question, result)
            _report(on_progress, "rerank", f"第 {round_index} 轮证据重排完成", retrieval=result, round=round_index)
            removed = _collect_irrelevant(result, excluded_chunk_ids)
            if removed:
                result = result.model_copy(update={"evidence": [item for item in result.evidence if str(item.chunk.chunk_id) not in excluded_chunk_ids]})
                _report(
                    on_progress,
                    "rerank",
                    f"第 {round_index} 轮剔除 {removed} 段不相关证据",
                    retrieval=result,
                    round=round_index,
                )

        return result.model_copy(update={"iterative_trace": trace})

    def _neighbour_additions(self, result: RetrievalResult, requests: list[dict], scope: SearchScope, allowed_kinds: set[str]) -> list[Chunk]:
        by_id = {item.evidence_id: item for item in result.evidence}
        additions: list[Chunk] = []
        allowed_kb_ids = {str(item) for item in (scope.kb_ids if scope is not None else [])}
        for request in requests:
            evidence = by_id.get(request["id"])
            if evidence is not None and (not allowed_kb_ids or str(evidence.chunk.kb_id) in allowed_kb_ids):
                additions.extend(self.retriever.neighbours(evidence.chunk, request["direction"], request["count"]))
        return [
            chunk for chunk in additions
            if chunk.source_kind in allowed_kinds
            and (not allowed_kb_ids or str(chunk.kb_id) in allowed_kb_ids)
        ]

    async def _supplementary(
        self,
        question: str,
        scope: SearchScope,
        limit: int,
        lexical_queries: list[str],
        query_embeddings: list[list[float]],
        keywords: list[str],
        allowed_kinds: set[str],
    ) -> list[Chunk]:
        extra_lexical = list(dict.fromkeys([*lexical_queries, *keywords]))
        extra_embeddings = list(query_embeddings)
        if self.embeddings.available:
            try:
                extra_embeddings += await self.embeddings.embed_texts(keywords)
            except Exception:
                pass
        sub = self.retriever.search(
            question, scope, max(6, limit),
            lexical_queries=extra_lexical, query_embeddings=extra_embeddings,
            source_kinds=list(allowed_kinds),
            **await self._dense_arguments(extra_embeddings, scope, list(allowed_kinds)),
        )
        return [evidence.chunk for evidence in sub.evidence]


def _next_evidence_index(evidence: list[Evidence]) -> int:
    highest = 0
    for item in evidence:
        digits = "".join(char for char in item.evidence_id if char.isdigit())
        if digits:
            highest = max(highest, int(digits))
    return highest + 1


def _report(on_progress, stage: str, message: str, **extra) -> None:
    if on_progress is None:
        return
    try:
        on_progress(stage, message, **extra)
    except Exception:
        pass


def _novel_keywords(keywords: list[str], evidence: list[Evidence]) -> list[str]:
    corpus = " ".join(item.chunk.text.lower() for item in evidence)
    novel: list[str] = []
    seen: set[str] = set()
    for keyword in keywords:
        term = keyword.strip()
        key = term.lower()
        if not term or key in seen or key in corpus:
            continue
        seen.add(key)
        novel.append(term)
    return novel


def _merge_evidence(
    current: list[Evidence],
    additions: list[Chunk],
    next_index: int,
    excluded_chunk_ids: set[str] | None = None,
) -> tuple[list[Evidence], int, int]:
    merged = [
        item for item in current
        if excluded_chunk_ids is None or str(item.chunk.chunk_id) not in excluded_chunk_ids
    ]
    known = {str(item.chunk.chunk_id) for item in merged}
    index = next_index
    added = 0
    for chunk in additions:
        key = str(chunk.chunk_id)
        if key in known or (excluded_chunk_ids is not None and key in excluded_chunk_ids):
            continue
        known.add(key)
        merged.append(Evidence(evidence_id=f"E{index}", chunk=chunk, score=0.0))
        index += 1
        added += 1
    return merged, index, added


def _collect_irrelevant(result: RetrievalResult, excluded_chunk_ids: set[str]) -> int:
    """Record newly irrelevant evidence in the exclusion set; returns how many."""

    removed = 0
    for entry in result.rerank_trace:
        if entry.get("relevance") != "irrelevant":
            continue
        chunk_id = str(entry.get("chunk_id", ""))
        if not chunk_id or chunk_id in excluded_chunk_ids:
            continue
        excluded_chunk_ids.add(chunk_id)
        removed += 1
    return removed
