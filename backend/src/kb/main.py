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

import asyncio
import io
import json
import logging
import re
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from time import perf_counter
from typing import Literal
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID, uuid4

import asyncpg
import httpx
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from .chunking import SemanticChunker
from .config import get_settings
from .auth import require_kb_access, require_kb_admin, require_kb_owner, resolve_principal
from .contracts import AnswerRequest, AnswerResponse, Chunk, ClearKnowledgeBaseRequest, DEFAULT_KB_ID, DocumentResponse, IndexCallbackRequest, KnowledgeBaseCreate, KnowledgeBaseMember, KnowledgeBaseResponse, RetrievalResult, SearchRequest, SearchScope
from .chunking import CodeChunker
from .document_parser import DocumentParseError, detect_format, detect_source_kind, parse_document
from .embeddings import EmbeddingError, EmbeddingGateway
from .glossary import GLOSSARY_COLLECTED, GLOSSARY_NONE, GLOSSARY_PENDING, GlossaryIndex, build_glossary_with_ai_assistance, detect_glossary_state, extract_glossary, glossary_candidates
from .generation import (
    CONSERVATIVE_TEMPERATURE,
    DEFAULT_DIVERGENT_TEMPERATURE,
    DIVERGENT_STYLE,
    AnswerValidator,
    OllamaGateway,
    OllamaGenerationError,
)
from .iterative import IterativeEvidenceService
from .retrieval import InMemoryHybridRetriever
from .workers import index_document
from .feedback import FeedbackRequest, FeedbackStore
from .conversations import ConversationCreate, ConversationRename, ConversationStore
from .catalog import DocumentCatalog
from .query_planner import QueryPlanner
from .progress import ProgressStore
from .provider_settings import ProviderPlanCreate, ProviderSettingsUpdate, StructuredOutputUpdate, provider_settings_storage, provider_settings_view
from .semantic_reranker import SemanticReranker

logger = logging.getLogger(__name__)


class PrototypeState:
    def __init__(self, settings=None) -> None:
        settings = settings or get_settings()
        self.settings = settings
        self.retriever = InMemoryHybridRetriever(
            settings.pipeline_revision,
            settings.retrieval_min_score,
            settings.retrieval_dense_min_score,
        )
        self.chunker = SemanticChunker()
        self.glossary = GlossaryIndex()
        self.progress = ProgressStore()
        self.embeddings = EmbeddingGateway(
            settings.effective_embedding_base_url,
            settings.embedding_model,
            settings.embedding_timeout_seconds,
            settings.embedding_batch_size,
            settings.effective_embedding_provider,
            settings.embedding_api_key,
        )
        self.retriever.embedding_revision = self.embeddings.revision_id if self.embeddings.available else None
        self.validator = AnswerValidator()
        self.reranker = SemanticReranker(
            settings.effective_ai_base_url,
            settings.effective_ai_model,
            45.0,
            settings.effective_ai_provider,
            settings.ai_api_key,
            settings.reranker_base_url,
            settings.reranker_model,
            settings.reranker_timeout_seconds,
            structured_mode=settings.structured_mode,
            tool_choice=settings.tool_choice,
        )
        self.iterative = IterativeEvidenceService(
            self.retriever,
            self.reranker,
            self.embeddings,
            settings.iterative_retrieval_enabled,
            settings.iterative_max_rounds,
            neighbor_max_count=settings.neighbor_max_count,
            supplementary_keywords_max=settings.supplementary_keywords_max,
        )
        self.query_planner = QueryPlanner(
            settings.effective_ai_base_url,
            settings.effective_ai_model,
            settings.request_timeout_seconds,
            settings.effective_ai_provider,
            settings.ai_api_key,
            settings.structured_mode,
            settings.tool_choice,
        )
        self.generator = OllamaGateway(
            settings.effective_ai_base_url,
            settings.effective_ai_model,
            settings.generation_timeout_seconds,
            settings.max_context_chars,
            settings.ollama_num_predict,
            settings.effective_ai_provider,
            settings.ai_api_key,
            settings.structured_mode,
            settings.tool_choice,
            divergent_temperature=settings.divergent_temperature,
        )
        self.documents: dict[UUID, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.feedback = FeedbackStore(get_settings().postgres_dsn)
    app.state.catalog = DocumentCatalog(get_settings().postgres_dsn)
    app.state.conversations = ConversationStore(get_settings().postgres_dsn)
    await app.state.feedback.open()
    await app.state.catalog.open()
    await app.state.conversations.open()
    settings = get_settings()
    await app.state.catalog.seed_default_membership(settings.default_user_id)
    for group in ("ai", "embedding"):
        saved_provider_settings = await app.state.catalog.load_runtime_settings(f"provider_{group}")
        if saved_provider_settings:
            settings = settings.model_copy(update=saved_provider_settings)
    app.state.prototype = PrototypeState(settings)
    app.state.provider_settings_lock = asyncio.Lock()
    documents, chunks = await app.state.catalog.load()
    for item in chunks:
        app.state.prototype.retriever.add([item])
    app.state.prototype.documents = {item.pop("document_id"): item for item in documents}
    glossary_counts = await app.state.catalog.glossary_counts()
    for document_id, document in app.state.prototype.documents.items():
        document["glossary_count"] = glossary_counts.get(document_id, 0)
    app.state.prototype.glossary.replace(await app.state.catalog.load_glossary())
    # 稠密召回后端：pgvector 可用时把候选计算下推到数据库（精确扫描，不建
    # ANN 索引），否则保持内存实现。KB_RETRIEVAL_BACKEND=memory 可强制对照。
    app.state.vector_backend = "memory"
    backend = (getattr(settings, "retrieval_backend", "auto") or "auto").strip().lower()
    if backend != "memory" and getattr(app.state.catalog, "vector_ready", False):
        await app.state.catalog.backfill_embedding_column()
        app.state.prototype.iterative.dense_provider = _make_dense_provider(
            app.state.catalog, app.state.prototype
        )
        app.state.vector_backend = "pgvector"
    app.state.embedding_backfill_task = asyncio.create_task(_backfill_embeddings(list(chunks)))
    try:
        yield
    finally:
        backfill = getattr(app.state, "embedding_backfill_task", None)
        if backfill and not backfill.done():
            backfill.cancel()
            await asyncio.gather(backfill, return_exceptions=True)
        await app.state.catalog.close()
        await app.state.feedback.close()
        await app.state.conversations.close()


app = FastAPI(title="Offline Knowledge Base API", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _make_dense_provider(catalog, prototype):
    """pgvector 稠密召回适配器。

    返回 ``None`` 表示本次查询走内存精确计算（扩展不可用或查询未向量化）。
    """

    async def provider(query_embeddings, scope, source_kinds):
        if not getattr(catalog, "vector_ready", False):
            return None
        kb_ids = [str(item) for item in (scope.kb_ids if scope is not None else [])]
        return await catalog.search_dense(
            query_embeddings,
            source_kinds,
            kb_ids,
            prototype.retriever.embedding_revision,
            prototype.retriever.dense_min_score,
        )

    return provider


def state() -> PrototypeState:
    return app.state.prototype


async def _embed_queries(texts: list[str]) -> tuple[list[list[float]], str | None]:
    gateway = state().embeddings
    if not gateway.available or not texts:
        return [], None
    try:
        return await gateway.embed_texts(texts), None
    except (EmbeddingError, httpx.HTTPError, OSError, ValueError, TimeoutError) as exc:
        return [], f"查询向量化不可用，已降级为关键词召回：{exc}"


def _glossary_plan(question: str, kb_ids=None) -> tuple[list, list[str]]:
    expansions = state().glossary.lookup(question, kb_ids)
    return expansions, state().glossary.variants(question, expansions)


def _progress_reporter(progress_id: str | None):
    key = (progress_id or "").strip()

    def report(stage: str, message: str, **extra) -> None:
        retrieval = extra.pop("retrieval", None)
        if retrieval is not None:
            extra["result"] = retrieval.model_dump(mode="json") if hasattr(retrieval, "model_dump") else retrieval
        query_analysis = extra.get("query_analysis")
        if query_analysis is not None and hasattr(query_analysis, "model_dump"):
            extra["query_analysis"] = query_analysis.model_dump(mode="json")
        state().progress.set(key, stage, message, **extra)

    return report


def _seed_progress(request: SearchRequest | AnswerRequest, kb_id: UUID) -> None:
    """Create the progress entry before the SSE client connects.

    The browser opens ``/progress/{id}/stream`` immediately after sending the
    request; without a seed the first connection can 404 and EventSource waits
    for its retry timer, which delays the whole timeline.
    """
    key = (request.progress_id or "").strip()
    if not key:
        return
    state().progress.set(
        key,
        "planning",
        "准备开始…",
        question=request.question,
        kb_ids=[str(kb_id)],
        source_kinds=list(request.source_kinds),
    )


def _safe_endpoint_url(value: str | None) -> str | None:
    if not value:
        return None
    try:
        parsed = urlsplit(value.strip())
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return None
    if not parsed.scheme or not hostname:
        return None
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = f"{hostname}:{port}" if port is not None else hostname
    return urlunsplit((parsed.scheme, netloc, parsed.path.rstrip("/"), "", ""))


def _chat_model_provenance(client) -> dict:
    try:
        endpoint_url = client.endpoint
    except ValueError:
        endpoint_url = None
    return {
        "provider": client.provider,
        "model": client.model,
        "base_url": _safe_endpoint_url(client.base_url),
        "endpoint_url": _safe_endpoint_url(endpoint_url),
    }


def _embedding_model_provenance(gateway) -> dict:
    if gateway.provider == "ollama":
        path = "/api/embed"
    elif gateway.provider in {"openai", "openai-compatible", "openai_compatible"}:
        path = "/embeddings"
    else:
        path = ""
    endpoint = f"{gateway.base_url.rstrip('/')}{path}" if gateway.base_url and path else None
    return {
        "provider": gateway.provider,
        "model": gateway.model,
        "base_url": _safe_endpoint_url(gateway.base_url),
        "endpoint_url": _safe_endpoint_url(endpoint),
        "revision": gateway.revision_id,
        "available": gateway.available,
    }


def _progress_timing_snapshot(progress: dict | None, progress_id: str | None) -> dict:
    events = []
    raw_history = progress.get("history", []) if progress else []
    for item in raw_history:
        try:
            timestamp_ms = int(item["timestamp_ms"])
        except (KeyError, TypeError, ValueError):
            continue
        events.append({
            "stage": item.get("stage"),
            "message": item.get("message"),
            "revision": item.get("revision"),
            "timestamp_ms": timestamp_ms,
            "timestamp_utc": datetime.fromtimestamp(timestamp_ms / 1000, timezone.utc).isoformat(),
        })
    if not events:
        return {
            "progress_id": progress_id,
            "started_at_ms": None,
            "completed_at_ms": None,
            "total_duration_ms": None,
            "events": [],
        }
    started_at_ms = events[0]["timestamp_ms"]
    completed_at_ms = events[-1]["timestamp_ms"]
    for index, event in enumerate(events):
        event["elapsed_from_start_ms"] = max(0, event["timestamp_ms"] - started_at_ms)
        next_timestamp = events[index + 1]["timestamp_ms"] if index + 1 < len(events) else completed_at_ms
        event["duration_to_next_ms"] = max(0, next_timestamp - event["timestamp_ms"])
    return {
        "progress_id": progress_id,
        "started_at_ms": started_at_ms,
        "started_at_utc": events[0]["timestamp_utc"],
        "completed_at_ms": completed_at_ms,
        "completed_at_utc": events[-1]["timestamp_utc"],
        "total_duration_ms": max(0, completed_at_ms - started_at_ms),
        "events": events,
    }


def _cached_retrieval(progress_id: str | None, question: str, kb_ids: list[UUID] | None = None, source_kinds: list[str] | None = None) -> RetrievalResult | None:
    key = (progress_id or "").strip()
    if not key:
        return None
    item = state().progress.get(key)
    expected_kb_ids = [str(item) for item in (kb_ids or [DEFAULT_KB_ID])]
    expected_kinds = sorted(source_kinds or ["document", "code"])
    stored_kinds = item.get("source_kinds") if item else None
    if (
        not item
        or item.get("question") != question
        or item.get("kb_ids") != expected_kb_ids
        or (stored_kinds is not None and sorted(stored_kinds) != expected_kinds)
        or not item.get("result")
    ):
        return None
    try:
        return RetrievalResult.model_validate(item["result"])
    except (TypeError, ValueError):
        return None


async def _retrieve(
    request: SearchRequest,
    report,
    history: list[dict] | None = None,
) -> tuple[RetrievalResult, object]:
    scope = SearchScope(kb_ids=request.kb_ids or [DEFAULT_KB_ID])
    model_provenance = {
        "retrieval_chat": _chat_model_provenance(state().query_planner.client),
        "embedding": _embedding_model_provenance(state().embeddings),
    }
    report(
        "planning",
        "语义分析与术语展开中（同时准备首轮检索）",
        question=request.question,
        kb_ids=[str(item) for item in scope.kb_ids],
        source_kinds=list(request.source_kinds),
        model_provenance=model_provenance,
    )
    expansions, variants = _glossary_plan(request.question, scope.kb_ids)
    planner_task = asyncio.create_task(
        state().query_planner.build_async(
            request.question, request.auto_semantic_analysis,
            getattr(request, "query_analysis", None), history,
        )
    )
    embedding_texts = list(dict.fromkeys([request.question, *variants]))
    # 多轮追问要先拿到改写后的问题再编码向量，单轮保持并行以压低首字延迟。
    embedding_task = None if history else asyncio.create_task(_embed_queries(embedding_texts))

    quick_analysis = None
    planner_build = getattr(state().query_planner, "build", None)
    if planner_build is not None:
        quick_analysis = planner_build(request.question, request.auto_semantic_analysis, history)
    quick_queries = [request.question]
    if quick_analysis is not None:
        quick_queries = list(dict.fromkeys([*quick_queries, *quick_analysis.retrieval_queries]))
    quick_result = state().retriever.search(
        request.question,
        scope,
        max(6, request.limit),
        lexical_queries=quick_queries,
        source_kinds=request.source_kinds,
    )
    report(
        "retrieval",
        "首批关键词证据已到达，正在继续准备向量与语义增强",
        retrieval=quick_result,
        query_analysis=quick_analysis,
        question=request.question,
        kb_ids=[str(item) for item in scope.kb_ids],
    )
    await asyncio.sleep(0)

    analysis = await planner_task
    analysis = analysis.model_copy(update={"glossary": expansions})
    if analysis.intent in ("chitchat", "meta"):
        # 纯 L1 判定：AI 说这条输入不需要查知识库，流程到此为止。不编码查询
        # 向量、不做混合召回、也不调用重排——否则「你好」仍会白付 30 秒。
        skipped = RetrievalResult(
            retrieval_method="skipped",
            rerank_method="skipped",
            evidence=[],
            skipped=True,
            skip_reason=analysis.intent,
            pipeline_revision=getattr(
                state().retriever, "pipeline_revision", state().settings.pipeline_revision
            ),
            query_analysis=analysis,
        )
        report(
            "done",
            "语义分析判定无需检索，已跳过检索与重排",
            retrieval=skipped,
            query_analysis=analysis,
            question=request.question,
            kb_ids=[str(item) for item in scope.kb_ids],
        )
        # 判定结果已出，并行预取的向量编码没有意义，取消掉避免多打一次
        # embedding 请求；取消异常只用于回收任务，不影响跳过结果。
        if embedding_task is not None and not embedding_task.done():
            embedding_task.cancel()
            await asyncio.gather(embedding_task, return_exceptions=True)
        return skipped, analysis
    report(
        "planning",
        f"语义分析完成，术语展开 {len(expansions)} 条，正在准备首轮混合检索",
        query_analysis=analysis,
        question=request.question,
    )
    # 追问用改写后的独立问题做向量召回与重排；原问题仍保留在 query_analysis 里。
    effective_question = (analysis.resolved_question or "").strip() or request.question
    if history:
        query_embeddings, embedding_warning = await _embed_queries(
            list(dict.fromkeys([effective_question, request.question, *variants]))
        )
    else:
        query_embeddings, embedding_warning = await embedding_task
    lexical_queries = list(dict.fromkeys([
        *analysis.retrieval_queries,
        *variants,
        *[entry.expansion for entry in expansions],
    ]))
    report("retrieval", "查询向量已就绪，首轮混合检索中", query_analysis=analysis, question=request.question)
    result = await state().iterative.run(
        question=effective_question,
        scope=scope,
        limit=request.limit,
        lexical_queries=lexical_queries,
        query_embeddings=query_embeddings,
        enabled=request.iterative_retrieval,
        source_kinds=request.source_kinds,
        on_progress=report,
    )
    result = result.model_copy(update={
        "query_analysis": analysis,
        "warnings": [*analysis.warnings, *([embedding_warning] if embedding_warning else []), *result.warnings],
    })
    report("done", "检索完成", retrieval=result, query_analysis=analysis, question=request.question, kb_ids=[str(item) for item in scope.kb_ids])
    return result, analysis


async def _ensure_embeddings(chunks: list[Chunk]) -> list[Chunk]:
    gateway = state().embeddings
    if not gateway.available or not chunks:
        return chunks
    revision = gateway.revision_id
    missing = [chunk for chunk in chunks if not chunk.embedding or chunk.embedding_revision != revision]
    if not missing:
        return chunks
    try:
        vectors = await gateway.embed_texts([chunk.text for chunk in missing])
    except (EmbeddingError, httpx.HTTPError, OSError, ValueError, TimeoutError):
        return chunks
    if len(vectors) != len(missing):
        return chunks
    mapping = {str(chunk.chunk_id): vector for chunk, vector in zip(missing, vectors)}
    return [
        chunk.model_copy(update={"embedding": mapping[str(chunk.chunk_id)], "embedding_revision": revision})
        if str(chunk.chunk_id) in mapping
        else chunk
        for chunk in chunks
    ]


async def _backfill_embeddings(chunks: list[Chunk]) -> None:
    revised = await _ensure_embeddings(chunks)
    changed: dict[UUID, list[Chunk]] = {}
    for before, after in zip(chunks, revised):
        if after.embedding and before.embedding_revision != after.embedding_revision:
            changed.setdefault(after.document_id, [])
    if not changed:
        return
    for document_id in changed:
        await app.state.catalog.replace_paragraphs(document_id, [chunk for chunk in revised if chunk.document_id == document_id])
    for chunk in revised:
        app.state.prototype.retriever.add([chunk])


async def _restart_embedding_backfill() -> None:
    previous = getattr(app.state, "embedding_backfill_task", None)
    if previous and not previous.done():
        previous.cancel()
        await asyncio.gather(previous, return_exceptions=True)
    task = asyncio.create_task(_backfill_embeddings(state().retriever.all_chunks()))
    app.state.embedding_backfill_task = task


def _apply_provider_settings(settings, group: str) -> None:
    prototype = state()
    prototype.settings = settings
    if group == "ai":
        prototype.reranker = SemanticReranker(
            settings.effective_ai_base_url,
            settings.effective_ai_model,
            45.0,
            settings.effective_ai_provider,
            settings.ai_api_key,
            settings.reranker_base_url,
            settings.reranker_model,
            settings.reranker_timeout_seconds,
            structured_mode=settings.structured_mode,
            tool_choice=settings.tool_choice,
        )
        prototype.iterative.reranker = prototype.reranker
        prototype.query_planner = QueryPlanner(
            settings.effective_ai_base_url,
            settings.effective_ai_model,
            settings.request_timeout_seconds,
            settings.effective_ai_provider,
            settings.ai_api_key,
        )
        prototype.generator = OllamaGateway(
            settings.effective_ai_base_url,
            settings.effective_ai_model,
            settings.generation_timeout_seconds,
            settings.max_context_chars,
            settings.ollama_num_predict,
            settings.effective_ai_provider,
            settings.ai_api_key,
            settings.structured_mode,
            settings.tool_choice,
            divergent_temperature=settings.divergent_temperature,
        )
    else:
        prototype.embeddings = EmbeddingGateway(
            settings.effective_embedding_base_url,
            settings.embedding_model,
            settings.embedding_timeout_seconds,
            settings.embedding_batch_size,
            settings.effective_embedding_provider,
            settings.embedding_api_key,
        )
        prototype.retriever.embedding_revision = (
            prototype.embeddings.revision_id if prototype.embeddings.available else None
        )
        prototype.iterative.embeddings = prototype.embeddings


def _provider_group_storage(settings, group: str) -> dict[str, str]:
    prefix = f"{group}_"
    stored = provider_settings_storage(settings)
    selected = {
        key: value for key, value in stored.items()
        if key.startswith(prefix)
    }
    if group == "ai":
        # Structured-output selection belongs to the AI provider but has no
        # "ai_" prefix because Settings exposes it as a top-level field.
        for key in ("structured_mode", "tool_choice"):
            selected[key] = stored[key]
    return selected


def _provider_plan_view(plan: dict, active_plan_id: str | None) -> dict:
    payload = plan["payload"]
    return {
        "id": str(plan["plan_id"]),
        "name": plan["name"],
        "created_at": plan["created_at"],
        "updated_at": plan["updated_at"],
        "active": str(plan["plan_id"]) == active_plan_id,
        "settings": {
            "ai": {
                "provider": payload["ai_provider"],
                "base_url": payload["ai_base_url"],
                "model": payload["ai_model"],
                "api_key_configured": bool(payload.get("ai_api_key")),
            },
            "embedding": {
                "provider": payload["embedding_provider"],
                "base_url": payload["embedding_base_url"],
                "model": payload["embedding_model"],
                "api_key_configured": bool(payload.get("embedding_api_key")),
            },
        },
    }


def _provider_plan_snapshot(payload: dict) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for group in ("ai", "embedding"):
        validated = ProviderSettingsUpdate.model_validate({
            "provider": payload.get(f"{group}_provider"),
            "base_url": payload.get(f"{group}_base_url"),
            "model": payload.get(f"{group}_model"),
            "api_key": payload.get(f"{group}_api_key", ""),
        })
        snapshot.update({
            f"{group}_provider": validated.provider,
            f"{group}_base_url": validated.base_url,
            f"{group}_model": validated.model,
            f"{group}_api_key": validated.api_key or "",
        })
    return snapshot


@app.get("/service-info")
@app.get("/")
async def root() -> dict:
    return {
        "service": "offline-knowledge-base",
        "version": "0.1.0",
        "status": "ok",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "health": "/health/ready",
        "api": "/api/v1",
        "model_service": "hybrid BM25 + configured embeddings / configured AI relevance rerank",
        "ai_provider": state().settings.effective_ai_provider,
        "ai_base_url": state().settings.effective_ai_base_url,
        "ai_model": state().settings.effective_ai_model,
        "ai_api_key_configured": bool(state().settings.ai_api_key),
        "embedding_provider": state().settings.effective_embedding_provider,
        "embedding_base_url": state().settings.effective_embedding_base_url,
        "embedding_model": state().settings.embedding_model if state().embeddings.available else None,
        "embedding_api_key_configured": bool(state().settings.embedding_api_key),
        "dense_retrieval_enabled": state().embeddings.available,
        "vector_backend": getattr(app.state, "vector_backend", "memory"),
        "iterative_retrieval_enabled": state().settings.iterative_retrieval_enabled,
        "iterative_max_rounds": state().settings.iterative_max_rounds,
        "ollama_base_url": state().settings.effective_ai_base_url if state().settings.effective_ai_provider == "ollama" else None,
        "ollama_model": state().settings.effective_ai_model if state().settings.effective_ai_provider == "ollama" else None,
        "generation_enabled": state().generator.client.available,
        "structured_mode": state().settings.structured_mode,
        "tool_choice": state().settings.tool_choice,
    }


@app.get("/api/v1/settings/providers")
async def read_provider_settings() -> dict:
    task = getattr(app.state, "embedding_backfill_task", None)
    return {
        **provider_settings_view(state().settings),
        "embedding_rebuild_running": bool(task and not task.done()),
    }


@app.get("/api/v1/settings/provider-plans")
async def list_provider_plans() -> dict:
    active = await app.state.catalog.load_runtime_settings("active_provider_plan")
    active_plan_id = str(active["id"]) if active and active.get("id") else None
    plans = await app.state.catalog.list_provider_plans()
    return {
        "active_plan_id": active_plan_id,
        "plans": [_provider_plan_view(plan, active_plan_id) for plan in plans],
    }


@app.post("/api/v1/settings/provider-plans")
async def create_provider_plan(request: ProviderPlanCreate) -> dict:
    async with app.state.provider_settings_lock:
        settings = state().settings
        payload = provider_settings_storage(settings)
        plan_id = uuid4()
        runtime_settings = {
            "provider_ai": _provider_group_storage(settings, "ai"),
            "provider_embedding": _provider_group_storage(settings, "embedding"),
            "active_provider_plan": {"id": str(plan_id)},
        }
        try:
            await app.state.catalog.create_provider_plan(
                plan_id, request.name, payload, runtime_settings
            )
        except asyncpg.UniqueViolationError as exc:
            raise HTTPException(status_code=409, detail="provider_plan_name_already_exists") from exc
    plan = await app.state.catalog.get_provider_plan(plan_id)
    return {"plan": _provider_plan_view(plan, str(plan_id))}


@app.post("/api/v1/settings/provider-plans/{plan_id}/activate")
async def activate_provider_plan(plan_id: UUID) -> dict:
    async with app.state.provider_settings_lock:
        plan = await app.state.catalog.get_provider_plan(plan_id)
        if plan is None:
            raise HTTPException(status_code=404, detail="provider_plan_not_found")
        try:
            payload = _provider_plan_snapshot(plan["payload"])
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail="provider_plan_configuration_invalid") from exc

        current = state().settings
        old_embedding_identity = (
            current.effective_embedding_provider,
            current.effective_embedding_base_url,
            current.embedding_model,
        )
        settings = current.model_copy(update=payload)
        await app.state.catalog.update_runtime_settings({
            "provider_ai": _provider_group_storage(settings, "ai"),
            "provider_embedding": _provider_group_storage(settings, "embedding"),
            "active_provider_plan": {"id": str(plan_id)},
        })
        _apply_provider_settings(settings, "ai")
        _apply_provider_settings(settings, "embedding")
        new_embedding_identity = (
            settings.effective_embedding_provider,
            settings.effective_embedding_base_url,
            settings.embedding_model,
        )
        embedding_changed = old_embedding_identity != new_embedding_identity
        if embedding_changed:
            await _restart_embedding_backfill()

    return {
        "plan_id": str(plan_id),
        "settings": provider_settings_view(state().settings),
        "embedding_rebuild_started": embedding_changed,
    }


@app.delete("/api/v1/settings/provider-plans/{plan_id}")
async def delete_provider_plan(plan_id: UUID) -> dict:
    async with app.state.provider_settings_lock:
        plan = await app.state.catalog.get_provider_plan(plan_id)
        if plan is None:
            raise HTTPException(status_code=404, detail="provider_plan_not_found")
        await app.state.catalog.delete_provider_plan(plan_id)
    return {"deleted": str(plan_id)}


@app.put("/api/v1/settings/providers/{group}")
async def update_provider_settings(group: str, request: ProviderSettingsUpdate) -> dict:
    if group not in {"ai", "embedding"}:
        raise HTTPException(status_code=404, detail="provider_settings_group_not_found")

    async with app.state.provider_settings_lock:
        current = state().settings
        stored = provider_settings_storage(current)
        prefix = "ai" if group == "ai" else "embedding"
        provider_key = f"{prefix}_provider"
        base_url_key = f"{prefix}_base_url"
        model_key = f"{prefix}_model"
        api_key_key = f"{prefix}_api_key"
        old_embedding_identity = (
            current.effective_embedding_provider,
            current.effective_embedding_base_url,
            current.embedding_model,
        )
        stored[provider_key] = request.provider
        stored[base_url_key] = request.base_url
        stored[model_key] = request.model
        if request.clear_api_key:
            stored[api_key_key] = ""
        elif request.api_key:
            stored[api_key_key] = request.api_key

        settings = current.model_copy(update=stored)
        await app.state.catalog.update_runtime_settings({
            f"provider_{group}": _provider_group_storage(settings, group),
            "active_provider_plan": None,
        })
        _apply_provider_settings(settings, group)

        new_embedding_identity = (
            settings.effective_embedding_provider,
            settings.effective_embedding_base_url,
            settings.embedding_model,
        )
        embedding_changed = old_embedding_identity != new_embedding_identity
        if group == "embedding" and embedding_changed:
            await _restart_embedding_backfill()

    return {
        "settings": provider_settings_view(state().settings),
        "updated": group,
        "embedding_rebuild_started": embedding_changed if group == "embedding" else False,
    }


@app.put("/api/v1/settings/structured-output")
async def update_structured_output(request: StructuredOutputUpdate) -> dict:
    """Switch how structured calls carry their JSON (tool calls or plain JSON)."""
    async with app.state.provider_settings_lock:
        current = state().settings
        settings = current.model_copy(update={"structured_mode": request.mode})
        await app.state.catalog.update_runtime_settings({
            "provider_ai": _provider_group_storage(settings, "ai"),
            "active_provider_plan": None,
        })
        _apply_provider_settings(settings, "ai")
    return {"settings": provider_settings_view(state().settings), "updated": "ai"}


@app.delete("/api/v1/settings/providers/{group}")
async def reset_provider_settings(group: str) -> dict:
    if group not in {"ai", "embedding"}:
        raise HTTPException(status_code=404, detail="provider_settings_group_not_found")

    async with app.state.provider_settings_lock:
        current = state().settings
        base = get_settings()
        other_group = "embedding" if group == "ai" else "ai"
        updates = _provider_group_storage(current, other_group)
        updates.update(_provider_group_storage(base, group))
        old_embedding_identity = (
            current.effective_embedding_provider,
            current.effective_embedding_base_url,
            current.embedding_model,
        )
        settings = base.model_copy(update=updates)
        await app.state.catalog.update_runtime_settings({
            f"provider_{group}": None,
            "active_provider_plan": None,
        })
        _apply_provider_settings(settings, group)
        new_embedding_identity = (
            settings.effective_embedding_provider,
            settings.effective_embedding_base_url,
            settings.embedding_model,
        )
        embedding_changed = old_embedding_identity != new_embedding_identity
        if group == "embedding" and embedding_changed:
            await _restart_embedding_backfill()

    return {
        "settings": provider_settings_view(state().settings),
        "reset": group,
        "embedding_rebuild_started": embedding_changed if group == "embedding" else False,
    }


@app.get("/health/live")
async def live() -> dict:
    return {"status": "ok"}


@app.get("/health/ready")
async def ready() -> dict:
    return {"status": "ready", "pipeline_revision": state().settings.pipeline_revision}


@app.post("/api/v1/documents", response_model=DocumentResponse, status_code=202)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> DocumentResponse:
    return await _upload_document_for_kb(DEFAULT_KB_ID, background_tasks, file)


_CODE_IGNORED_DIRS = {
    ".git", ".svn", ".hg", "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist", "build", "target", "vendor", ".idea", ".vscode", ".next", ".nuxt",
    "coverage", ".pytest_cache", ".mypy_cache", ".ruff_cache", "__MACOSX",
}


def _safe_relative_path(name: str) -> str | None:
    """Normalize an uploaded folder/zip path and reject escapes and vendored dirs."""
    normalized = (name or "").replace("\\", "/").strip()
    if not normalized or normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        return None
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        return None
    if len(parts) > 20 or len(normalized) > 400:
        return None
    if any(part in _CODE_IGNORED_DIRS for part in parts[:-1]):
        return None
    return "/".join(parts)


def _has_ignored_dir(name: str) -> bool:
    parts = [part for part in (name or "").replace("\\", "/").split("/") if part not in ("", ".")]
    return any(part in _CODE_IGNORED_DIRS for part in parts[:-1])


def _looks_binary(content: bytes) -> bool:
    return b"\x00" in content[:8192]


def _code_skip_reason(content: bytes, max_bytes: int) -> str | None:
    if not content:
        return "empty"
    if len(content) > max_bytes:
        return "too_large"
    if _looks_binary(content):
        return "binary"
    return None


def _code_document_index(kb_id: UUID) -> dict[str, UUID]:
    return {
        document.get("title"): document_id
        for document_id, document in state().documents.items()
        if document.get("source_kind") == "code" and document.get("kb_id", DEFAULT_KB_ID) == kb_id
    }


async def _register_document(
    kb_id: UUID,
    title: str,
    content: bytes,
    content_type: str | None,
    source_kind: str,
    background_tasks: BackgroundTasks,
    existing_document_id: UUID | None = None,
) -> dict:
    """Write one uploaded file and enqueue indexing.

    ``existing_document_id`` reuses a document id so re-uploading the same code
    path replaces its version instead of piling up duplicates.
    """
    document_id = existing_document_id or uuid4()
    version_id = uuid4()
    document_format = detect_format(title, content_type)
    suffix = PurePosixPath(title).suffix.lower() or ".bin"
    upload_dir = Path(state().settings.ingestion_dir) / str(kb_id) / str(document_id) / str(version_id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    content_path = upload_dir / f"original{suffix}"
    content_path.write_bytes(content)
    created_at = datetime.now(timezone.utc).isoformat()
    document = {
        "kb_id": kb_id,
        "version_id": version_id,
        "title": title,
        "format": document_format,
        "source_kind": source_kind,
        "status": "processing",
        "chunk_count": 0,
        "error": None,
        "task_id": None,
        "created_at": created_at,
        "glossary_state": GLOSSARY_PENDING,
        "glossary_count": 0,
    }
    state().documents[document_id] = document
    if existing_document_id is not None:
        # Old chunks must disappear before the replacement version is ready.
        state().retriever.remove_document_chunks(document_id)
    await app.state.catalog.save_document(document_id, document, str(content_path))

    try:
        task = index_document.delay(
            str(document_id),
            str(version_id),
            title,
            str(content_path),
            content_type,
            str(kb_id),
        )
        task_id = task.id
        state().documents[document_id]["task_id"] = task_id
        await app.state.catalog.update_document(document_id, state().documents[document_id])
    except Exception as exc:
        # Local development can run without Redis; Compose uses the Celery path.
        task_id = None
        state().documents[document_id]["error"] = f"queue_unavailable: {exc}"
        await app.state.catalog.update_document(document_id, state().documents[document_id])
        background_tasks.add_task(_index_document_local, document_id, version_id, title, content, content_type, kb_id)
        state().documents[document_id]["error"] = None

    return {
        "document_id": document_id,
        "version_id": version_id,
        "title": title,
        "format": document_format,
        "source_kind": source_kind,
        "task_id": task_id,
        "created_at": created_at,
    }


async def _upload_document_for_kb(kb_id: UUID, background_tasks: BackgroundTasks, file: UploadFile) -> DocumentResponse:
    content = await file.read()
    if len(content) > state().settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="file_too_large")

    title = file.filename or "untitled"
    if detect_source_kind(title, file.content_type) == "code":
        # 文档与代码分属两个入库通道，避免代码素材出现在文档列表中。
        raise HTTPException(status_code=400, detail="code_file_uses_code_library")
    registered = await _register_document(kb_id, title, content, file.content_type, "document", background_tasks)
    return DocumentResponse(
        kb_id=kb_id,
        document_id=registered["document_id"],
        version_id=registered["version_id"],
        title=title,
        chunk_count=0,
        status="processing",
        format=registered["format"],
        source_kind="document",
        task_id=registered["task_id"],
        created_at=datetime.fromisoformat(registered["created_at"]),
        glossary_state=GLOSSARY_PENDING,
    )


async def _accept_code_file(
    kb_id: UUID,
    background_tasks: BackgroundTasks,
    existing: dict[str, UUID],
    raw_path: str,
    content: bytes,
    content_type: str | None,
) -> tuple[dict | None, dict | None]:
    title = _safe_relative_path(raw_path)
    if title is None:
        return None, {"path": raw_path, "reason": "ignored_dir" if _has_ignored_dir(raw_path) else "unsafe_path"}
    if detect_source_kind(title, content_type) != "code":
        return None, {"path": title, "reason": "not_code"}
    reason = _code_skip_reason(content, state().settings.code_max_file_bytes)
    if reason is not None:
        return None, {"path": title, "reason": reason}
    existing_id = existing.pop(title, None)
    registered = await _register_document(
        kb_id, title, content, content_type, "code", background_tasks, existing_id,
    )
    existing[title] = registered["document_id"]
    return {
        "title": title,
        "document_id": str(registered["document_id"]),
        "status": "processing",
        "replaced": existing_id is not None,
    }, None


async def _upload_code_files_for_kb(
    kb_id: UUID,
    background_tasks: BackgroundTasks,
    files: list[UploadFile],
    paths: list[str],
) -> dict:
    settings = state().settings
    if len(files) > settings.code_max_files_per_request:
        raise HTTPException(status_code=413, detail="too_many_files")
    existing = _code_document_index(kb_id)
    accepted: list[dict] = []
    skipped: list[dict] = []
    for index, upload in enumerate(files):
        raw_path = paths[index] if index < len(paths) and paths[index] else (upload.filename or "")
        content = await upload.read()
        entry, skip = await _accept_code_file(kb_id, background_tasks, existing, raw_path, content, upload.content_type)
        if entry is not None:
            accepted.append(entry)
        if skip is not None:
            skipped.append(skip)
    return {"total": len(files), "accepted": accepted, "skipped": skipped}


async def _upload_code_zip_for_kb(
    kb_id: UUID,
    background_tasks: BackgroundTasks,
    file: UploadFile,
) -> dict:
    settings = state().settings
    content = await file.read()
    if len(content) > settings.code_zip_max_bytes:
        raise HTTPException(status_code=413, detail="zip_too_large")
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail="invalid_zip") from exc

    infos = archive.infolist()
    if len(infos) > settings.code_max_zip_entries:
        raise HTTPException(status_code=413, detail="too_many_zip_entries")
    if sum(item.file_size for item in infos) > settings.code_zip_max_extracted_bytes:
        raise HTTPException(status_code=413, detail="zip_expands_too_large")

    existing = _code_document_index(kb_id)
    accepted: list[dict] = []
    skipped: list[dict] = []
    for info in infos:
        if info.is_dir():
            continue
        try:
            data = archive.read(info)
        except (RuntimeError, zipfile.BadZipFile, OSError):
            skipped.append({"path": info.filename, "reason": "unreadable"})
            continue
        entry, skip = await _accept_code_file(kb_id, background_tasks, existing, info.filename, data, None)
        if entry is not None:
            accepted.append(entry)
        if skip is not None:
            skipped.append(skip)
    return {"total": len(infos), "accepted": accepted, "skipped": skipped}


@app.post("/api/v1/code/uploads")
async def upload_code_files(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    paths: list[str] = Form(default=[]),
) -> dict:
    """递归接收文件夹上传：files 为文件，paths 为对应的相对路径。"""
    return await _upload_code_files_for_kb(DEFAULT_KB_ID, background_tasks, files, paths)


@app.post("/api/v1/code/zip")
async def upload_code_zip(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> dict:
    """解压 ZIP 后递归入库其中的代码文件。"""
    return await _upload_code_zip_for_kb(DEFAULT_KB_ID, background_tasks, file)


async def _index_document_local(document_id: UUID, version_id: UUID, title: str, content: bytes, content_type: str | None, kb_id: UUID = DEFAULT_KB_ID) -> None:
    source_kind = detect_source_kind(title, content_type)
    try:
        parsed = parse_document(title, content, content_type)
        chunker = CodeChunker() if source_kind == "code" else state().chunker
        chunks = chunker.split(parsed.text, document_id, version_id, title, state().settings.pipeline_revision, parsed)
        chunks = [chunk.model_copy(update={"kb_id": kb_id, "source_kind": source_kind}) for chunk in chunks]
        chunks = await _ensure_embeddings(chunks)
        state().retriever.add(chunks)
        state().documents[document_id].update({
            "format": parsed.format, "source_kind": source_kind, "status": "ready", "chunk_count": len(chunks), "error": None,
            "glossary_state": detect_glossary_state(chunks),
        })
        await app.state.catalog.replace_paragraphs(document_id, chunks)
        await app.state.catalog.update_document(document_id, state().documents[document_id])
    except DocumentParseError as exc:
        state().documents[document_id].update({"status": "failed", "error": str(exc), "chunk_count": 0})


@app.post("/internal/v1/documents/{document_id}/index")
async def index_callback(document_id: UUID, request: IndexCallbackRequest) -> dict:
    document = state().documents.get(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="document_not_found")
    if request.error:
        document.update({"format": request.format, "source_kind": request.source_kind, "status": "failed", "error": request.error, "chunk_count": 0, "task_id": request.task_id})
        await app.state.catalog.update_document(document_id, document)
        return {"status": "failed", "document_id": str(document_id)}
    kb_id = document.get("kb_id", DEFAULT_KB_ID)
    chunks = [chunk.model_copy(update={"kb_id": kb_id}) for chunk in request.chunks]
    chunks = await _ensure_embeddings(chunks)
    state().retriever.add(chunks)
    document.update({
        "format": request.format,
        "source_kind": chunks[0].source_kind if chunks else request.source_kind,
        "status": "ready", "error": None, "chunk_count": len(chunks),
        "task_id": request.task_id, "glossary_state": detect_glossary_state(chunks),
    })
    await app.state.catalog.replace_paragraphs(document_id, chunks)
    await app.state.catalog.update_document(document_id, document)
    return {"status": "ready", "document_id": str(document_id), "chunk_count": len(chunks)}


@app.post("/api/v1/documents/{document_id}/glossary")
async def build_document_glossary(
    document_id: UUID,
    strategy: Literal["rules", "ai_assisted"] = Query("rules"),
    progress_id: str = Query("", max_length=64),
) -> dict:
    return await _build_document_glossary(document_id, strategy, progress_id, DEFAULT_KB_ID)


async def _build_document_glossary(
    document_id: UUID,
    strategy: Literal["rules", "ai_assisted"],
    progress_id: str,
    expected_kb_id: UUID,
) -> dict:
    document = state().documents.get(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="document_not_found")
    document_kb_id = document.get("kb_id", DEFAULT_KB_ID)
    if document_kb_id != (expected_kb_id or DEFAULT_KB_ID):
        raise HTTPException(status_code=404, detail="document_not_found")
    chunks = state().retriever.ordered_document_chunks(document_id)
    current_version_id = document.get("version_id")
    if current_version_id:
        chunks = [chunk for chunk in chunks if str(chunk.version_id) == str(current_version_id)]
    candidates = glossary_candidates(chunks, chunks)
    assistant = None
    extraction_audit: list[dict] = []
    if strategy == "ai_assisted":
        settings = state().settings

        def report_glossary(stage: str, message: str) -> None:
            if progress_id:
                state().progress.set(progress_id, stage, message)

        existing_entries = [
            entry for entry in state().glossary.entries
            if entry.document_id == document_id and str(entry.version_id) == str(document.get("version_id"))
            and (entry.kb_id is None or entry.kb_id == document_kb_id)
        ]
        assistant = await build_glossary_with_ai_assistance(
            settings.effective_ai_base_url,
            settings.effective_ai_model,
            document.get("title", ""),
            chunks,
            state().retriever,
            state().embeddings,
            settings.glossary_timeout_seconds,
            provider=settings.effective_ai_provider,
            api_key=settings.ai_api_key,
            max_rounds=settings.iterative_max_rounds,
            neighbor_max_count=settings.neighbor_max_count,
            known_entries=existing_entries,
            on_progress=report_glossary,
        )
        chunks_by_id = {str(chunk.chunk_id): chunk for chunk in chunks}
        candidates = [chunks_by_id[chunk_id] for chunk_id in assistant.selected_ids if chunk_id in chunks_by_id]
        entries = assistant.entries
    else:
        entries = []

    candidate_count = len(candidates)
    if strategy == "rules" and not candidates:
        existing_count = await app.state.catalog.count_glossary(document_id, kb_id=document_kb_id)
        document["glossary_state"] = GLOSSARY_COLLECTED if existing_count else GLOSSARY_NONE
        document["glossary_count"] = existing_count
        await app.state.catalog.update_document(document_id, document)
        if progress_id:
            state().progress.set(progress_id, "done", "规则检测完成：没有候选段落")
        return {
            "document_id": str(document_id), "glossary_state": document["glossary_state"],
            "strategy": strategy, "candidate_count": 0, "glossary_count": existing_count, "entries": [],
            "extraction_status": "no_candidates", "extraction_audit": [], "warnings": [],
            "search_terms": [], "rounds": [],
        }
    if strategy == "rules":
        settings = state().settings
        if progress_id:
            state().progress.set(progress_id, "extraction", f"按规则分析 {len(candidates)} 个候选段落")
        try:
            entries = await extract_glossary(
                settings.effective_ai_base_url,
                settings.effective_ai_model,
                candidates,
                settings.glossary_timeout_seconds,
                provider=settings.effective_ai_provider,
                api_key=settings.ai_api_key,
                audit=extraction_audit,
            )
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise HTTPException(status_code=503, detail=f"glossary_extraction_unavailable: {exc}") from exc
    version_id = document["version_id"]
    version_id = version_id if isinstance(version_id, UUID) else UUID(str(version_id))
    entries = [
        entry.model_copy(update={"kb_id": document_kb_id, "document_id": document_id, "version_id": version_id})
        for entry in entries
    ]
    if not entries:
        # Empty model output is not evidence that the document has no glossary.
        # Keep it retryable and preserve any previously collected terms.
        existing_count = await app.state.catalog.count_glossary(document_id, kb_id=document_kb_id)
        document["glossary_state"] = GLOSSARY_COLLECTED if existing_count else GLOSSARY_PENDING
        document["glossary_count"] = existing_count
        await app.state.catalog.update_document(document_id, document)
        extraction_audit = assistant.extraction_audit if assistant else extraction_audit
        extraction_status = "failed" if extraction_audit and all(item.get("error_code") for item in extraction_audit) else "empty"
        if extraction_status == "empty" and any(item.get("error_code") for item in extraction_audit):
            extraction_status = "partial_failure"
        response = {
            "document_id": str(document_id), "glossary_state": document["glossary_state"],
            "strategy": strategy,
            "candidate_count": candidate_count,
            "rule_candidate_count": assistant.retrieval_trace.get("rule_candidate_count", candidate_count) if assistant else candidate_count,
            "glossary_count": existing_count,
            "entries": [],
            "extraction_status": extraction_status,
            "search_terms": assistant.search_terms if assistant else [],
            "retrieval_trace": assistant.retrieval_trace if assistant else {},
            "rounds": assistant.rounds if assistant else [],
            "extraction_audit": extraction_audit,
            "warnings": assistant.warnings if assistant else [],
        }
        if progress_id:
            state().progress.set(progress_id, "done", "术语提取流程完成", result=response)
        return response
    newly_extracted = entries
    if assistant:
        existing_entries = [
            entry for entry in state().glossary.entries
            if entry.document_id == document_id and str(entry.version_id) == str(version_id)
            and (entry.kb_id is None or entry.kb_id == document_kb_id)
        ]
        seen_pairs = {(entry.term.casefold(), entry.expansion.casefold()) for entry in existing_entries}
        entries = [*existing_entries, *[
            entry for entry in newly_extracted
            if (entry.term.casefold(), entry.expansion.casefold()) not in seen_pairs
        ]]
    await app.state.catalog.replace_glossary(document_id, version_id, entries)
    document["glossary_state"] = GLOSSARY_COLLECTED
    document["glossary_count"] = len(entries)
    await app.state.catalog.update_document(document_id, document)
    state().glossary.replace(await app.state.catalog.load_glossary())
    extraction_audit = assistant.extraction_audit if assistant else extraction_audit
    response = {
        "document_id": str(document_id),
        "glossary_state": document["glossary_state"],
        "candidate_count": candidate_count,
        "glossary_count": len(entries),
        "entries": [entry.model_dump(mode="json") for entry in newly_extracted],
        "strategy": strategy,
        "extraction_status": "partial_failure" if any(item.get("error_code") for item in extraction_audit) else "completed",
        "extraction_audit": extraction_audit,
        "warnings": assistant.warnings if assistant else [],
    }
    if assistant:
        response.update({
            "rule_candidate_count": assistant.retrieval_trace.get("rule_candidate_count", 0),
            "search_terms": assistant.search_terms,
            "queries": assistant.queries,
            "retrieval_trace": assistant.retrieval_trace,
            "rounds": assistant.rounds,
            "extraction_audit": assistant.extraction_audit,
            "warnings": assistant.warnings,
        })
    if progress_id:
        state().progress.set(progress_id, "done", "术语提取流程完成", result=response)
    return response


async def _delete_glossary_entry(document_id: UUID, term: str, expansion: str, expected_kb_id: UUID) -> dict:
    """Remove one term/expansion pair and keep document counts and index in sync."""
    document = state().documents.get(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="document_not_found")
    document_kb_id = document.get("kb_id", DEFAULT_KB_ID)
    if document_kb_id != (expected_kb_id or DEFAULT_KB_ID):
        raise HTTPException(status_code=404, detail="document_not_found")
    deleted = await app.state.catalog.delete_glossary_entry(document_id, term, expansion, document_kb_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="glossary_entry_not_found")
    state().glossary.replace(await app.state.catalog.load_glossary())
    remaining = await app.state.catalog.count_glossary(document_id, kb_id=document_kb_id)
    if remaining:
        document["glossary_state"] = GLOSSARY_COLLECTED
    else:
        # 删空后回到可重试状态，而不是宣称文档没有术语段。
        chunks = state().retriever.ordered_document_chunks(document_id)
        version_id = document.get("version_id")
        if version_id:
            chunks = [chunk for chunk in chunks if str(chunk.version_id) == str(version_id)]
        document["glossary_state"] = detect_glossary_state(chunks)
    document["glossary_count"] = remaining
    await app.state.catalog.update_document(document_id, document)
    return {
        "document_id": str(document_id), "deleted": deleted,
        "glossary_count": remaining, "glossary_state": document["glossary_state"],
    }


@app.delete("/api/v1/documents/{document_id}/glossary")
async def delete_document_glossary_entry(
    document_id: UUID,
    term: str = Query(..., min_length=1, max_length=200),
    expansion: str = Query(..., min_length=1, max_length=2000),
) -> dict:
    return await _delete_glossary_entry(document_id, term, expansion, DEFAULT_KB_ID)


@app.get("/api/v1/progress/{progress_id}")
async def retrieval_progress(progress_id: str) -> dict:
    return state().progress.get(progress_id) or {"stage": "unknown"}


@app.get("/api/v1/progress/{progress_id}/stream")
async def retrieval_progress_stream(progress_id: str, request: Request, after_revision: int = Query(default=0, ge=0)) -> StreamingResponse:
    progress = state().progress
    last_event_id = request.headers.get("last-event-id")
    try:
        revision = max(0, int(last_event_id)) if last_event_id is not None else after_revision
    except ValueError:
        revision = after_revision

    async def events():
        nonlocal revision
        while not await request.is_disconnected():
            info = await progress.wait_for_update(progress_id, revision)
            if info is None:
                yield ": keep-alive\n\n"
                continue

            next_revision = int(info.get("revision", revision))
            event = dict(info)
            event["history"] = [item for item in info.get("history", []) if int(item.get("revision", 0)) > revision]
            if int(info.get("result_revision", 0)) <= revision:
                event.pop("result", None)
            payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
            yield f"id: {next_revision}\nevent: progress\ndata: {payload}\n\n"
            revision = next_revision
            if info.get("stage") in {"done", "error"}:
                return

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _slugify_knowledge_base(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "-", value.strip().lower()).strip("-")
    return slug[:80] or "knowledge-base"


def _scoped_request(request_model, kb_id: UUID):
    return request_model.model_copy(update={"kb_ids": [kb_id]})


def _progress_for_kb(kb_id: UUID, progress_id: str) -> dict:
    item = state().progress.get(progress_id)
    if item is None or item.get("kb_ids") != [str(kb_id)]:
        raise HTTPException(status_code=404, detail="progress_not_found")
    return item


@app.get("/api/v2/knowledge-bases", response_model=list[KnowledgeBaseResponse])
async def list_knowledge_bases(request: Request) -> list[KnowledgeBaseResponse]:
    principal = resolve_principal(request, state().settings)
    rows = await app.state.catalog.list_knowledge_bases(principal.user_id)
    return [KnowledgeBaseResponse.model_validate(row) for row in rows]


@app.post("/api/v2/knowledge-bases", response_model=KnowledgeBaseResponse, status_code=201)
async def create_knowledge_base(request: Request, payload: KnowledgeBaseCreate) -> KnowledgeBaseResponse:
    principal = resolve_principal(request, state().settings)
    slug = _slugify_knowledge_base(payload.slug or payload.name)
    kb_id = uuid4()
    try:
        row = await app.state.catalog.create_knowledge_base(kb_id, payload.name.strip(), slug, payload.description, principal.user_id)
    except asyncpg.UniqueViolationError as exc:
        raise HTTPException(status_code=409, detail="knowledge_base_slug_exists") from exc
    row["role"] = "owner"
    return KnowledgeBaseResponse.model_validate(row)


@app.get("/api/v2/knowledge-bases/{kb_id}/members", response_model=list[KnowledgeBaseMember])
async def list_knowledge_base_members(kb_id: UUID, request: Request) -> list[KnowledgeBaseMember]:
    await require_kb_access(request, kb_id)
    return [KnowledgeBaseMember.model_validate(row) for row in await app.state.catalog.list_members(kb_id)]


@app.post("/api/v2/knowledge-bases/{kb_id}/members", response_model=KnowledgeBaseMember)
async def upsert_knowledge_base_member(kb_id: UUID, member: KnowledgeBaseMember, request: Request) -> KnowledgeBaseMember:
    principal = await require_kb_admin(request, kb_id)
    if member.role == "owner" and await app.state.catalog.get_membership_role(kb_id, principal.user_id) != "owner":
        raise HTTPException(status_code=403, detail="owner_role_required")
    await app.state.catalog.upsert_membership(kb_id, member.user_id, member.role)
    return member


@app.delete("/api/v2/knowledge-bases/{kb_id}/members/{user_id}")
async def remove_knowledge_base_member(kb_id: UUID, user_id: UUID, request: Request) -> dict:
    await require_kb_owner(request, kb_id)
    if user_id == resolve_principal(request, state().settings).user_id:
        raise HTTPException(status_code=409, detail="owner_cannot_remove_self")
    await app.state.catalog.remove_membership(kb_id, user_id)
    return {"status": "removed", "kb_id": str(kb_id), "user_id": str(user_id)}


@app.get("/api/v2/knowledge-bases/{kb_id}/documents")
async def list_v2_documents(
    kb_id: UUID,
    request: Request,
    limit: int | None = Query(None, ge=1, le=200),
    offset: int = Query(0, ge=0),
    source_kind: Literal["document", "code"] | None = Query(None),
) -> list[dict] | dict:
    await require_kb_access(request, kb_id)
    items = [{"document_id": str(document_id), **value, "source_kind": value.get("source_kind", "document")}
             for document_id, value in state().documents.items()
             if value.get("kb_id", DEFAULT_KB_ID) == kb_id
             and (source_kind is None or value.get("source_kind", "document") == source_kind)]
    items.sort(key=lambda item: (item.get("created_at", ""), item["document_id"]), reverse=True)
    if limit is None:
        return items
    return {"total": len(items), "limit": limit, "offset": offset, "documents": items[offset:offset + limit]}


@app.post("/api/v2/knowledge-bases/{kb_id}/documents", response_model=DocumentResponse, status_code=202)
async def upload_v2_document(kb_id: UUID, request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> DocumentResponse:
    await require_kb_access(request, kb_id, "editor")
    return await _upload_document_for_kb(kb_id, background_tasks, file)


@app.post("/api/v2/knowledge-bases/{kb_id}/code/uploads")
async def upload_v2_code_files(
    kb_id: UUID,
    request: Request,
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    paths: list[str] = Form(default=[]),
) -> dict:
    await require_kb_access(request, kb_id, "editor")
    return await _upload_code_files_for_kb(kb_id, background_tasks, files, paths)


@app.post("/api/v2/knowledge-bases/{kb_id}/code/zip")
async def upload_v2_code_zip(
    kb_id: UUID,
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> dict:
    await require_kb_access(request, kb_id, "editor")
    return await _upload_code_zip_for_kb(kb_id, background_tasks, file)


@app.post("/api/v2/knowledge-bases/{kb_id}/search")
async def search_v2(kb_id: UUID, request: Request, payload: SearchRequest) -> dict:
    await require_kb_access(request, kb_id)
    return await _search_scoped(_scoped_request(payload, kb_id))


@app.get("/api/v2/knowledge-bases/{kb_id}/progress/{progress_id}")
async def retrieval_progress_v2(kb_id: UUID, progress_id: str, request: Request) -> dict:
    await require_kb_access(request, kb_id)
    return _progress_for_kb(kb_id, progress_id)


@app.get("/api/v2/knowledge-bases/{kb_id}/progress/{progress_id}/stream")
async def retrieval_progress_stream_v2(kb_id: UUID, progress_id: str, request: Request, after_revision: int = Query(default=0, ge=0)) -> StreamingResponse:
    await require_kb_access(request, kb_id)
    _progress_for_kb(kb_id, progress_id)
    return await retrieval_progress_stream(progress_id, request, after_revision)


@app.post("/api/v2/knowledge-bases/{kb_id}/answers", response_model=AnswerResponse)
async def answer_v2(kb_id: UUID, request: Request, payload: AnswerRequest) -> AnswerResponse:
    await require_kb_access(request, kb_id)
    return await _answer_scoped(_scoped_request(payload, kb_id), kb_id)


@app.post("/api/v2/knowledge-bases/{kb_id}/answers/stream")
async def answer_stream_v2(kb_id: UUID, request: Request, payload: AnswerRequest) -> StreamingResponse:
    await require_kb_access(request, kb_id)
    return await _answer_stream_scoped(_scoped_request(payload, kb_id), kb_id)


@app.post("/api/v2/knowledge-bases/{kb_id}/answers/{answer_id}/feedback")
async def submit_v2_feedback(kb_id: UUID, answer_id: UUID, feedback: FeedbackRequest, request: Request) -> dict:
    await require_kb_access(request, kb_id)
    if kb_id not in await app.state.feedback.answer_kb_ids(answer_id):
        raise HTTPException(status_code=404, detail="answer_not_found")
    return await submit_feedback(answer_id, feedback)


async def _conversation_list(kb_id: UUID, limit: int, offset: int) -> dict:
    try:
        total, items = await app.state.conversations.list(kb_id, limit, offset)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="conversation_storage_unavailable") from exc
    return {"kb_id": str(kb_id), "total": total, "limit": limit, "offset": offset, "conversations": items}


async def _conversation_detail(kb_id: UUID, conversation_id: UUID) -> dict:
    try:
        conversation = await app.state.conversations.get(kb_id, conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="conversation_not_found")
        messages = await app.state.conversations.messages(kb_id, conversation_id) or []
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail="conversation_storage_unavailable") from exc
    return {"conversation": conversation, "messages": messages}


async def _conversation_create(kb_id: UUID, payload: ConversationCreate, user_id: UUID | None) -> dict:
    try:
        return await app.state.conversations.create(kb_id, user_id, payload.title)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="conversation_storage_unavailable") from exc


async def _conversation_rename(kb_id: UUID, conversation_id: UUID, payload: ConversationRename) -> dict:
    try:
        conversation = await app.state.conversations.rename(kb_id, conversation_id, payload.title)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="conversation_storage_unavailable") from exc
    if conversation is None:
        raise HTTPException(status_code=404, detail="conversation_not_found")
    return conversation


async def _conversation_delete(kb_id: UUID, conversation_id: UUID) -> dict:
    try:
        deleted = await app.state.conversations.delete(kb_id, conversation_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="conversation_storage_unavailable") from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="conversation_not_found")
    return {"status": "deleted", "conversation_id": str(conversation_id)}


@app.get("/api/v2/knowledge-bases/{kb_id}/conversations")
async def list_v2_conversations(
    kb_id: UUID,
    request: Request,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> dict:
    await require_kb_access(request, kb_id)
    return await _conversation_list(kb_id, limit, offset)


@app.post("/api/v2/knowledge-bases/{kb_id}/conversations", status_code=201)
async def create_v2_conversation(kb_id: UUID, request: Request, payload: ConversationCreate) -> dict:
    principal = await require_kb_access(request, kb_id)
    return await _conversation_create(kb_id, payload, principal.user_id)


@app.get("/api/v2/knowledge-bases/{kb_id}/conversations/{conversation_id}")
async def read_v2_conversation(kb_id: UUID, conversation_id: UUID, request: Request) -> dict:
    await require_kb_access(request, kb_id)
    return await _conversation_detail(kb_id, conversation_id)


@app.patch("/api/v2/knowledge-bases/{kb_id}/conversations/{conversation_id}")
async def rename_v2_conversation(
    kb_id: UUID,
    conversation_id: UUID,
    request: Request,
    payload: ConversationRename,
) -> dict:
    await require_kb_access(request, kb_id)
    return await _conversation_rename(kb_id, conversation_id, payload)


@app.delete("/api/v2/knowledge-bases/{kb_id}/conversations/{conversation_id}")
async def delete_v2_conversation(kb_id: UUID, conversation_id: UUID, request: Request) -> dict:
    await require_kb_access(request, kb_id)
    return await _conversation_delete(kb_id, conversation_id)


@app.get("/api/v2/knowledge-bases/{kb_id}/glossary")
async def list_v2_glossary(kb_id: UUID, request: Request, limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)) -> dict:
    await require_kb_access(request, kb_id)
    total = await app.state.catalog.count_glossary(kb_id=kb_id)
    entries = await app.state.catalog.list_glossary(kb_id=kb_id, limit=limit, offset=offset)
    return {"kb_id": str(kb_id), "total": total, "limit": limit, "offset": offset, "entries": entries}


@app.post("/api/v2/knowledge-bases/{kb_id}/documents/{document_id}/glossary")
async def build_v2_document_glossary(kb_id: UUID, document_id: UUID, request: Request, strategy: Literal["rules", "ai_assisted"] = Query("rules"), progress_id: str = Query("", max_length=64)) -> dict:
    await require_kb_access(request, kb_id, "editor")
    record = await app.state.catalog.get_document_source(document_id)
    if not record or record.get("kb_id") != kb_id:
        raise HTTPException(status_code=404, detail="document_not_found")
    return await _build_document_glossary(document_id, strategy, progress_id, kb_id)


@app.get("/api/v2/knowledge-bases/{kb_id}/documents/{document_id}/glossary")
async def v2_document_glossary(kb_id: UUID, document_id: UUID, request: Request) -> dict:
    await require_kb_access(request, kb_id)
    record = await app.state.catalog.get_document_source(document_id)
    if not record or record.get("kb_id") != kb_id:
        raise HTTPException(status_code=404, detail="document_not_found")
    entries = await app.state.catalog.list_glossary(document_id=document_id, kb_id=kb_id)
    document = state().documents.get(document_id, {})
    return {"kb_id": str(kb_id), "document_id": str(document_id), "title": record["title"], "glossary_state": document.get("glossary_state", 0), "count": len(entries), "entries": entries}


@app.delete("/api/v2/knowledge-bases/{kb_id}/documents/{document_id}/glossary")
async def delete_v2_document_glossary_entry(
    kb_id: UUID,
    document_id: UUID,
    request: Request,
    term: str = Query(..., min_length=1, max_length=200),
    expansion: str = Query(..., min_length=1, max_length=2000),
) -> dict:
    await require_kb_access(request, kb_id, "editor")
    record = await app.state.catalog.get_document_source(document_id)
    if not record or record.get("kb_id") != kb_id:
        raise HTTPException(status_code=404, detail="document_not_found")
    return await _delete_glossary_entry(document_id, term, expansion, kb_id)


@app.get("/api/v2/knowledge-bases/{kb_id}/documents/{document_id}/content")
async def v2_document_content(kb_id: UUID, document_id: UUID, request: Request) -> FileResponse:
    await require_kb_access(request, kb_id)
    record = await app.state.catalog.get_document_source(document_id)
    if not record or record.get("kb_id") != kb_id:
        raise HTTPException(status_code=404, detail="document_not_found")
    return await _document_content_for_kb(document_id, kb_id)


@app.get("/api/v2/knowledge-bases/{kb_id}/documents/{document_id}/reader-text")
async def v2_document_reader_text(kb_id: UUID, document_id: UUID, request: Request) -> dict:
    await require_kb_access(request, kb_id)
    record = await app.state.catalog.get_document_source(document_id)
    if not record or record.get("kb_id") != kb_id:
        raise HTTPException(status_code=404, detail="document_not_found")
    return await _document_reader_text_for_kb(document_id, kb_id)


@app.get("/api/v2/knowledge-bases/{kb_id}/export")
async def export_v2_knowledge_base(kb_id: UUID, request: Request) -> StreamingResponse:
    await require_kb_admin(request, kb_id)
    bundle = await app.state.catalog.export_knowledge_base(kb_id)
    if not bundle:
        raise HTTPException(status_code=404, detail="knowledge_base_not_found")
    knowledge_base = bundle["knowledge_base"]
    export_id = uuid4()
    manifest = {
        "schema_version": 1, "export_id": str(export_id), "kb_id": str(kb_id),
        "slug": knowledge_base["slug"], "exported_at": datetime.now(timezone.utc).isoformat(),
        "document_count": len(bundle["documents"]), "chunk_count": len(bundle["chunks"]),
        "glossary_count": len(bundle["glossary"]), "includes_embeddings": True,
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
        archive.writestr("knowledge_base.json", json.dumps(knowledge_base, ensure_ascii=False, indent=2, default=str))
        archive.writestr("documents.jsonl", "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in bundle["documents"]))
        archive.writestr("chunks.jsonl", "".join(json.dumps({**row, "payload": json.loads(row["payload"]) if isinstance(row["payload"], str) else row["payload"]}, ensure_ascii=False, default=str) + "\n" for row in bundle["chunks"]))
        archive.writestr("glossary.jsonl", "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in bundle["glossary"]))
        for document in bundle["documents"]:
            path = _resolve_source_path(document)
            if path is not None:
                archive.write(path, f"source/{document['document_id']}/{document['version_id']}/original{path.suffix}")
    output.seek(0)
    return StreamingResponse(output, media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="knowledge-base-{knowledge_base["slug"]}-{export_id}.zip"'})


@app.delete("/api/v2/knowledge-bases/{kb_id}/documents")
async def clear_v2_knowledge_base_documents(kb_id: UUID, request: Request, payload: ClearKnowledgeBaseRequest) -> dict:
    await require_kb_admin(request, kb_id)
    count, source_paths = await app.state.catalog.clear_knowledge_base_documents(kb_id)
    for document_id, document in list(state().documents.items()):
        if document.get("kb_id", DEFAULT_KB_ID) == kb_id:
            state().documents.pop(document_id, None)
            state().retriever.remove_document_chunks(document_id)
    deleted_files = 0
    for source_path in source_paths:
        path = Path(source_path)
        if path.is_file() and Path(state().settings.ingestion_dir).resolve() in path.resolve().parents:
            path.unlink()
            deleted_files += 1
    return {"status": "cleared", "kb_id": str(kb_id), "documents": count, "files": deleted_files}


@app.delete("/api/v2/knowledge-bases/{kb_id}")
async def delete_v2_knowledge_base(kb_id: UUID, request: Request, payload: ClearKnowledgeBaseRequest) -> dict:
    await require_kb_owner(request, kb_id)
    if kb_id == DEFAULT_KB_ID:
        raise HTTPException(status_code=409, detail="default_knowledge_base_cannot_be_deleted")
    count, source_paths = await app.state.catalog.delete_knowledge_base(kb_id)
    for document_id, document in list(state().documents.items()):
        if document.get("kb_id", DEFAULT_KB_ID) == kb_id:
            state().documents.pop(document_id, None)
            state().retriever.remove_document_chunks(document_id)
    for source_path in source_paths:
        path = Path(source_path)
        if path.is_file() and Path(state().settings.ingestion_dir).resolve() in path.resolve().parents:
            path.unlink()
    return {"status": "deleted", "kb_id": str(kb_id), "documents": count}


@app.post("/api/v1/search")
async def search(request: SearchRequest) -> dict:
    return await _search_scoped(_scoped_request(request, DEFAULT_KB_ID))


async def _search_scoped(request: SearchRequest) -> dict:
    _seed_progress(request, request.kb_ids[0] if request.kb_ids else DEFAULT_KB_ID)
    report = _progress_reporter(request.progress_id)
    result, _ = await _retrieve(request, report)
    return result.model_dump(mode="json")


def _generation_progress_message(style: str) -> str:
    return "基于证据生成发散型回答中" if style == DIVERGENT_STYLE else "基于证据生成回答中"


def _generation_temperature(style: str) -> float:
    if style != DIVERGENT_STYLE:
        return CONSERVATIVE_TEMPERATURE
    return getattr(state().generator, "divergent_temperature", DEFAULT_DIVERGENT_TEMPERATURE)


def _direct_response(request_id: UUID, request: AnswerRequest, analysis, text: str) -> AnswerResponse:
    """Envelope for an L1 no-retrieval turn: no claims, no sources, no evidence."""
    return AnswerResponse(
        request_id=request_id,
        status="no_retrieval_needed",
        answer=text,
        confidence=0.0,
        confidence_kind="no_retrieval",
        verification_level="citation_only",
        generation_style=request.generation_style,
        warnings=["本轮判定为无需检索，回答未引用知识库证据。"],
        query_analysis=analysis,
    )


async def _direct_answer(
    request: AnswerRequest,
    request_id: UUID,
    analysis,
    history: list[dict] | None,
) -> AnswerResponse:
    text = await state().generator.direct_reply(request.question, history=history)
    return _direct_response(request_id, request, analysis, text)


async def _finalize_answer(
    request: AnswerRequest,
    request_id: UUID,
    result: RetrievalResult,
    analysis,
    draft,
    dropped_ids: list[str],
    response: AnswerResponse,
    started: float,
    report,
    history_used: bool = False,
) -> AnswerResponse:
    if dropped_ids:
        report(
            "generation",
            f"上下文预算不足，已从生成上下文丢弃最不相干的 {len(dropped_ids)} 条证据（{'、'.join(dropped_ids)}），证据仍保留在检索结果中",
        )
    report("done", "回答完成")
    context_warning = (
        [f"上下文预算不足（{state().settings.max_context_chars} 字符），已从生成上下文丢弃最不相干的 {len(dropped_ids)} 条证据：{'、'.join(dropped_ids)}。证据仍保留在检索结果中。"]
        if dropped_ids else []
    )
    response = response.model_copy(update={
        "degraded": result.degraded,
        "query_analysis": analysis,
        "warnings": [*response.warnings, *context_warning, *result.warnings],
    })
    progress_snapshot = state().progress.get(request.progress_id) if request.progress_id else None
    progress_models = (progress_snapshot or {}).get("model_provenance") or {}
    retrieval_chat = progress_models.get("retrieval_chat") or _chat_model_provenance(state().query_planner.client)
    generation_chat = (progress_snapshot or {}).get("generation_chat")
    if draft is not None and not generation_chat:
        generation_chat = _chat_model_provenance(state().generator.client)
    embedding_model = progress_models.get("embedding") or _embedding_model_provenance(state().embeddings)
    primary_chat = generation_chat or retrieval_chat
    progress_timing = _progress_timing_snapshot(progress_snapshot, request.progress_id)
    snapshot = {
        "schema_version": 2,
        "input": request.model_dump(mode="json"),
        "output": response.model_dump(mode="json"),
        "retrieval": result.model_dump(mode="json", exclude={"raw_error_text"}),
        "metadata": {
            "git_tag": state().settings.git_tag or None,
            "git_commit": state().settings.git_commit or None,
            "model": primary_chat["model"] if request.mode == "grounded" and result.evidence else None,
            "model_provider": primary_chat["provider"] if request.mode == "grounded" and result.evidence else None,
            "model_provenance": {
                "chat": {"retrieval": retrieval_chat, "generation": generation_chat},
                "embedding": embedding_model,
            },
            "progress_timing": progress_timing,
            "pipeline_revision": state().settings.pipeline_revision,
            "prompt_revision": (
                ("divergent-zh-v1" if request.generation_style == DIVERGENT_STYLE else "grounded-zh-v1")
                + ("-thread-v1" if history_used else "")
            ),
            "generation_style": request.generation_style,
            "latency_ms": round((perf_counter() - started) * 1000),
            "temperature": _generation_temperature(request.generation_style),
            "num_predict": state().generator.effective_max_tokens,
            "max_context_chars": state().settings.max_context_chars,
            "verification_level": response.verification_level,
            "generation_draft": draft.model_dump(mode="json") if draft is not None else None,
            "context_dropped_evidence_ids": dropped_ids,
        },
    }
    try:
        await app.state.feedback.save_answer(response.answer_id, snapshot)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="answer_storage_unavailable") from exc
    return response


async def _answer_inputs(
    request: AnswerRequest,
    report,
    history: list[dict] | None = None,
) -> tuple[RetrievalResult, object]:
    # A cached retrieval belongs to a single question; follow-ups must run their
    # own retrieval against the rewritten question.
    result = None if history else _cached_retrieval(
        request.progress_id,
        request.question,
        request.kb_ids or [DEFAULT_KB_ID],
        list(request.source_kinds),
    )
    if result is not None and (result.query_analysis is not None or request.query_analysis is not None):
        analysis = result.query_analysis or request.query_analysis
        report("retrieval", "已复用首轮检索结果，开始生成回答", retrieval=result, query_analysis=analysis, question=request.question)
        return result, analysis
    return await _retrieve(request, report, history)


async def _begin_turn(
    request: AnswerRequest,
    kb_id: UUID,
) -> tuple[list[dict] | None, list[str]]:
    """Load prior turns and persist the new question of a continuing chat."""
    if request.conversation_id is None:
        return None, []
    store = app.state.conversations
    try:
        conversation = await store.get(kb_id, request.conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="conversation_not_found")
        history = await store.history(request.conversation_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail="conversation_storage_unavailable") from exc
    warnings: list[str] = []
    try:
        await store.append_user(kb_id, request.conversation_id, request.question)
    except Exception:  # noqa: BLE001 - 会话写入失败不应丢掉已经生成的答案
        logger.exception("failed to append user message conversation_id=%s", request.conversation_id)
        warnings.append("本轮提问未能写入会话记录，刷新后可能看不到这条提问。")
    return history, warnings


def _conversation_metadata(
    response: AnswerResponse,
    analysis,
    result: RetrievalResult,
    progress_events: list[dict] | None = None,
) -> dict:
    return {
        "status": response.status,
        "confidence": response.confidence,
        "verification_level": response.verification_level,
        "generation_style": response.generation_style,
        "claims": [claim.model_dump(mode="json") for claim in response.claims],
        "sources": response.sources,
        "suggested_actions": response.suggested_actions,
        "warnings": response.warnings,
        "evidence_count": len(result.evidence),
        "query_analysis": analysis.model_dump(mode="json") if analysis is not None else None,
        # 每轮检索过程随答案一起归档，历史会话的 AI 气泡可以逐条展开回看。
        "progress_events": list(progress_events or [])[-32:],
    }


async def _finish_turn(
    request: AnswerRequest,
    kb_id: UUID,
    response: AnswerResponse,
    analysis,
    result: RetrievalResult,
) -> AnswerResponse:
    if request.conversation_id is None:
        return response
    progress_events: list[dict] = []
    progress_key = (request.progress_id or "").strip()
    if progress_key:
        snapshot = state().progress.get(progress_key) or {}
        progress_events = list(snapshot.get("history") or [])
    try:
        await app.state.conversations.append_assistant(
            kb_id,
            request.conversation_id,
            response.answer,
            response.answer_id,
            _conversation_metadata(response, analysis, result, progress_events),
        )
    except Exception:  # noqa: BLE001 - 答案已生成并校验，会话归档失败只告警
        logger.exception("failed to append assistant message conversation_id=%s", request.conversation_id)
        response = response.model_copy(update={
            "warnings": [*response.warnings, "本轮回答未能写入会话记录，刷新后可能看不到这条回答。"],
        })
    return response.model_copy(update={"conversation_id": request.conversation_id})


@app.post("/api/v1/answers", response_model=AnswerResponse)
async def answer(request: AnswerRequest) -> AnswerResponse:
    return await _answer_scoped(_scoped_request(request, DEFAULT_KB_ID), DEFAULT_KB_ID)


async def _answer_scoped(request: AnswerRequest, kb_id: UUID = DEFAULT_KB_ID) -> AnswerResponse:
    started = perf_counter()
    _seed_progress(request, kb_id)
    report = _progress_reporter(request.progress_id)
    history, conversation_warnings = await _begin_turn(request, kb_id)
    result, analysis = await _answer_inputs(request, report, history)
    request_id = uuid4()
    dropped_ids: list[str] = []
    draft = None
    style = request.generation_style
    if result.skipped:
        report("generation", "判定为无需检索，直接回答")
        try:
            response = await _direct_answer(request, request_id, analysis, history)
        except OllamaGenerationError as exc:
            report("error", "生成服务不可用", rawErrorOutput=exc.raw_error_output)
            detail = {"code": "generation_unavailable", "message": str(exc)}
            if exc.raw_error_output:
                detail["rawErrorOutput"] = exc.raw_error_output
            raise HTTPException(status_code=503, detail=detail) from exc
    elif request.mode == "extractive" or not result.evidence:
        response = state().validator.validate(request_id, request.question, result.evidence, request.mode)
    else:
        try:
            generator = state().generator
            report(
                "generation",
                _generation_progress_message(style),
                generation_chat=_chat_model_provenance(generator.client),
            )
            outcome = await generator.generate(
                request_id, request.question, result.evidence, analysis.glossary,
                style=style, history=history,
            )
            draft = outcome.draft
            dropped_ids = outcome.dropped_evidence_ids
        except OllamaGenerationError as exc:
            report("error", "生成服务不可用", rawErrorOutput=exc.raw_error_output)
            detail = {"code": "generation_unavailable", "message": str(exc)}
            if exc.raw_error_output:
                detail["rawErrorOutput"] = exc.raw_error_output
            raise HTTPException(status_code=503, detail=detail) from exc
        response = state().validator.validate_generated(request_id, draft, outcome.evidence_used, style=style)
    response = await _finalize_answer(
        request, request_id, result, analysis, draft, dropped_ids, response, started, report,
        history_used=bool(history),
    )
    response = await _finish_turn(request, kb_id, response, analysis, result)
    if conversation_warnings:
        response = response.model_copy(update={"warnings": [*conversation_warnings, *response.warnings]})
    return response


@app.post("/api/v1/answers/stream")
async def answer_stream(request: AnswerRequest) -> StreamingResponse:
    return await _answer_stream_scoped(_scoped_request(request, DEFAULT_KB_ID), DEFAULT_KB_ID)


async def _answer_stream_scoped(request: AnswerRequest, kb_id: UUID = DEFAULT_KB_ID) -> StreamingResponse:
    async def event_stream():
        started = perf_counter()
        report = _progress_reporter(request.progress_id)

        def event(name: str, payload: dict) -> str:
            return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"

        try:
            _seed_progress(request, kb_id)
            history, conversation_warnings = await _begin_turn(request, kb_id)
            result, analysis = await _answer_inputs(request, report, history)
            request_id = uuid4()
            dropped_ids: list[str] = []
            draft = None
            style = request.generation_style
            if result.skipped:
                generator = state().generator
                report(
                    "generation",
                    "判定为无需检索，直接回答",
                    generation_chat=_chat_model_provenance(generator.client),
                )
                try:
                    text = await generator.direct_reply(request.question, history=history)
                except OllamaGenerationError as exc:
                    report("error", "生成服务不可用", rawErrorOutput=exc.raw_error_output)
                    error_payload = {"message": str(exc)}
                    if exc.raw_error_output:
                        error_payload["rawErrorOutput"] = exc.raw_error_output
                    yield event("error", error_payload)
                    return
                yield event("preview", {"text": text})
                response = _direct_response(request_id, request, analysis, text)
            elif request.mode == "extractive" or not result.evidence:
                response = state().validator.validate(request_id, request.question, result.evidence, request.mode)
            else:
                generator = state().generator
                report(
                    "generation",
                    _generation_progress_message(style),
                    generation_chat=_chat_model_provenance(generator.client),
                )
                outcome = None
                try:
                    async for kind, value in generator.generate_stream(
                        request_id, request.question, result.evidence, analysis.glossary,
                        style=style, history=history,
                    ):
                        if kind == "preview":
                            yield event("preview", {"text": value})
                        else:
                            outcome = value
                except OllamaGenerationError as exc:
                    report("error", "生成服务不可用", rawErrorOutput=exc.raw_error_output)
                    error_payload = {"message": str(exc)}
                    if exc.raw_error_output:
                        error_payload["rawErrorOutput"] = exc.raw_error_output
                    yield event("error", error_payload)
                    return
                if outcome is None:
                    yield event("error", {"message": "模型未返回完整回答。"})
                    return
                draft = outcome.draft
                dropped_ids = outcome.dropped_evidence_ids
                response = state().validator.validate_generated(request_id, draft, outcome.evidence_used, style=style)
            response = await _finalize_answer(
                request, request_id, result, analysis, draft, dropped_ids, response, started, report,
                history_used=bool(history),
            )
            response = await _finish_turn(request, kb_id, response, analysis, result)
            if conversation_warnings:
                response = response.model_copy(update={"warnings": [*conversation_warnings, *response.warnings]})
            yield event("complete", response.model_dump(mode="json"))
        except HTTPException as exc:
            detail = exc.detail
            message = detail.get("message", str(detail)) if isinstance(detail, dict) else str(detail)
            error_payload = {"message": message}
            if isinstance(detail, dict) and detail.get("rawErrorOutput"):
                error_payload["rawErrorOutput"] = detail["rawErrorOutput"]
            yield event("error", error_payload)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


@app.post("/api/v1/answers/{answer_id}/feedback")
async def submit_feedback(answer_id: UUID, feedback: FeedbackRequest) -> dict:
    try:
        return await app.state.feedback.rate(answer_id, feedback)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="answer_not_found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail="feedback_id_conflict") from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="feedback_storage_unavailable") from exc


@app.get("/api/v1/conversations")
async def list_conversations(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> dict:
    return await _conversation_list(DEFAULT_KB_ID, limit, offset)


@app.post("/api/v1/conversations", status_code=201)
async def create_conversation(payload: ConversationCreate) -> dict:
    return await _conversation_create(DEFAULT_KB_ID, payload, None)


@app.get("/api/v1/conversations/{conversation_id}")
async def read_conversation(conversation_id: UUID) -> dict:
    return await _conversation_detail(DEFAULT_KB_ID, conversation_id)


@app.patch("/api/v1/conversations/{conversation_id}")
async def rename_conversation(conversation_id: UUID, payload: ConversationRename) -> dict:
    return await _conversation_rename(DEFAULT_KB_ID, conversation_id, payload)


@app.delete("/api/v1/conversations/{conversation_id}")
async def delete_conversation(conversation_id: UUID) -> dict:
    return await _conversation_delete(DEFAULT_KB_ID, conversation_id)


@app.get("/api/v1/documents")
async def list_documents(
    limit: int | None = Query(None, ge=1, le=200),
    offset: int = Query(0, ge=0),
    source_kind: Literal["document", "code"] | None = Query(None),
) -> list[dict] | dict:
    prototype = state()
    retriever = getattr(prototype, "retriever", None)
    items = [{
        "document_id": str(document_id),
        **value,
        "source_kind": value.get("source_kind", "document"),
        "glossary_candidate_count": len(glossary_candidates(
            [chunk for chunk in retriever.ordered_document_chunks(document_id)
             if chunk.search_state == "published"
             and (not value.get("version_id") or str(chunk.version_id) == str(value["version_id"]))],
        )) if retriever else 0,
    } for document_id, value in prototype.documents.items()
        if value.get("kb_id", DEFAULT_KB_ID) == DEFAULT_KB_ID
        and (source_kind is None or value.get("source_kind", "document") == source_kind)]
    items.reverse()
    if limit is None:
        return items
    return {"total": len(items), "limit": limit, "offset": offset, "documents": items[offset:offset + limit]}


@app.get("/api/v1/glossary")
async def list_glossary(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)) -> dict:
    total = await app.state.catalog.count_glossary()
    entries = await app.state.catalog.list_glossary(limit=limit, offset=offset)
    return {"total": total, "limit": limit, "offset": offset, "entries": entries}


@app.get("/api/v1/documents/{document_id}/glossary")
async def document_glossary(document_id: UUID) -> dict:
    record = await app.state.catalog.get_document_source(document_id)
    if not record or record.get("kb_id", DEFAULT_KB_ID) != DEFAULT_KB_ID:
        raise HTTPException(status_code=404, detail="document_not_found")
    entries = await app.state.catalog.list_glossary(document_id=document_id)
    document = state().documents.get(document_id, {})
    return {
        "document_id": str(document_id),
        "title": record["title"],
        "glossary_state": document.get("glossary_state", 0),
        "count": len(entries),
        "entries": entries,
    }


def _resolve_source_path(record: dict) -> Path | None:
    """Resolve the stored source file, rejecting legacy pseudo-paths and escapes."""
    source = record.get("source_path") or ""
    if "://" in source:
        return None
    path = Path(source).resolve()
    root = Path(state().settings.ingestion_dir).resolve()
    if root not in path.parents or not path.is_file():
        return None
    return path


@app.get("/api/v1/documents/{document_id}/content")
async def document_content(document_id: UUID) -> FileResponse:
    return await _document_content_for_kb(document_id, DEFAULT_KB_ID)


async def _document_content_for_kb(document_id: UUID, expected_kb_id: UUID) -> FileResponse:
    record = await app.state.catalog.get_document_source(document_id)
    if not record or record.get("kb_id", DEFAULT_KB_ID) != expected_kb_id or record["format"] not in {"pdf", "docx", "pptx", "text"}:
        raise HTTPException(status_code=404, detail="document_not_found")
    path = _resolve_source_path(record)
    if path is None:
        raise HTTPException(status_code=404, detail="document_content_unavailable")
    media_types = {"pdf": "application/pdf", "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                   "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation", "text": "text/plain; charset=utf-8"}
    return FileResponse(path, media_type=media_types[record["format"]], filename=record["title"], content_disposition_type="inline")


@app.get("/api/v1/documents/{document_id}/reader-text")
async def document_reader_text(document_id: UUID) -> dict:
    return await _document_reader_text_for_kb(document_id, DEFAULT_KB_ID)


async def _document_reader_text_for_kb(document_id: UUID, expected_kb_id: UUID) -> dict:
    record = await app.state.catalog.get_document_source(document_id)
    if not record or record.get("kb_id", DEFAULT_KB_ID) != expected_kb_id or record["format"] not in {"docx", "pptx", "text", "pdf"}:
        raise HTTPException(status_code=404, detail="document_not_found")
    path = _resolve_source_path(record)
    if path is not None:
        parsed = parse_document(record["title"], path.read_bytes(), None)
        text, source_available = parsed.text, True
    else:
        # The source may be missing (legacy migration or an earlier ingestion
        # run that deleted it). Fall back to the persisted paragraphs so the
        # reader still shows the indexed content.
        text = "\n\n".join(chunk.text for chunk in state().retriever.ordered_document_chunks(document_id))
        source_available = False
    if not text:
        raise HTTPException(status_code=404, detail="document_content_unavailable")
    return {"document_id": str(document_id), "version_id": str(record["version_id"]), "title": record["title"],
            "format": record["format"], "text": text, "source_available": source_available}


@app.delete("/api/v1/knowledge-base")
async def clear_knowledge_base(request: ClearKnowledgeBaseRequest) -> dict:
    # TODO(security): require authenticated administrator authorization and
    # write an auditable destructive-operation record before production use.
    document_count, source_paths = await app.state.catalog.clear_knowledge_base_documents(DEFAULT_KB_ID)
    document_ids = [document_id for document_id, document in state().documents.items() if document.get("kb_id", DEFAULT_KB_ID) == DEFAULT_KB_ID]
    for document_id in document_ids:
        state().documents.pop(document_id, None)
        state().retriever.remove_document_chunks(document_id)
    deleted_files = 0
    for source_path in source_paths:
        path = Path(source_path)
        if path.is_file() and Path(state().settings.ingestion_dir).resolve() in path.resolve().parents:
            path.unlink()
            deleted_files += 1
    return {
        "status": "cleared",
        "documents": document_count or len(document_ids),
        "files": deleted_files,
        "preserved": ["kb_answer_snapshots", "kb_human_feedback"],
    }
