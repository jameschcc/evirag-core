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
from pathlib import Path
from uuid import UUID

import httpx
from celery import Celery

from .chunking import CodeChunker, SemanticChunker
from .config import get_settings
from .contracts import Chunk, DEFAULT_KB_ID
from .document_parser import DocumentParseError, detect_source_kind, parse_document
from .embeddings import EmbeddingError, EmbeddingGateway

settings = get_settings()
celery_app = Celery("knowledge-base", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.update(task_acks_late=True, task_reject_on_worker_lost=True, worker_prefetch_multiplier=1)


def _embed_chunks(chunks: list[Chunk]) -> list[Chunk]:
    gateway = EmbeddingGateway(
        settings.effective_embedding_base_url,
        settings.embedding_model,
        settings.embedding_timeout_seconds,
        settings.embedding_batch_size,
        settings.effective_embedding_provider,
        settings.embedding_api_key,
    )
    if not gateway.available or not chunks:
        return chunks
    revision = gateway.revision_id
    missing = [chunk for chunk in chunks if not chunk.embedding or chunk.embedding_revision != revision]
    if not missing:
        return chunks
    try:
        vectors = asyncio.run(gateway.embed_texts([chunk.text for chunk in missing]))
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


@celery_app.task(name="kb.healthcheck")
def healthcheck() -> dict[str, str]:
    return {"status": "ok", "service": "ingestion-worker"}


def _notify_api(document_id: str, payload: dict) -> None:
    response = httpx.post(
        f"{settings.api_internal_url}/internal/v1/documents/{document_id}/index",
        json=payload,
        timeout=settings.request_timeout_seconds,
    )
    response.raise_for_status()


@celery_app.task(bind=True, name="kb.index_document", max_retries=3)
def index_document(
    task,
    document_id: str,
    version_id: str,
    title: str,
    content_path: str,
    content_type: str | None = None,
    kb_id: str = str(DEFAULT_KB_ID),
) -> dict[str, str | int]:
    path = Path(content_path)
    try:
        parsed = parse_document(title, path.read_bytes(), content_type)
    except DocumentParseError as exc:
        _notify_api(document_id, {"format": path.suffix.lstrip(".") or "text", "chunks": [], "error": str(exc), "task_id": task.request.id})
        return {"status": "failed", "error": str(exc)}

    source_kind = detect_source_kind(title, content_type)
    chunker = CodeChunker() if source_kind == "code" else SemanticChunker()
    chunks = chunker.split(
        parsed.text,
        document_id=UUID(document_id),
        version_id=UUID(version_id),
        title=title,
        pipeline_revision=settings.pipeline_revision,
        parsed_document=parsed,
    )
    chunks = [chunk.model_copy(update={"kb_id": UUID(kb_id), "source_kind": source_kind}) for chunk in chunks]
    chunks = _embed_chunks(chunks)
    payload = {
        "format": parsed.format,
        "source_kind": source_kind,
        "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
        "error": None,
        "task_id": task.request.id,
    }
    try:
        _notify_api(document_id, payload)
    except httpx.HTTPError as exc:
        # Keep the source until the API durably accepts the callback. A
        # transient API restart must not turn a retry into data loss.
        if task.request.retries >= task.max_retries:
            # Out of retries: best-effort tell the API so the document does not
            # stay "processing" forever and the UI stops polling. Use an empty
            # chunk list because the full payload is what the API rejected.
            try:
                _notify_api(document_id, {"format": parsed.format, "source_kind": source_kind, "chunks": [], "error": f"index_callback_failed: {exc}", "task_id": task.request.id})
            except httpx.HTTPError:
                pass
            raise
        raise task.retry(exc=exc, countdown=2 ** task.request.retries)
    # Retain the uploaded source so the source reader can serve the original
    # PDF/DOCX/PPTX and locate the cited page. Deleting it here made every
    # reader request return 404 once indexing succeeded.
    return {"status": "ready", "format": parsed.format, "source_kind": source_kind, "chunk_count": len(chunks)}
