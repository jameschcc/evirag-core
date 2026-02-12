# ==============================================================================
# EviRAG — Enterprise-grade High-performance Offline Knowledge Base with
# Traceable Retrieval and Generation Architecture
# Personal Research Project · Author: S.Chang (常帅)
# Paper: 常帅. 企业级高性能离网知识库可溯源检索与生成架构[J].
#        China Information World, 2026(2).
#        DOI: 10.3969/j.issn.1671-3370.2026.02.032
#        https://d.wanfangdata.com.cn/periodical/zgxxj202602032
# ==============================================================================

"""One-time migration of the running prototype's documents into PostgreSQL.

Run before restarting the old API. It reads the old document list through the
API and source files from a separately supplied directory, then preserves the
existing document/version IDs while creating durable paragraphs.
"""
import asyncio
import os
from pathlib import Path

import httpx

from .catalog import DocumentCatalog
from .chunking import SemanticChunker
from .config import get_settings
from .document_parser import parse_document
from .contracts import DocumentResponse


async def migrate() -> None:
    settings = get_settings()
    source_dir = Path(os.environ["KB_LEGACY_SOURCE_DIR"])
    api_url = os.environ.get("KB_LEGACY_API_URL", "http://api:8000").rstrip("/")
    catalog = DocumentCatalog(settings.postgres_dsn)
    await catalog.open()
    try:
        count = await catalog.pool.fetchval("SELECT count(*) FROM kb_documents")
        if count:
            raise RuntimeError("refusing migration: kb_documents is not empty")
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(f"{api_url}/api/v1/documents")
            response.raise_for_status()
            documents = [DocumentResponse.model_validate(item) for item in response.json()]
        chunker = SemanticChunker()
        for document in documents:
            path = source_dir / document.title
            if not path.is_file():
                raise FileNotFoundError(path)
            content = path.read_bytes()
            parsed = parse_document(document.title, content, document.format)
            chunks = chunker.split(
                parsed.text, document.document_id, document.version_id,
                document.title, settings.pipeline_revision, parsed,
            )
            value = document.model_dump(mode="json")
            value["status"] = "ready"
            value["chunk_count"] = len(chunks)
            value["error"] = None
            await catalog.save_document(document.document_id, value, f"legacy://{document.document_id}")
            await catalog.replace_paragraphs(document.document_id, chunks)
            print(f"migrated {document.document_id} {document.title}: {len(chunks)} paragraphs")
    finally:
        await catalog.close()


if __name__ == "__main__":
    asyncio.run(migrate())
