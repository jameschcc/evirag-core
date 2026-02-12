# ==============================================================================
# EviRAG — Enterprise-grade High-performance Offline Knowledge Base with
# Traceable Retrieval and Generation Architecture
#         DOI: 10.3969/j.issn.1671-3370.2026.02.032 (S.Chang) https://d.wanfangdata.com.cn/periodical/zgxxj202602032
# ==============================================================================

"""Durable multi-turn chat history per knowledge base.

Answers remain evidence-bound: conversation history is only used to resolve
follow-up references (pronouns, elided subjects) and never becomes a source of
facts.  Messages are stored verbatim so a refreshed client can replay the
thread, while the assistant metadata keeps the validated answer envelope
(sources, claims, warnings) for citation rendering without re-running the AI.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Iterable
from uuid import UUID, uuid4

import asyncpg
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# How many stored messages are fed back into query rewriting and generation.
# One turn is a user question plus an assistant answer.
HISTORY_MESSAGE_LIMIT = 12
MAX_TITLE_CHARS = 60


class ConversationCreate(BaseModel):
    title: str = Field(default="", max_length=MAX_TITLE_CHARS)


class ConversationRename(BaseModel):
    title: str = Field(min_length=1, max_length=MAX_TITLE_CHARS)

    @field_validator("title")
    @classmethod
    def require_visible_title(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("会话标题不能为空")
        return stripped


def derive_title(question: str, limit: int = MAX_TITLE_CHARS) -> str:
    """Build a single-line conversation title from the first question."""
    compact = re.sub(r"\s+", " ", (question or "").strip())
    if not compact:
        return ""
    return compact[:limit].rstrip() + ("…" if len(compact) > limit else "")


def preview(text: str, limit: int = 160) -> str:
    compact = re.sub(r"\s+", " ", (text or "").strip())
    return compact[:limit] + ("…" if len(compact) > limit else "")


def history_from_messages(messages: Iterable[dict]) -> list[dict]:
    """Normalize stored messages into the compact history shape prompts use."""
    result: list[dict] = []
    for message in messages:
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        result.append({"role": role, "content": content})
    return result


class ConversationStore:
    """PostgreSQL-backed conversation and message storage."""

    def __init__(self, dsn):
        self.dsn = dsn.replace("postgresql+asyncpg://", "postgresql://", 1)
        self.pool = None

    async def open(self):
        self.pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=4, command_timeout=10)
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS kb_conversations (
                    conversation_id uuid PRIMARY KEY,
                    kb_id uuid NOT NULL,
                    user_id uuid,
                    title text NOT NULL DEFAULT '',
                    created_at timestamptz NOT NULL DEFAULT now(),
                    updated_at timestamptz NOT NULL DEFAULT now()
                );
                CREATE INDEX IF NOT EXISTS kb_conversations_kb_idx
                    ON kb_conversations(kb_id, updated_at DESC);
                CREATE TABLE IF NOT EXISTS kb_conversation_messages (
                    message_id uuid PRIMARY KEY,
                    conversation_id uuid NOT NULL REFERENCES kb_conversations(conversation_id) ON DELETE CASCADE,
                    ordinal integer NOT NULL,
                    role text NOT NULL CHECK (role IN ('user', 'assistant')),
                    content text NOT NULL DEFAULT '',
                    answer_id uuid,
                    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
                    created_at timestamptz NOT NULL DEFAULT now(),
                    UNIQUE (conversation_id, ordinal)
                );
                CREATE INDEX IF NOT EXISTS kb_conversation_messages_idx
                    ON kb_conversation_messages(conversation_id, ordinal);
            """)
        updated = await self.backfill_progress_from_snapshots()
        if updated:
            logger.info("backfilled retrieval progress for %s conversation messages", updated)

    async def backfill_progress_from_snapshots(self) -> int:
        """Fill missing per-turn progress from the persisted answer snapshots.

        Messages written before ``progress_events`` existed (or whose live
        progress entry had already expired) still have the retrieval timeline
        in ``kb_answer_snapshots``. This is idempotent and only fills empty or
        missing arrays, so a live turn is never overwritten.
        """
        try:
            result = await self.pool.execute("""
                UPDATE kb_conversation_messages AS m
                   SET metadata = jsonb_set(
                           m.metadata, '{progress_events}',
                           s.snapshot->'metadata'->'progress_timing'->'events', true
                       )
                  FROM kb_answer_snapshots AS s
                 WHERE m.role = 'assistant'
                   AND m.answer_id = s.answer_id
                   AND (m.metadata->'progress_events' IS NULL
                        OR jsonb_typeof(m.metadata->'progress_events') <> 'array'
                        OR jsonb_array_length(m.metadata->'progress_events') = 0)
                   AND jsonb_typeof(s.snapshot->'metadata'->'progress_timing'->'events') = 'array'
                   AND jsonb_array_length(s.snapshot->'metadata'->'progress_timing'->'events') > 0
            """)
        except asyncpg.UndefinedTableError:
            # ConversationStore may be opened before FeedbackStore in a bare
            # test harness; the snapshots table then does not exist yet.
            return 0
        except Exception:  # noqa: BLE001 - 回填失败不能阻止 API 启动
            logger.exception("failed to backfill conversation progress from answer snapshots")
            return 0
        try:
            return int(str(result).split()[-1])
        except (ValueError, IndexError):
            return 0

    async def close(self):
        if self.pool:
            await self.pool.close()

    async def create(self, kb_id: UUID, user_id: UUID | None = None, title: str = "") -> dict:
        conversation_id = uuid4()
        await self.pool.execute(
            "INSERT INTO kb_conversations(conversation_id, kb_id, user_id, title) VALUES ($1, $2, $3, $4)",
            conversation_id, kb_id, user_id, (title or "").strip()[:MAX_TITLE_CHARS],
        )
        row = await self._summary(conversation_id, kb_id)
        if row is None:  # pragma: no cover - row was just inserted
            raise LookupError("conversation_not_found")
        return row

    async def list(self, kb_id: UUID, limit: int = 50, offset: int = 0) -> tuple[int, list[dict]]:
        total = await self.pool.fetchval("SELECT count(*) FROM kb_conversations WHERE kb_id=$1", kb_id)
        rows = await self.pool.fetch("""
            SELECT c.conversation_id, c.kb_id, c.title, c.created_at, c.updated_at,
                   (SELECT count(*) FROM kb_conversation_messages m
                     WHERE m.conversation_id = c.conversation_id) AS message_count,
                   (SELECT left(m.content, 240) FROM kb_conversation_messages m
                     WHERE m.conversation_id = c.conversation_id AND m.role = 'user'
                     ORDER BY m.ordinal DESC LIMIT 1) AS last_question,
                   (SELECT left(m.content, 240) FROM kb_conversation_messages m
                     WHERE m.conversation_id = c.conversation_id AND m.role = 'assistant'
                     ORDER BY m.ordinal DESC LIMIT 1) AS last_answer
            FROM kb_conversations c
            WHERE c.kb_id = $1
            ORDER BY c.updated_at DESC, c.conversation_id DESC
            LIMIT $2 OFFSET $3
        """, kb_id, limit, offset)
        return int(total or 0), [self._summary_row(dict(row)) for row in rows]

    async def get(self, kb_id: UUID, conversation_id: UUID) -> dict | None:
        return await self._summary(conversation_id, kb_id)

    async def messages(self, kb_id: UUID, conversation_id: UUID) -> list[dict] | None:
        if await self.get(kb_id, conversation_id) is None:
            return None
        rows = await self.pool.fetch("""
            SELECT message_id, role, content, answer_id, metadata, created_at, ordinal
            FROM kb_conversation_messages
            WHERE conversation_id = $1
            ORDER BY ordinal
        """, conversation_id)
        return [self._message_row(dict(row)) for row in rows]

    async def history(self, conversation_id: UUID, limit: int = HISTORY_MESSAGE_LIMIT) -> list[dict]:
        """Return the newest ``limit`` messages in chronological order."""
        rows = await self.pool.fetch("""
            SELECT role, content FROM kb_conversation_messages
            WHERE conversation_id = $1
            ORDER BY ordinal DESC
            LIMIT $2
        """, conversation_id, limit)
        return history_from_messages(reversed([dict(row) for row in rows]))

    async def rename(self, kb_id: UUID, conversation_id: UUID, title: str) -> dict | None:
        result = await self.pool.execute(
            "UPDATE kb_conversations SET title=$3, updated_at=now() WHERE conversation_id=$1 AND kb_id=$2",
            conversation_id, kb_id, (title or "").strip()[:MAX_TITLE_CHARS],
        )
        if result == "UPDATE 0":
            return None
        return await self._summary(conversation_id, kb_id)

    async def delete(self, kb_id: UUID, conversation_id: UUID) -> bool:
        result = await self.pool.execute(
            "DELETE FROM kb_conversations WHERE conversation_id=$1 AND kb_id=$2",
            conversation_id, kb_id,
        )
        return result != "DELETE 0"

    async def append_user(self, kb_id: UUID, conversation_id: UUID, content: str) -> dict | None:
        return await self._append(
            kb_id, conversation_id, "user", content, answer_id=None, metadata=None,
            title_from=content,
        )

    async def append_assistant(
        self,
        kb_id: UUID,
        conversation_id: UUID,
        content: str,
        answer_id: UUID | None,
        metadata: dict | None = None,
    ) -> dict | None:
        return await self._append(
            kb_id, conversation_id, "assistant", content, answer_id=answer_id, metadata=metadata or {},
        )

    async def _append(
        self,
        kb_id: UUID,
        conversation_id: UUID,
        role: str,
        content: str,
        answer_id: UUID | None = None,
        metadata: dict | None = None,
        title_from: str | None = None,
    ) -> dict | None:
        message_id = uuid4()
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                exists = await conn.fetchval(
                    "SELECT title FROM kb_conversations WHERE conversation_id=$1 AND kb_id=$2 FOR UPDATE",
                    conversation_id, kb_id,
                )
                if exists is None:
                    return None
                ordinal = await conn.fetchval(
                    "SELECT COALESCE(MAX(ordinal), 0) + 1 FROM kb_conversation_messages WHERE conversation_id=$1",
                    conversation_id,
                )
                await conn.execute("""
                    INSERT INTO kb_conversation_messages(
                        message_id, conversation_id, ordinal, role, content, answer_id, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                """, message_id, conversation_id, ordinal, role, content, answer_id,
                    json.dumps(metadata or {}, ensure_ascii=False))
                title_update = ""
                params: list = [conversation_id]
                if title_from and not (exists or "").strip():
                    title_update = ", title=$2"
                    params.append(derive_title(title_from))
                await conn.execute(
                    f"UPDATE kb_conversations SET updated_at=now(){title_update} WHERE conversation_id=$1",
                    *params,
                )
        return {
            "message_id": str(message_id),
            "conversation_id": str(conversation_id),
            "ordinal": ordinal,
            "role": role,
            "content": content,
            "answer_id": str(answer_id) if answer_id else None,
            "metadata": metadata or {},
        }

    async def _summary(self, conversation_id: UUID, kb_id: UUID) -> dict | None:
        row = await self.pool.fetchrow("""
            SELECT c.conversation_id, c.kb_id, c.title, c.created_at, c.updated_at,
                   (SELECT count(*) FROM kb_conversation_messages m
                     WHERE m.conversation_id = c.conversation_id) AS message_count,
                   (SELECT left(m.content, 240) FROM kb_conversation_messages m
                     WHERE m.conversation_id = c.conversation_id AND m.role = 'user'
                     ORDER BY m.ordinal DESC LIMIT 1) AS last_question,
                   (SELECT left(m.content, 240) FROM kb_conversation_messages m
                     WHERE m.conversation_id = c.conversation_id AND m.role = 'assistant'
                     ORDER BY m.ordinal DESC LIMIT 1) AS last_answer
            FROM kb_conversations c
            WHERE c.conversation_id = $1 AND c.kb_id = $2
        """, conversation_id, kb_id)
        return self._summary_row(dict(row)) if row else None

    @staticmethod
    def _summary_row(row: dict) -> dict:
        return {
            "conversation_id": str(row["conversation_id"]),
            "kb_id": str(row["kb_id"]),
            "title": row.get("title") or "",
            "created_at": row["created_at"].isoformat(),
            "updated_at": row["updated_at"].isoformat(),
            "message_count": int(row.get("message_count") or 0),
            "last_question": row.get("last_question") or "",
            "last_answer_preview": preview(row.get("last_answer") or ""),
        }

    @staticmethod
    def _message_row(row: dict) -> dict:
        metadata = row.get("metadata")
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        return {
            "message_id": str(row["message_id"]),
            "role": row["role"],
            "content": row.get("content") or "",
            "answer_id": str(row["answer_id"]) if row.get("answer_id") else None,
            "metadata": metadata or {},
            "created_at": row["created_at"].isoformat(),
        }
