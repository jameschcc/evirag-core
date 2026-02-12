# ==============================================================================
# EviRAG — Enterprise-grade High-performance Offline Knowledge Base with
# Traceable Retrieval and Generation Architecture
# Personal Research Project · Author: S.Chang (常帅)
# Paper: 常帅. 企业级高性能离网知识库可溯源检索与生成架构[J].
#        China Information World, 2026(2).
#        DOI: 10.3969/j.issn.1671-3370.2026.02.032
#        https://d.wanfangdata.com.cn/periodical/zgxxj202602032
# ==============================================================================

"""Durable answer snapshots and human ratings; no automatic training."""
import json
from typing import Literal
from uuid import UUID

import asyncpg
from pydantic import BaseModel, ConfigDict, model_validator


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    feedback_id: UUID
    rating: Literal["positive", "negative"]
    category: Literal["irrelevant", "incorrect", "incomplete", "citation", "other"] | None = None

    @model_validator(mode="after")
    def validate_category(self):
        if (self.rating == "negative") != (self.category is not None):
            raise ValueError("错误反馈必须选择分类，点赞不携带错误分类")
        return self


class FeedbackStore:
    def __init__(self, dsn):
        self.dsn = dsn.replace("postgresql+asyncpg://", "postgresql://", 1)
        self.pool = None

    async def open(self):
        self.pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=4, command_timeout=10)
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS kb_answer_snapshots (
                    answer_id uuid PRIMARY KEY,
                    created_at timestamptz NOT NULL DEFAULT now(),
                    snapshot jsonb NOT NULL
                );
                CREATE TABLE IF NOT EXISTS kb_human_feedback (
                    feedback_id uuid PRIMARY KEY,
                    answer_id uuid NOT NULL REFERENCES kb_answer_snapshots(answer_id),
                    created_at timestamptz NOT NULL DEFAULT now(),
                    rating text NOT NULL CHECK (rating IN ('positive', 'negative')),
                    score smallint NOT NULL CHECK (score IN (0, 1)),
                    category text CHECK (category IN ('irrelevant', 'incorrect', 'incomplete', 'citation', 'other')),
                    source text NOT NULL DEFAULT 'human',
                    review_status text NOT NULL DEFAULT 'pending',
                    CHECK ((rating = 'positive' AND score = 1 AND category IS NULL)
                        OR (rating = 'negative' AND score = 0 AND category IS NOT NULL))
                );
                CREATE INDEX IF NOT EXISTS kb_feedback_answer_idx ON kb_human_feedback(answer_id);
            """)

    async def close(self):
        if self.pool:
            await self.pool.close()

    async def save_answer(self, answer_id, snapshot):
        await self.pool.execute(
            "INSERT INTO kb_answer_snapshots(answer_id, snapshot) VALUES ($1, $2::jsonb)",
            answer_id, json.dumps(snapshot, ensure_ascii=False),
        )

    async def rate(self, answer_id, feedback):
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                if not await conn.fetchval("SELECT 1 FROM kb_answer_snapshots WHERE answer_id=$1", answer_id):
                    raise LookupError("answer_not_found")
                await conn.execute("""
                    INSERT INTO kb_human_feedback(feedback_id, answer_id, rating, score, category)
                    VALUES ($1, $2, $3, $4, $5) ON CONFLICT (feedback_id) DO NOTHING
                """, feedback.feedback_id, answer_id, feedback.rating, int(feedback.rating == "positive"), feedback.category)
                row = await conn.fetchrow("SELECT * FROM kb_human_feedback WHERE feedback_id=$1", feedback.feedback_id)
                if row["answer_id"] != answer_id or row["rating"] != feedback.rating or row["category"] != feedback.category:
                    raise ValueError("feedback_id_conflict")
                snapshot_json = await conn.fetchval(
                    "SELECT snapshot FROM kb_answer_snapshots WHERE answer_id=$1", answer_id)
                metadata = json.loads(snapshot_json).get("metadata", {})
                return {"feedback_id": str(row["feedback_id"]), "status": "saved", "created_at": row["created_at"].isoformat(),
                        "git_tag": metadata.get("git_tag"), "git_commit": metadata.get("git_commit")}

    async def answer_kb_ids(self, answer_id) -> list[UUID]:
        snapshot_json = await self.pool.fetchval(
            "SELECT snapshot FROM kb_answer_snapshots WHERE answer_id=$1", answer_id
        )
        if snapshot_json is None:
            return []
        snapshot = json.loads(snapshot_json)
        values = snapshot.get("input", {}).get("kb_ids") or snapshot.get("metadata", {}).get("kb_ids") or []
        result = []
        for value in values:
            try:
                result.append(UUID(str(value)))
            except (TypeError, ValueError):
                continue
        return result

    async def export(self):
        """Operator-only JSONL export. Feedback is not a reference answer."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                async for row in conn.cursor("""
                    SELECT f.*, s.created_at AS answer_created_at, s.snapshot
                    FROM kb_human_feedback f JOIN kb_answer_snapshots s USING (answer_id)
                    ORDER BY f.created_at, f.feedback_id
                """):
                    record = dict(row)
                    record["snapshot"] = json.loads(record["snapshot"])
                    record["reference_output"] = None
                    yield record


async def _export():
    from kb.config import get_settings
    store = FeedbackStore(get_settings().postgres_dsn)
    # Export must not create or migrate tables.
    store.pool = await asyncpg.create_pool(store.dsn, min_size=1, max_size=1, command_timeout=30)
    try:
        async for record in store.export():
            print(json.dumps(record, ensure_ascii=False, default=str))
    finally:
        await store.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(_export())
