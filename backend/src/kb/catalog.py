# ==============================================================================
# EviRAG — Enterprise-grade High-performance Offline Knowledge Base with
# Traceable Retrieval and Generation Architecture
#         DOI: 10.3969/j.issn.1671-3370.2026.02.032 (S.Chang) https://d.wanfangdata.com.cn/periodical/zgxxj202602032
# ==============================================================================

"""Durable document/version/paragraph catalog.

PostgreSQL is the source of truth. The in-process retriever is only a
rebuildable read cache.
"""
import json
from datetime import datetime
from uuid import UUID

import asyncpg

from .contracts import Chunk, DEFAULT_KB_ID, GlossaryEntry


# 与 bge-m3 对齐；维度不同的历史 stub（例如早期 64 维）不允许写入该列，
# 否则 pgvector 的 vector(1024) 会整批拒绝，导致入库失败。
EMBEDDING_DIMENSIONS = 1024


def _vector_literal(values) -> str | None:
    """把 float 列表转成 pgvector 字面量；维度不符或空向量返回 None。"""
    if not values or len(values) != EMBEDDING_DIMENSIONS:
        return None
    return "[" + ",".join(f"{float(value):.7g}" for value in values) + "]"


def paragraph_rows(document_id: UUID, chunks: list[Chunk], vector_ready: bool) -> list[tuple]:
    """Shape paragraph rows for the exact INSERT the deployment supports.

    Without pgvector the table has five columns, so the embedding tuple fields
    must be dropped; keeping them makes asyncpg reject the batch.
    """
    rows = [
        (chunk.chunk_id, chunk.kb_id, document_id, chunk.version_id,
         json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False),
         _vector_literal(chunk.embedding), chunk.embedding_revision or None)
        for chunk in chunks
    ]
    return rows if vector_ready else [row[:5] for row in rows]


class DocumentCatalog:
    def __init__(self, dsn: str):
        self.dsn = dsn.replace("postgresql+asyncpg://", "postgresql://", 1)
        self.pool: asyncpg.Pool | None = None
        # 只有 pgvector 扩展可用时才启用 SQL 稠密召回；否则调用方继续走内存计算，
        # 这样未换镜像的旧部署仍能启动。
        self.vector_ready = False

    async def open(self) -> None:
        self.pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=4, command_timeout=15)
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS kb_knowledge_bases (
                    kb_id uuid PRIMARY KEY,
                    slug text NOT NULL UNIQUE,
                    name text NOT NULL,
                    description text NOT NULL DEFAULT '',
                    status text NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'archived')),
                    created_at timestamptz NOT NULL DEFAULT now(),
                    updated_at timestamptz NOT NULL DEFAULT now()
                );
                INSERT INTO kb_knowledge_bases(kb_id, slug, name)
                VALUES ('00000000-0000-0000-0000-000000000001', 'default', '默认知识库')
                ON CONFLICT (kb_id) DO NOTHING;
                CREATE TABLE IF NOT EXISTS kb_memberships (
                    kb_id uuid NOT NULL REFERENCES kb_knowledge_bases(kb_id) ON DELETE CASCADE,
                    user_id uuid NOT NULL,
                    role text NOT NULL CHECK (role IN ('owner', 'admin', 'editor', 'viewer')),
                    created_at timestamptz NOT NULL DEFAULT now(),
                    updated_at timestamptz NOT NULL DEFAULT now(),
                    PRIMARY KEY (kb_id, user_id)
                );
                CREATE INDEX IF NOT EXISTS kb_memberships_user_idx ON kb_memberships(user_id, kb_id);
                CREATE TABLE IF NOT EXISTS kb_documents (
                    document_id uuid PRIMARY KEY,
                    kb_id uuid NOT NULL REFERENCES kb_knowledge_bases(kb_id),
                    version_id uuid NOT NULL,
                    title text NOT NULL,
                    format text NOT NULL,
                    status text NOT NULL CHECK (status IN ('ready', 'processing', 'failed')),
                    chunk_count integer NOT NULL DEFAULT 0,
                    error text,
                    task_id text,
                    created_at timestamptz NOT NULL,
                    source_path text NOT NULL,
                    glossary_state smallint NOT NULL DEFAULT 0
                );
                ALTER TABLE kb_documents ADD COLUMN IF NOT EXISTS kb_id uuid;
                UPDATE kb_documents SET kb_id='00000000-0000-0000-0000-000000000001' WHERE kb_id IS NULL;
                ALTER TABLE kb_documents ALTER COLUMN kb_id SET NOT NULL;
                CREATE INDEX IF NOT EXISTS kb_documents_kb_idx ON kb_documents(kb_id, created_at DESC);
                ALTER TABLE kb_documents ADD COLUMN IF NOT EXISTS glossary_state smallint NOT NULL DEFAULT 0;
                ALTER TABLE kb_documents ADD COLUMN IF NOT EXISTS source_kind text NOT NULL DEFAULT 'document';
                CREATE TABLE IF NOT EXISTS kb_paragraphs (
                    para_id uuid PRIMARY KEY,
                    kb_id uuid NOT NULL REFERENCES kb_knowledge_bases(kb_id),
                    document_id uuid NOT NULL REFERENCES kb_documents(document_id) ON DELETE CASCADE,
                    version_id uuid NOT NULL,
                    payload jsonb NOT NULL,
                    updated_at timestamptz NOT NULL DEFAULT now(),
                    row_id bigint
                );
                ALTER TABLE kb_paragraphs ADD COLUMN IF NOT EXISTS kb_id uuid;
                UPDATE kb_paragraphs p SET kb_id=d.kb_id FROM kb_documents d WHERE d.document_id=p.document_id AND p.kb_id IS NULL;
                ALTER TABLE kb_paragraphs ALTER COLUMN kb_id SET NOT NULL;
                CREATE INDEX IF NOT EXISTS kb_paragraphs_document_idx ON kb_paragraphs(document_id);
                CREATE INDEX IF NOT EXISTS kb_paragraphs_kb_idx ON kb_paragraphs(kb_id, document_id);
                CREATE SEQUENCE IF NOT EXISTS kb_paragraphs_row_seq;
                ALTER TABLE kb_paragraphs ADD COLUMN IF NOT EXISTS row_id bigint;
                ALTER TABLE kb_paragraphs ALTER COLUMN row_id SET DEFAULT nextval('kb_paragraphs_row_seq');
                UPDATE kb_paragraphs SET row_id = nextval('kb_paragraphs_row_seq') WHERE row_id IS NULL;
                CREATE TABLE IF NOT EXISTS kb_glossary (
                    kb_id uuid NOT NULL REFERENCES kb_knowledge_bases(kb_id),
                    document_id uuid NOT NULL REFERENCES kb_documents(document_id) ON DELETE CASCADE,
                    version_id uuid NOT NULL,
                    term text NOT NULL,
                    term_key text NOT NULL,
                    expansion text NOT NULL,
                    chunk_id uuid,
                    origin text NOT NULL DEFAULT 'ai',
                    created_at timestamptz NOT NULL DEFAULT now(),
                    PRIMARY KEY (kb_id, document_id, term_key, expansion)
                );
                ALTER TABLE kb_glossary ADD COLUMN IF NOT EXISTS kb_id uuid;
                UPDATE kb_glossary g SET kb_id=d.kb_id FROM kb_documents d WHERE d.document_id=g.document_id AND g.kb_id IS NULL;
                ALTER TABLE kb_glossary ALTER COLUMN kb_id SET NOT NULL;
                CREATE UNIQUE INDEX IF NOT EXISTS kb_glossary_kb_term_unique ON kb_glossary(kb_id, document_id, term_key, expansion);
                CREATE INDEX IF NOT EXISTS kb_glossary_term_idx ON kb_glossary(term_key);
                CREATE INDEX IF NOT EXISTS kb_glossary_kb_term_idx ON kb_glossary(kb_id, term_key);
                CREATE TABLE IF NOT EXISTS kb_runtime_settings (
                    setting_key text PRIMARY KEY,
                    payload jsonb NOT NULL,
                    updated_at timestamptz NOT NULL DEFAULT now()
                );
                CREATE TABLE IF NOT EXISTS kb_provider_plans (
                    plan_id uuid PRIMARY KEY,
                    name text NOT NULL UNIQUE,
                    payload jsonb NOT NULL,
                    created_at timestamptz NOT NULL DEFAULT now(),
                    updated_at timestamptz NOT NULL DEFAULT now()
                );
            """)
        await self._prepare_vector_columns()

    async def _prepare_vector_columns(self) -> None:
        """启用 pgvector 稠密召回所需的扩展与列。

        失败（镜像里没有 pgvector、账号无建扩展权限）时不抛异常：``vector_ready``
        保持 False，检索退回内存精确计算，避免整个 API 起不来。
        """
        try:
            await self.pool.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await self.pool.execute(
                f"ALTER TABLE kb_paragraphs ADD COLUMN IF NOT EXISTS embedding vector({EMBEDDING_DIMENSIONS})"
            )
            await self.pool.execute(
                "ALTER TABLE kb_paragraphs ADD COLUMN IF NOT EXISTS embedding_revision text"
            )
            self.vector_ready = True
        except Exception:  # noqa: BLE001 - 旧镜像/无权限时必须降级而不是崩溃
            self.vector_ready = False

    async def backfill_embedding_column(self) -> int:
        """把 payload 里已有的向量搬到 embedding 列（存量数据一次性迁移）。

        只处理维度正确且列仍为空的片段；维度不符的历史 stub 保持 NULL，等
        ``_backfill_embeddings`` 重新向量化后再写入。
        """
        if not self.vector_ready:
            return 0
        try:
            result = await self.pool.execute(
                f"""
                UPDATE kb_paragraphs
                SET embedding = (payload->>'embedding')::vector,
                    embedding_revision = payload->>'embedding_revision'
                WHERE embedding IS NULL
                  AND jsonb_typeof(payload->'embedding') = 'array'
                  AND jsonb_array_length(payload->'embedding') = {EMBEDDING_DIMENSIONS}
                """
            )
        except Exception:  # noqa: BLE001
            return 0
        try:
            return int(str(result).split()[-1])
        except (ValueError, IndexError):
            return 0

    async def search_dense(
        self,
        query_vectors: list[list[float]],
        source_kinds: list[str] | None,
        kb_ids: list[str] | None,
        embedding_revision: str | None,
        min_score: float,
        limit: int = 500,
    ) -> tuple[list[tuple[str, float]], int, int] | None:
        """精确扫描（不建 ANN 索引）返回稠密候选，与内存实现数学等价。

        返回值是 ``(候选, 有向量的可见片段数, 缺向量的可见片段数)``。候选只按
        ``min_score`` 过滤后取前 ``limit`` 条，Python 侧再做与内存路径相同的
        top-100 截断；``limit`` 远大于 100 是为了保证与全量排序结果一致。
        """
        vectors = [vector for vector in query_vectors if vector]
        if not self.vector_ready or not vectors:
            return None
        payload = f"[{','.join('[' + ','.join(f'{float(v):.7g}' for v in vector) + ']' for vector in vectors)}]"
        kinds = list(source_kinds) if source_kinds else None
        scope_ids = [str(item) for item in kb_ids] if kb_ids else None
        from_clause = """
            FROM kb_paragraphs p
            JOIN kb_documents d ON d.document_id = p.document_id
        """
        where_clause = """
            WHERE COALESCE(p.payload->>'search_state', 'published') = 'published'
              AND ($1::text[] IS NULL OR d.source_kind = ANY($1::text[]))
              AND ($2::uuid[] IS NULL OR p.kb_id = ANY($2::uuid[]))
        """
        counts = await self.pool.fetchrow(
            f"""
            SELECT
                count(*) FILTER (
                    WHERE p.embedding IS NOT NULL AND p.embedding_revision IS NOT DISTINCT FROM $3
                ) AS available,
                count(*) FILTER (
                    WHERE p.embedding IS NULL OR p.embedding_revision IS DISTINCT FROM $3
                ) AS missing
            {from_clause}
            {where_clause}
            """,
            kinds,
            scope_ids,
            embedding_revision,
        )
        rows = await self.pool.fetch(
            f"""
            WITH q AS (
                SELECT (value #>> '{{}}')::vector AS vec
                FROM jsonb_array_elements($4::jsonb) AS element(value)
            )
            SELECT p.para_id::text AS chunk_id,
                   MAX(1 - (p.embedding <=> q.vec)) AS score
            {from_clause}
            CROSS JOIN q
            {where_clause}
              AND p.embedding IS NOT NULL
              AND vector_norm(p.embedding) > 0
              AND p.embedding_revision IS NOT DISTINCT FROM $3
            GROUP BY p.para_id
            HAVING MAX(1 - (p.embedding <=> q.vec)) >= $5
            ORDER BY score DESC, chunk_id
            LIMIT $6
            """,
            kinds,
            scope_ids,
            embedding_revision,
            payload,
            float(min_score),
            int(limit),
        )
        candidates = [(row["chunk_id"], float(row["score"])) for row in rows]
        return candidates, int(counts["available"] or 0), int(counts["missing"] or 0)

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()

    async def load_runtime_settings(self, setting_key: str) -> dict | None:
        payload = await self.pool.fetchval(
            "SELECT payload FROM kb_runtime_settings WHERE setting_key=$1", setting_key
        )
        if payload is None:
            return None
        return json.loads(payload) if isinstance(payload, str) else dict(payload)

    async def save_runtime_settings(self, setting_key: str, payload: dict) -> None:
        await self.pool.execute(
            """
            INSERT INTO kb_runtime_settings(setting_key, payload, updated_at)
            VALUES ($1, $2::jsonb, now())
            ON CONFLICT (setting_key) DO UPDATE
            SET payload=EXCLUDED.payload, updated_at=now()
            """,
            setting_key,
            json.dumps(payload, ensure_ascii=False),
        )

    async def delete_runtime_settings(self, setting_key: str) -> None:
        await self.pool.execute("DELETE FROM kb_runtime_settings WHERE setting_key=$1", setting_key)

    async def update_runtime_settings(self, values: dict[str, dict | None]) -> None:
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                for setting_key, payload in values.items():
                    if payload is None:
                        await conn.execute(
                            "DELETE FROM kb_runtime_settings WHERE setting_key=$1", setting_key
                        )
                    else:
                        await conn.execute(
                            """
                            INSERT INTO kb_runtime_settings(setting_key, payload, updated_at)
                            VALUES ($1, $2::jsonb, now())
                            ON CONFLICT (setting_key) DO UPDATE
                            SET payload=EXCLUDED.payload, updated_at=now()
                            """,
                            setting_key,
                            json.dumps(payload, ensure_ascii=False),
                        )

    async def create_provider_plan(
        self, plan_id: UUID, name: str, payload: dict, runtime_settings: dict[str, dict]
    ) -> None:
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "INSERT INTO kb_provider_plans(plan_id, name, payload) VALUES ($1, $2, $3::jsonb)",
                    plan_id,
                    name,
                    json.dumps(payload, ensure_ascii=False),
                )
                for setting_key, setting_payload in runtime_settings.items():
                    await conn.execute(
                        """
                        INSERT INTO kb_runtime_settings(setting_key, payload, updated_at)
                        VALUES ($1, $2::jsonb, now())
                        ON CONFLICT (setting_key) DO UPDATE
                        SET payload=EXCLUDED.payload, updated_at=now()
                        """,
                        setting_key,
                        json.dumps(setting_payload, ensure_ascii=False),
                    )

    async def list_provider_plans(self) -> list[dict]:
        rows = await self.pool.fetch(
            "SELECT plan_id, name, payload, created_at, updated_at FROM kb_provider_plans ORDER BY created_at, name"
        )
        return [
            {
                "plan_id": row["plan_id"],
                "name": row["name"],
                "payload": json.loads(row["payload"]) if isinstance(row["payload"], str) else dict(row["payload"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    async def get_provider_plan(self, plan_id: UUID) -> dict | None:
        row = await self.pool.fetchrow(
            "SELECT plan_id, name, payload, created_at, updated_at FROM kb_provider_plans WHERE plan_id=$1",
            plan_id,
        )
        if row is None:
            return None
        return {
            "plan_id": row["plan_id"],
            "name": row["name"],
            "payload": json.loads(row["payload"]) if isinstance(row["payload"], str) else dict(row["payload"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    async def delete_provider_plan(self, plan_id: UUID) -> None:
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("DELETE FROM kb_provider_plans WHERE plan_id=$1", plan_id)
                await conn.execute(
                    """
                    DELETE FROM kb_runtime_settings
                    WHERE setting_key='active_provider_plan' AND payload->>'id'=$1
                    """,
                    str(plan_id),
                )

    async def seed_default_membership(self, user_id: UUID) -> None:
        await self.pool.execute(
            """
            INSERT INTO kb_memberships(kb_id, user_id, role)
            VALUES ($1, $2, 'owner')
            ON CONFLICT (kb_id, user_id) DO NOTHING
            """,
            DEFAULT_KB_ID,
            user_id,
        )

    async def list_knowledge_bases(self, user_id: UUID) -> list[dict]:
        rows = await self.pool.fetch(
            """
            SELECT k.kb_id, k.slug, k.name, k.description, k.status, k.created_at,
                   m.role,
                   (SELECT count(*)::int FROM kb_documents d WHERE d.kb_id=k.kb_id) AS document_count,
                   (SELECT count(*)::int FROM kb_paragraphs p WHERE p.kb_id=k.kb_id) AS chunk_count
            FROM kb_knowledge_bases k
            JOIN kb_memberships m ON m.kb_id=k.kb_id AND m.user_id=$1
            GROUP BY k.kb_id, k.slug, k.name, k.description, k.status, k.created_at, m.role
            ORDER BY k.created_at, k.slug
            """,
            user_id,
        )
        return [dict(row) for row in rows]

    async def get_knowledge_base(self, kb_id: UUID) -> dict | None:
        row = await self.pool.fetchrow(
            "SELECT kb_id, slug, name, description, status, created_at FROM kb_knowledge_bases WHERE kb_id=$1",
            kb_id,
        )
        return dict(row) if row else None

    async def create_knowledge_base(self, kb_id: UUID, name: str, slug: str, description: str, owner_user_id: UUID) -> dict:
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "INSERT INTO kb_knowledge_bases(kb_id, slug, name, description) VALUES ($1, $2, $3, $4)",
                    kb_id, slug, name, description,
                )
                await conn.execute(
                    "INSERT INTO kb_memberships(kb_id, user_id, role) VALUES ($1, $2, 'owner')",
                    kb_id, owner_user_id,
                )
        return (await self.get_knowledge_base(kb_id)) or {}

    async def get_membership_role(self, kb_id: UUID, user_id: UUID) -> str | None:
        return await self.pool.fetchval(
            "SELECT role FROM kb_memberships WHERE kb_id=$1 AND user_id=$2",
            kb_id, user_id,
        )

    async def list_members(self, kb_id: UUID) -> list[dict]:
        rows = await self.pool.fetch(
            "SELECT user_id, role, created_at, updated_at FROM kb_memberships WHERE kb_id=$1 ORDER BY created_at, user_id",
            kb_id,
        )
        return [dict(row) for row in rows]

    async def upsert_membership(self, kb_id: UUID, user_id: UUID, role: str) -> None:
        await self.pool.execute(
            """
            INSERT INTO kb_memberships(kb_id, user_id, role, updated_at)
            VALUES ($1, $2, $3, now())
            ON CONFLICT (kb_id, user_id) DO UPDATE SET role=EXCLUDED.role, updated_at=now()
            """,
            kb_id, user_id, role,
        )

    async def remove_membership(self, kb_id: UUID, user_id: UUID) -> None:
        await self.pool.execute("DELETE FROM kb_memberships WHERE kb_id=$1 AND user_id=$2", kb_id, user_id)

    async def save_document(self, document_id: UUID, value: dict, source_path: str) -> None:
        created_at = value["created_at"]
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        await self.pool.execute("""
            INSERT INTO kb_documents
                (document_id, kb_id, version_id, title, format, source_kind, status, chunk_count, error, task_id, created_at, source_path, glossary_state)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            ON CONFLICT (document_id) DO UPDATE SET
                kb_id=EXCLUDED.kb_id, version_id=EXCLUDED.version_id, title=EXCLUDED.title, format=EXCLUDED.format,
                source_kind=EXCLUDED.source_kind,
                status=EXCLUDED.status, chunk_count=EXCLUDED.chunk_count, error=EXCLUDED.error,
                task_id=EXCLUDED.task_id, source_path=EXCLUDED.source_path, glossary_state=EXCLUDED.glossary_state
        """, document_id, value.get("kb_id", DEFAULT_KB_ID), value["version_id"], value["title"], value["format"],
            value.get("source_kind", "document"), value["status"],
            value["chunk_count"], value["error"], value["task_id"], created_at, source_path,
            int(value.get("glossary_state", 0)))

    async def update_document(self, document_id: UUID, value: dict) -> None:
        await self.pool.execute("""
            UPDATE kb_documents
            SET kb_id=$2, version_id=$3, title=$4, format=$5, source_kind=$6, status=$7, chunk_count=$8, error=$9, task_id=$10, glossary_state=$11
            WHERE document_id=$1
        """, document_id, value.get("kb_id", DEFAULT_KB_ID), value["version_id"], value["title"], value["format"],
            value.get("source_kind", "document"), value["status"],
            value["chunk_count"], value["error"], value["task_id"], int(value.get("glossary_state", 0)))

    async def replace_paragraphs(self, document_id: UUID, chunks: list[Chunk]) -> None:
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("DELETE FROM kb_paragraphs WHERE document_id=$1", document_id)
                rows = paragraph_rows(document_id, chunks, self.vector_ready)
                if not self.vector_ready:
                    # 非 pgvector 部署只有 5 个占位符；多带的 embedding 列必须丢掉，
                    # 否则 asyncpg 会报参数个数不匹配并中断整个入库。
                    await conn.executemany("""
                        INSERT INTO kb_paragraphs(para_id, kb_id, document_id, version_id, payload)
                        VALUES ($1, $2, $3, $4, $5::jsonb)
                    """, rows)
                    return
                await conn.executemany("""
                    INSERT INTO kb_paragraphs(para_id, kb_id, document_id, version_id, payload, embedding, embedding_revision)
                    VALUES ($1, $2, $3, $4, $5::jsonb, $6::vector, $7)
                """, rows)

    async def replace_glossary(self, document_id: UUID, version_id: UUID, entries: list[GlossaryEntry]) -> None:
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("DELETE FROM kb_glossary WHERE document_id=$1", document_id)
                if entries:
                    await conn.executemany("""
                        INSERT INTO kb_glossary(kb_id, document_id, version_id, term, term_key, expansion, chunk_id, origin)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        ON CONFLICT (kb_id, document_id, term_key, expansion) DO NOTHING
                    """, [(entry.kb_id or DEFAULT_KB_ID, document_id, version_id, entry.term, entry.term.lower(), entry.expansion,
                           entry.chunk_id, entry.origin) for entry in entries])

    async def delete_glossary_entry(self, document_id: UUID, term: str, expansion: str, kb_id: UUID | None = None) -> int:
        """Delete one term/expansion pair; return the number of removed rows."""
        query = "DELETE FROM kb_glossary WHERE document_id=$1 AND term_key=$2 AND expansion=$3"
        args: list = [document_id, term.strip().lower(), expansion]
        if kb_id is not None:
            query += " AND kb_id=$4"
            args.append(kb_id)
        async with self.pool.acquire() as conn:
            status = await conn.execute(query, *args)
        return int(str(status).rsplit(" ", 1)[-1])

    async def load_glossary(self) -> list[GlossaryEntry]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT kb_id, document_id, version_id, term, expansion, chunk_id, origin FROM kb_glossary ORDER BY kb_id, document_id, term_key"
            )
        return [GlossaryEntry(
            kb_id=row["kb_id"], document_id=row["document_id"], version_id=row["version_id"], term=row["term"],
            expansion=row["expansion"], chunk_id=row["chunk_id"], origin=row["origin"],
        ) for row in rows]

    async def glossary_counts(self) -> dict[UUID, int]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT document_id, count(*) AS total FROM kb_glossary GROUP BY document_id")
        return {row["document_id"]: int(row["total"]) for row in rows}

    async def list_glossary(self, document_id: UUID | None = None, kb_id: UUID | None = DEFAULT_KB_ID, limit: int | None = None, offset: int = 0) -> list[dict]:
        query = """
            SELECT g.kb_id, g.document_id, g.version_id, g.term, g.expansion, g.chunk_id, g.origin, d.title
            FROM kb_glossary g
            JOIN kb_documents d ON d.document_id = g.document_id
        """
        args: list = []
        if document_id is not None:
            args.append(document_id)
            query += f" WHERE g.document_id = ${len(args)}"
        if kb_id is not None:
            args.append(kb_id)
            query += f" {'AND' if document_id is not None else 'WHERE'} g.kb_id = ${len(args)}"
        query += " ORDER BY g.term_key, g.expansion"
        if limit is not None:
            args.append(limit)
            limit_index = len(args)
            args.append(offset)
            query += f" LIMIT ${limit_index} OFFSET ${len(args)}"
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *args)
        return [
            {"kb_id": row["kb_id"], "document_id": row["document_id"], "version_id": row["version_id"], "term": row["term"],
             "expansion": row["expansion"], "chunk_id": row["chunk_id"], "origin": row["origin"], "title": row["title"]}
            for row in rows
        ]

    async def count_glossary(self, document_id: UUID | None = None, kb_id: UUID | None = DEFAULT_KB_ID) -> int:
        async with self.pool.acquire() as conn:
            if document_id is None and kb_id is None:
                return int(await conn.fetchval("SELECT count(*) FROM kb_glossary"))
            if kb_id is not None and document_id is None:
                return int(await conn.fetchval("SELECT count(*) FROM kb_glossary WHERE kb_id=$1", kb_id))
            if kb_id is None:
                return int(await conn.fetchval("SELECT count(*) FROM kb_glossary WHERE document_id=$1", document_id))
            return int(await conn.fetchval("SELECT count(*) FROM kb_glossary WHERE kb_id=$1 AND document_id=$2", kb_id, document_id))

    async def load(self) -> tuple[list[dict], list[Chunk]]:
        documents = []
        chunks = []
        async with self.pool.acquire() as conn:
            for row in await conn.fetch("SELECT * FROM kb_documents ORDER BY created_at, document_id"):
                value = dict(row)
                document_id = value.pop("document_id")
                value.pop("source_path", None)
                value["version_id"] = str(value["version_id"])
                value["created_at"] = value["created_at"].isoformat()
                documents.append({"document_id": document_id, **value})
            for row in await conn.fetch("SELECT kb_id, payload FROM kb_paragraphs ORDER BY kb_id, document_id, row_id"):
                payload = json.loads(row["payload"])
                payload.setdefault("kb_id", row["kb_id"])
                chunks.append(Chunk.model_validate(payload))
        self._assign_missing_ordinals(chunks)
        return documents, chunks

    @staticmethod
    def _assign_missing_ordinals(chunks: list[Chunk]) -> None:
        """Row order matches ingestion order; backfill ordinals for older chunks."""
        by_document: dict[UUID, list[Chunk]] = {}
        for chunk in chunks:
            by_document.setdefault(chunk.document_id, []).append(chunk)
        for document_chunks in by_document.values():
            if all(chunk.ordinal is None for chunk in document_chunks):
                for index, chunk in enumerate(document_chunks):
                    chunk.ordinal = index

    async def get_document_source(self, document_id: UUID) -> dict | None:
        row = await self.pool.fetchrow(
            "SELECT document_id, kb_id, version_id, title, format, source_path FROM kb_documents WHERE document_id=$1",
            document_id,
        )
        return dict(row) if row else None

    async def export_knowledge_base(self, kb_id: UUID) -> dict:
        async with self.pool.acquire() as conn:
            knowledge_base = await conn.fetchrow(
                "SELECT kb_id, slug, name, description, status, created_at FROM kb_knowledge_bases WHERE kb_id=$1",
                kb_id,
            )
            if knowledge_base is None:
                return {}
            documents = await conn.fetch(
                "SELECT document_id, kb_id, version_id, title, format, status, chunk_count, error, task_id, created_at, source_path, glossary_state FROM kb_documents WHERE kb_id=$1 ORDER BY created_at, document_id",
                kb_id,
            )
            chunks = await conn.fetch(
                "SELECT para_id, kb_id, document_id, version_id, payload, updated_at FROM kb_paragraphs WHERE kb_id=$1 ORDER BY document_id, row_id",
                kb_id,
            )
            glossary = await conn.fetch(
                "SELECT kb_id, document_id, version_id, term, term_key, expansion, chunk_id, origin, created_at FROM kb_glossary WHERE kb_id=$1 ORDER BY document_id, term_key",
                kb_id,
            )
        return {
            "knowledge_base": dict(knowledge_base),
            "documents": [dict(row) for row in documents],
            "chunks": [dict(row) for row in chunks],
            "glossary": [dict(row) for row in glossary],
        }

    async def delete_knowledge_base(self, kb_id: UUID) -> tuple[int, list[str]]:
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                source_paths = await conn.fetch(
                    "SELECT source_path FROM kb_documents WHERE kb_id=$1", kb_id
                )
                count = await conn.fetchval(
                    "SELECT count(*) FROM kb_documents WHERE kb_id=$1", kb_id
                )
                await conn.execute("DELETE FROM kb_knowledge_bases WHERE kb_id=$1", kb_id)
        return int(count or 0), [row["source_path"] for row in source_paths]

    async def clear_knowledge_base_documents(self, kb_id: UUID) -> tuple[int, list[str]]:
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                source_paths = await conn.fetch("SELECT source_path FROM kb_documents WHERE kb_id=$1", kb_id)
                count = await conn.fetchval("SELECT count(*) FROM kb_documents WHERE kb_id=$1", kb_id)
                await conn.execute("DELETE FROM kb_documents WHERE kb_id=$1", kb_id)
        return int(count or 0), [row["source_path"] for row in source_paths]

    async def clear_documents(self) -> int:
        async with self.pool.acquire() as conn:
            count = await conn.fetchval("SELECT count(*) FROM kb_documents")
            await conn.execute("DELETE FROM kb_documents")
            return int(count)
