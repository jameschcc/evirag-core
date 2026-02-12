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
from datetime import datetime, timezone
from typing import Literal
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, field_validator

DEFAULT_KB_ID = UUID("00000000-0000-0000-0000-000000000001")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    slug: str | None = Field(default=None, min_length=1, max_length=80)
    description: str = Field(default="", max_length=2000)


class KnowledgeBaseResponse(BaseModel):
    kb_id: UUID
    slug: str
    name: str
    description: str = ""
    status: Literal["active", "archived"] = "active"
    role: Literal["owner", "admin", "editor", "viewer"] = "viewer"
    document_count: int = 0
    chunk_count: int = 0
    created_at: datetime = Field(default_factory=utc_now)


class KnowledgeBaseMember(BaseModel):
    user_id: UUID
    role: Literal["owner", "admin", "editor", "viewer"]


class SourceLocator(BaseModel):
    kind: Literal["text", "pdf", "docx", "pptx"] = "text"
    page: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    rects: list[list[float]] = Field(default_factory=list)
    coordinate_system: str | None = None
class SourceSpan(BaseModel):
    block_id: UUID
    text_start: int
    text_end: int
    locator: SourceLocator
class Chunk(BaseModel):
    chunk_id: UUID = Field(default_factory=uuid4)
    kb_id: UUID = DEFAULT_KB_ID
    document_id: UUID
    version_id: UUID
    title: str
    text: str
    # document = 文档正文（PDF/DOCX/PPTX/纯文本），code = 代码文件。检索时按数据源过滤。
    source_kind: Literal["document", "code"] = "document"
    source_spans: list[SourceSpan] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    embedding: list[float] = Field(default_factory=list)
    embedding_revision: str = ""
    ordinal: int | None = None
    search_state: Literal["staged", "published", "deleted"] = "published"
    pipeline_revision: str
class SearchScope(BaseModel):
    user_id: UUID = Field(default_factory=uuid4)
    kb_ids: list[UUID] = Field(default_factory=list)


class GlossaryEntry(BaseModel):
    term: str
    expansion: str
    kb_id: UUID | None = None
    document_id: UUID | None = None
    version_id: UUID | None = None
    chunk_id: UUID | None = None
    origin: str = "ai"


class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    kb_ids: list[UUID] = Field(default_factory=list)
    limit: int = Field(default=5, ge=1, le=20)
    auto_semantic_analysis: bool = True
    iterative_retrieval: bool = True
    # 数据源至少一种：document（文档正文）/ code（代码文件）。默认两者都检索。
    source_kinds: list[Literal["document", "code"]] = Field(
        default_factory=lambda: ["document", "code"]
    )
    progress_id: str | None = Field(default=None, max_length=64)

    @field_validator("source_kinds")
    @classmethod
    def require_at_least_one_source(cls, value: list[str]) -> list[str]:
        kinds = list(dict.fromkeys(value))
        if not kinds:
            raise ValueError("source_kinds 至少需要包含 document 或 code 之一")
        return kinds

class QueryAnalysis(BaseModel):
    enabled: bool = True
    original_question: str
    # 多轮会话中由查询理解模型（或规则兜底）改写出的独立问题，用于实测检索；
    # 空字符串表示单轮提问，直接用 original_question 检索。
    resolved_question: str = ""
    retrieval_query: str
    keywords: list[str] = Field(default_factory=list)
    expansions: list[str] = Field(default_factory=list)
    identifiers: list[str] = Field(default_factory=list)
    retrieval_queries: list[str] = Field(default_factory=list)
    glossary: list[GlossaryEntry] = Field(default_factory=list)
    # L1 意图判定：只有明确的闲聊或元问题才允许跳过检索。默认值必须是
    # kb_question——旧缓存、规则兜底与关闭语义分析的路径都会落到这里，
    # 任何缺失或非法意图都必须退回「照常检索」，不能静默丢掉检索。
    intent: Literal["kb_question", "chitchat", "meta"] = "kb_question"
    analysis_method: Literal["ai", "rules", "fallback", "disabled"] = "rules"
    warnings: list[str] = Field(default_factory=list)
    revision: str = "query-planner-v1"
class RetrievalHit(BaseModel):
    chunk: Chunk
    bm25_rank: int | None = None
    dense_rank: int | None = None
    fusion_score: float = 0.0
    rerank_score: float | None = None
class Evidence(BaseModel):
    evidence_id: str
    chunk: Chunk
    score: float
class RetrievalResult(BaseModel):
    retrieval_method: str = "lexical"
    rerank_method: str = "lexical"
    warnings_detail: str = ""
    rerank_trace: list[dict] = Field(default_factory=list)
    retrieval_plan: dict = Field(default_factory=dict)
    iterative_trace: list[dict] = Field(default_factory=list)
    evidence: list[Evidence]
    # 语义分析判定无需检索时为 True：没有候选、没有重排，前端也不应渲染证据区。
    skipped: bool = False
    skip_reason: str = ""
    degraded: bool = False
    warnings: list[str] = Field(default_factory=list)
    raw_error_text: str | None = None
    pipeline_revision: str
    query_analysis: QueryAnalysis | None = None
class AnswerRequest(SearchRequest):
    conversation_id: UUID | None = None
    mode: Literal["extractive", "grounded"] = "grounded"
    # conservative 只发布证据直接支持的事实；divergent 允许基于证据做推断与延伸，
    # 但每条条目仍必须绑定本次实际送入模型的证据 ID。
    generation_style: Literal["conservative", "divergent"] = "conservative"
    query_analysis: QueryAnalysis | None = None
class Claim(BaseModel):
    id: str
    text: str
    source_ids: list[str] = Field(default_factory=list)
    # 发散模式下用于区分直接依据（fact）与推断/类比/影响/假设；保守模式恒为 fact。
    kind: str = "fact"
class AnswerResponse(BaseModel):
    request_id: UUID = Field(default_factory=uuid4)
    answer_id: UUID = Field(default_factory=uuid4)
    # 多轮会话归属；单轮问答为 None。
    conversation_id: UUID | None = None
    status: Literal["answered", "partial", "insufficient_evidence", "conflicting_evidence", "no_retrieval_needed"]
    answer: str
    confidence: float = Field(ge=0, le=1)
    confidence_kind: str = "evidence_coverage"
    verification_level: Literal["citation_only", "extractive", "semantic_checked"]
    generation_style: Literal["conservative", "divergent"] = "conservative"
    claims: list[Claim] = Field(default_factory=list)
    sources: list[dict] = Field(default_factory=list)
    suggested_actions: list[str] = Field(default_factory=list)
    degraded: bool = False
    warnings: list[str] = Field(default_factory=list)
    query_analysis: QueryAnalysis | None = None
class DocumentResponse(BaseModel):
    kb_id: UUID = DEFAULT_KB_ID
    document_id: UUID
    version_id: UUID
    title: str
    chunk_count: int
    status: Literal["ready", "processing", "failed"]
    created_at: datetime = Field(default_factory=utc_now)
    format: str = "text"
    source_kind: Literal["document", "code"] = "document"
    error: str | None = None
    task_id: str | None = None
    glossary_state: int = 0

class IndexCallbackRequest(BaseModel):
    format: str
    source_kind: Literal["document", "code"] = "document"
    chunks: list[Chunk] = Field(default_factory=list)
    error: str | None = None
    task_id: str | None = None

class ClearKnowledgeBaseRequest(BaseModel):
    confirmation: Literal["CLEAR_KNOWLEDGE_BASE"]
