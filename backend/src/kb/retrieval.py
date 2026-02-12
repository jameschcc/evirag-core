# ==============================================================================
# EviRAG — Enterprise-grade High-performance Offline Knowledge Base with
# Traceable Retrieval and Generation Architecture
#         DOI: 10.3969/j.issn.1671-3370.2026.02.032 (S.Chang) https://d.wanfangdata.com.cn/periodical/zgxxj202602032
# ==============================================================================

from __future__ import annotations
import math
import re
from collections import defaultdict
from .contracts import Chunk, Evidence, RetrievalHit, RetrievalResult, SearchScope

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[一-鿿]+")
_CJK_PATTERN = re.compile(r"[一-鿿]+")
_IDENT_SPLIT = re.compile(r"_+")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _identifier_terms(segment: str) -> set[str]:
    """拆分一个标识符：``isup_iam_parm`` → isup_iam_parm / isup / iam / parm。

    代码库里的缩写几乎都长在标识符里（``ISUP_IAM``、``tali_iam``），不拆下划线与
    驼峰的话，问 ``IAM`` 只能命中注释，命不中定义。
    """
    terms: set[str] = set()
    for part in _IDENT_SPLIT.split(segment):
        if not part:
            continue
        terms.update(piece.lower() for piece in _CAMEL_BOUNDARY.split(part) if piece)
    terms.add(segment.lower())
    return terms


def tokenize(text: str) -> set[str]:
    tokens: set[str] = set()
    for segment in _TOKEN_PATTERN.findall(text):
        if not _CJK_PATTERN.fullmatch(segment):
            tokens.update(_identifier_terms(segment))
            continue
        if len(segment) == 1:
            tokens.add(segment)
            continue
        tokens.update(segment[index:index + 2] for index in range(len(segment) - 1))
        if len(segment) >= 3:
            tokens.update(segment[index:index + 3] for index in range(len(segment) - 2))
    return tokens


def is_question_example(text: str, query: str) -> bool:
    query_terms = tokenize(query)
    if not query_terms:
        return False
    for line in text.splitlines():
        if "question" not in line.lower():
            continue
        overlap = len(query_terms & tokenize(line)) / len(query_terms)
        if overlap >= 0.6:
            return True
    return False


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = left_norm = right_norm = 0.0
    for a, b in zip(left, right):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if not left_norm or not right_norm:
        return 0.0
    return dot / math.sqrt(left_norm * right_norm)


def weighted_rrf(rank_lists: dict[str, list[str]], weights: dict[str, float] | None = None, rank_constant: int = 60) -> list[tuple[str, float]]:
    weights = weights or {name: 1.0 for name in rank_lists}
    scores: defaultdict[str, float] = defaultdict(float)
    for name, ids in rank_lists.items():
        weight = weights.get(name, 1.0)
        for rank, item_id in enumerate(ids, start=1):
            scores[item_id] += weight / (rank_constant + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
def _chunk_order_key(chunk: Chunk) -> tuple:
    if chunk.ordinal is not None:
        return (0, chunk.ordinal, 0)
    span = chunk.source_spans[0] if chunk.source_spans else None
    page = span.locator.page if span and span.locator.page is not None else 0
    start = span.text_start if span else 0
    return (1, page, start)
class InMemoryHybridRetriever:
    def __init__(
        self,
        pipeline_revision: str = "prototype-v1",
        min_score: float = 0.15,
        dense_min_score: float = 0.45,
        lexical_weight: float = 0.5,
        dense_weight: float = 0.5,
    ) -> None:
        self.pipeline_revision = pipeline_revision
        self.min_score = min_score
        self.dense_min_score = dense_min_score
        self.lexical_weight = lexical_weight
        self.dense_weight = dense_weight
        self.embedding_revision: str | None = None
        self._chunks: dict[str, Chunk] = {}
        self._token_cache: dict[str, set[str]] = {}
    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            key = str(chunk.chunk_id)
            self._chunks[key] = chunk
            self._token_cache.pop(key, None)
    def clear(self) -> None:
        self._chunks.clear()
        self._token_cache.clear()
    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks.values())
    def chunks_for_document(self, document_id) -> list[Chunk]:
        return [chunk for chunk in self._chunks.values() if chunk.document_id == document_id]
    def remove_document_chunks(self, document_id) -> None:
        for chunk_id in [key for key, chunk in self._chunks.items() if chunk.document_id == document_id]:
            self._chunks.pop(chunk_id, None)
            self._token_cache.pop(chunk_id, None)
    def ordered_document_chunks(self, document_id) -> list[Chunk]:
        return sorted(self.chunks_for_document(document_id), key=_chunk_order_key)
    def _tokens_for(self, chunk: Chunk) -> set[str]:
        key = str(chunk.chunk_id)
        cached = self._token_cache.get(key)
        if cached is None:
            # 切片数上万时分词缓存会吃掉数百 MB，超过上限直接清空重建。
            if len(self._token_cache) >= 25_000:
                self._token_cache.clear()
            cached = tokenize(chunk.text)
            self._token_cache[key] = cached
        return cached
    def neighbours(self, chunk: Chunk, direction: str = "nearby", count: int = 1) -> list[Chunk]:
        if count <= 0 or direction not in {"prev", "next", "nearby"}:
            return []
        ordered = self.ordered_document_chunks(chunk.document_id)
        target = str(chunk.chunk_id)
        for index, item in enumerate(ordered):
            if str(item.chunk_id) != target:
                continue
            if direction == "prev":
                return ordered[max(0, index - count):index]
            if direction == "next":
                return ordered[index + 1:index + 1 + count]
            nearby = [
                (candidate_index, ordered[candidate_index])
                for candidate_index in (*range(max(0, index - count), index), *range(index + 1, min(len(ordered), index + count + 1)))
            ]
            nearby.sort(key=lambda item: (abs(item[0] - index), 0 if item[0] < index else 1))
            selected = sorted(candidate_index for candidate_index, _ in nearby[:count])
            return [ordered[candidate_index] for candidate_index in selected]
        return []
    def search(
        self,
        query: str,
        scope: SearchScope,
        limit: int = 5,
        lexical_query: str | None = None,
        lexical_queries: list[str] | None = None,
        query_embedding: list[float] | None = None,
        query_embeddings: list[list[float]] | None = None,
        source_kinds: list[str] | None = None,
        dense_candidates: list[tuple[str, float]] | None = None,
        dense_available_count: int = 0,
        dense_missing_count: int = 0,
    ) -> RetrievalResult:
        allowed_kb_ids = {str(item) for item in (scope.kb_ids if scope is not None else [])}
        allowed_kinds = {str(kind) for kind in (source_kinds or ["document", "code"])}
        visible = [
            chunk for chunk in self._chunks.values()
            if chunk.search_state == "published"
            and chunk.source_kind in allowed_kinds
            and (not allowed_kb_ids or str(chunk.kb_id) in allowed_kb_ids)
        ]
        queries = lexical_queries or [lexical_query or query]
        query_terms = [tokenize(item) for item in queries]
        terms: set[str] = set().union(*query_terms) if query_terms else set()
        chunk_terms = {str(chunk.chunk_id): self._tokens_for(chunk) for chunk in visible}
        # 关键词得分按 IDF 加权：语料里根本不出现的词（纯代码库里的中文问句）
        # 不计入分母，只在少数文件出现的缩写（IAM）权重最高；否则
        # 「IAM消息格式」= 1/6 会被稀释到 0.15 阈值以下。
        document_frequency: dict[str, int] = {}
        for item_terms in chunk_terms.values():
            for term in item_terms & terms:
                document_frequency[term] = document_frequency.get(term, 0) + 1
        total_documents = max(len(chunk_terms), 1)
        idf = {
            term: math.log(1 + total_documents / df)
            for term, df in document_frequency.items()
        }
        query_weights = [
            {term: idf[term] for term in item if term in idf}
            for item in query_terms
        ]
        lexical_scores = {
            chunk_id: sum(idf[term] for term in item_terms & terms if term in idf)
            for chunk_id, item_terms in chunk_terms.items()
        }
        lexical = sorted(visible, key=lambda chunk: (-lexical_scores[str(chunk.chunk_id)], str(chunk.chunk_id)))
        lexical_ids = [str(chunk.chunk_id) for chunk in lexical if lexical_scores[str(chunk.chunk_id)] > 0][:100]

        query_vectors = [vector for vector in (query_embeddings or ([query_embedding] if query_embedding else [])) if vector]
        dense_scores: dict[str, float] = {}
        if dense_candidates is not None:
            # 外部（pgvector）已按同一批查询向量做过精确扫描，并取了每个片段在
            # 所有查询向量上的最大 cosine；这里只复现内存路径的阈值过滤与
            # top-100 截断，保证两条后端的结果集完全一致。
            dense_scores = {str(item_id): float(score) for item_id, score in dense_candidates}
            dense_ids = [
                item_id
                for item_id, score in sorted(dense_scores.items(), key=lambda item: (-item[1], item[0]))
                if score >= self.dense_min_score
            ][:100]
            dense_active = dense_available_count > 0
            missing = dense_missing_count
        else:
            embedded = [
                chunk for chunk in visible
                if chunk.embedding and query_vectors
                and (self.embedding_revision is None or chunk.embedding_revision == self.embedding_revision)
                and len(chunk.embedding) == len(query_vectors[0])
            ]
            if embedded:
                for chunk in embedded:
                    dense_scores[str(chunk.chunk_id)] = max(cosine(vector, chunk.embedding) for vector in query_vectors)
            dense_ids = [
                item_id
                for item_id, score in sorted(dense_scores.items(), key=lambda item: (-item[1], item[0]))
                if score >= self.dense_min_score
            ][:100]
            dense_active = bool(embedded)
            missing = len(visible) - len(embedded)

        fused = weighted_rrf(
            {"bm25": lexical_ids, "dense": dense_ids},
            {"bm25": self.lexical_weight, "dense": self.dense_weight},
        )
        lexical_rank = {item_id: index for index, item_id in enumerate(lexical_ids, start=1)}
        dense_rank = {item_id: index for index, item_id in enumerate(dense_ids, start=1)}
        hits: list[RetrievalHit] = []
        scores: dict[str, tuple[float, float]] = {}
        by_id = {str(chunk.chunk_id): chunk for chunk in visible}
        # 融合窗口放宽到 60：窗口只取融合前 20/40 条时，向量假阳性会把排在
        # 队尾的关键词真命中整段挤出候选，重排再正确也救不回来。
        for item_id, fusion_score in fused[: max(limit * 12, 60)]:
            chunk = by_id[item_id]
            item_terms = chunk_terms[item_id]
            lexical_score = max(
                (
                    0.0
                    if is_question_example(chunk.text, item)
                    else self._weighted_coverage(weights, item_terms)
                )
                for item, weights in zip(queries, query_weights)
            )
            semantic_score = dense_scores.get(item_id, 0.0)
            scores[item_id] = (lexical_score, semantic_score)
            hits.append(RetrievalHit(chunk=chunk, bm25_rank=lexical_rank.get(item_id), dense_rank=dense_rank.get(item_id), fusion_score=fusion_score, rerank_score=max(lexical_score, semantic_score)))
        hits.sort(key=lambda hit: (-float(hit.rerank_score or 0), -hit.fusion_score, str(hit.chunk.chunk_id)))

        evidence = []
        for hit in hits:
            item_id = str(hit.chunk.chunk_id)
            lexical_score, semantic_score = scores[item_id]
            if lexical_score < self.min_score and semantic_score < self.dense_min_score:
                continue
            evidence.append(Evidence(evidence_id=f"E{len(evidence) + 1}", chunk=hit.chunk, score=float(hit.rerank_score or 0.0)))
            if len(evidence) >= limit:
                break

        warnings: list[str] = []
        dropped_terms = [term for term in terms if term not in idf]
        if dropped_terms and idf:
            warnings.append(
                f"{len(dropped_terms)} 个查询词在语料中未出现，已从关键词计分中剔除。"
            )
        if dense_active:
            if missing:
                warnings.append(f"{missing} 个片段缺少向量，已跳过向量召回。")
        elif query_vectors:
            warnings.append("查询已向量化，但没有可用的片段向量，仍使用关键词候选。")
        else:
            warnings.append("真实向量召回未启用，当前使用关键词候选。")
        if not evidence:
            warnings.append(f"没有证据达到最低相关度（关键词 {self.min_score:.2f} / 向量 {self.dense_min_score:.2f}）。")
        degraded = not dense_active or missing > 0
        return RetrievalResult(
            evidence=evidence,
            warnings=warnings,
            pipeline_revision=self.pipeline_revision,
            degraded=degraded,
            retrieval_method="hybrid" if dense_active else "lexical",
            rerank_method="hybrid_fusion" if dense_active else "lexical",
            warnings_detail="; ".join(warnings),
        )
    @staticmethod
    def _weighted_coverage(weights: dict[str, float], text_terms: set[str]) -> float:
        if not weights:
            return 0.0
        total = sum(weights.values())
        if total <= 0:
            return 0.0
        matched = sum(weight for term, weight in weights.items() if term in text_terms)
        return matched / total
