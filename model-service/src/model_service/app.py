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
import hashlib
import math
import re
from fastapi import FastAPI
from pydantic import BaseModel, Field
app = FastAPI(title="Knowledge Base Model Service", version="0.1.0")
class Item(BaseModel):
    id: str
    text: str
class EmbeddingRequest(BaseModel):
    items: list[Item] = Field(min_length=1, max_length=128)
    purpose: str = "document"
    model_revision: str = "stub-embedding-v1"
class RerankRequest(BaseModel):
    query: str
    items: list[Item] = Field(min_length=1, max_length=128)
    model_revision: str = "stub-reranker-v1"
def embedding(text: str, dimensions: int = 64) -> list[float]:
    vector = [0.0] * dimensions
    for token in re.findall(r"[\w\u4e00-\u9fff]+", text.lower()):
        digest = hashlib.sha256(token.encode()).digest()
        vector[int.from_bytes(digest[:4], "big") % dimensions] += 1 if digest[4] % 2 else -1
    norm = math.sqrt(sum(value * value for value in vector)) or 1
    return [value / norm for value in vector]
@app.get("/health/live")
def live() -> dict:
    return {"status": "ok", "backend": "stub"}
@app.get("/health/ready")
def ready() -> dict:
    return {"status": "ready", "revision": "stub-embedding-v1"}
@app.post("/internal/v1/embeddings")
def embeddings(request: EmbeddingRequest) -> dict:
    return {"items": [{"id": item.id, "vector": embedding(item.text)} for item in request.items], "dimension": 64, "model_revision": request.model_revision}
@app.post("/internal/v1/rerank")
def rerank(request: RerankRequest) -> dict:
    query_terms = set(re.findall(r"[\w\u4e00-\u9fff]+", request.query.lower()))
    scored = []
    for item in request.items:
        text_terms = set(re.findall(r"[\w\u4e00-\u9fff]+", item.text.lower()))
        scored.append({"id": item.id, "score": len(query_terms & text_terms) / max(len(query_terms), 1)})
    return {"items": sorted(scored, key=lambda item: (-item["score"], item["id"])), "model_revision": request.model_revision}
