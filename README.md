# EviRAG Core

Offline, self-hosted retrieval-augmented generation for private document
collections: hybrid sparse/dense retrieval, query understanding, iterative
evidence completion and citation-gated answer generation.

Personal research project. Author: S.Chang.

## What is inside

| Path | Purpose |
| --- | --- |
| `backend/` | FastAPI service, Celery worker, PostgreSQL catalog, retrieval and generation pipeline |
| `model-service/` | Small deterministic model adapter used by the pipeline |
| `web/` | Static console: upload, search, evidence, glossary and settings |
| `deploy/` | Docker Compose stack, nginx gateway, optional Kubernetes manifest |

## Requirements

- Docker Engine 20.10+ with Compose v2 (or Docker Desktop).
- An Ollama instance or any OpenAI-compatible `/chat/completions` and `/embeddings` endpoint.
  Models are not bundled.
- Ports `8080` (console/API) and, if you use it, `11434` for a local Ollama.

## Quick start

```bash
cd deploy
cp .env.example .env     # set KB_AI_* and KB_EMBEDDING_* to your model endpoints
docker compose up -d --build
```

- Console: <http://localhost:8080>
- API docs: <http://localhost:8080/docs>
- Health: <http://localhost:8080/health/ready>

`deploy/.env.example` documents the main switches: chat provider and model,
embedding provider and model, optional reranker, HTTP port and Postgres
password. With `KB_AI_PROVIDER=ollama` (the default) the service talks to an
Ollama endpoint; with `openai` it uses any OpenAI-compatible base URL and key.
Timeouts and the generation budget are configurable.

## Using it

1. Open the console and upload PDF, DOCX, PPTX or UTF-8 text documents.
2. Wait for the document to reach `ready`; indexing runs in a Celery worker.
3. Ask a question on the search page. Evidence is shown first, then a grounded
   answer whose claims cite evidence ids.
4. Optional: extract a per-document glossary of abbreviations and tune the
   runtime settings on the settings page.

## Design notes

- BM25 and dense vectors (BGE-M3 by default) are fused with weighted RRF; a
  Cross-Encoder or LLM reranker is optional.
- Answers are generated only from supplied evidence; every claim must
  reference a valid evidence id, otherwise the answer is downgraded and marked
  as reference-only.
- Retrieval runs in rounds: the reranker may request neighbouring chunks
  (`prev-N`, `next-N`, `nearby-N`) and supplementary keywords, bounded by a
  configurable maximum.

## License

Private research project; no license is granted for redistribution or
commercial use.
