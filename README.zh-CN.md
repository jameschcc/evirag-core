# EviRAG Core 部署指南

## 目录说明

| 目录 | 内容 |
| --- | --- |
| `backend/` | FastAPI 服务、Celery worker、PostgreSQL 目录库。`src/kb/` 下是检索、切片、生成、术语库等模块，`main.py` 是 API 入口，`workers.py` 是异步索引任务 |
| `model-service/` | 小型确定性模型适配器，提供与正式嵌入/重排服务一致的接口契约，不加载真实权重 |
| `web/` | 静态控制台：上传、检索、问答、证据、术语库、设置；由 nginx 直接托管，无需构建 |
| `deploy/` | Docker Compose 编排、nginx 网关、`.env.example` 环境变量模板、可选 Kubernetes 清单 |

## 需要准备的模型

服务本身不下载模型权重，需要提前准备以下端点：

| 用途 | 默认模型 | 要求 |
| --- | --- | --- |
| 嵌入（向量化） | `bge-m3` | Ollama `/api/embed`，1024 维、归一化 |
| 对话/生成 | `qwen3:14b` | Ollama 或任意 OpenAI 兼容 `/chat/completions` |
| 重排（可选） | `bge-reranker-v2-m3` | Infinity 兼容 `/rerank`；不配置则用 LLM 重排 |

用 Ollama 时先拉取模型：

```bash
ollama pull bge-m3
ollama pull qwen3:14b
# 可选
ollama pull bge-reranker-v2-m3
```

`qwen3:14b` 约需 10 GB 内存或显存；资源有限可换更小的 Ollama 模型，或改用 OpenAI 兼容的云端接口，在 `.env` 中调整 `KB_AI_MODEL` 与 `KB_AI_BASE_URL`。

## 环境要求

- Docker Engine 20.10+ 与 Compose v2（或 Docker Desktop）
- 一个 Ollama 实例，或兼容 OpenAI 的 `/chat/completions` 与 `/embeddings` 端点
- 端口 `8080`（控制台/API）；本地 Ollama 另需 `11434`

## 部署步骤

1. 安装 Docker 与 Compose，确认 `docker compose version` 可用。
2. 准备 Ollama 并拉取上表模型；确认 `curl http://<ollama地址>:11434/api/tags` 能返回模型列表。
3. 进入 `deploy/`，复制环境变量模板并填写模型端点：

   ```bash
   cd deploy
   cp .env.example .env
   # 编辑 .env，填入 KB_AI_* 与 KB_EMBEDDING_* 的地址、模型名与可选 API Key
   ```

4. 启动整套服务：

   ```bash
   docker compose up -d --build
   ```

5. 验证：

   ```bash
   curl http://localhost:8080/health/ready
   ```

   返回 `ready` 后，浏览器打开控制台，上传一份文档，等状态变为 `ready`，再在检索页提问。

## 访问地址

| 地址 | 用途 |
| --- | --- |
| <http://localhost:8080> | 控制台 |
| <http://localhost:8080/docs> | API 文档 |
| <http://localhost:8080/health/ready> | 健康检查 |

## 配置

`.env.example` 已列出主要开关，复制为 `.env` 后按需修改：

- `KB_AI_PROVIDER`：`ollama`（默认）或 `openai`
- `KB_AI_BASE_URL` / `KB_AI_MODEL` / `KB_AI_API_KEY`：对话模型端点
- `KB_EMBEDDING_PROVIDER` / `KB_EMBEDDING_BASE_URL` / `KB_EMBEDDING_MODEL`：嵌入端点
- `KB_HTTP_PORT`：控制台端口，默认 `8080`
- `POSTGRES_PASSWORD`：数据库密码

使用 Ollama 时确认目标机可访问其 `11434` 端口；容器内默认通过 `host.docker.internal` 访问宿主机。

## 使用

1. 打开控制台，上传 PDF、DOCX、PPTX 或 UTF-8 文本。
2. 等待文档状态变为 `ready`（索引由 Celery worker 在后台完成）。
3. 在检索页提问，先看证据，再看带引用编号的答案。

## 常用命令

```bash
docker compose ps          # 查看服务状态
docker compose logs -f api # 查看 API 日志
docker compose down        # 停止并移除容器
docker compose down -v     # 同时删除数据卷（会清空数据）
```

## 说明

- 默认不下载模型权重，需自行准备 Ollama 或兼容端点。
- 控制台无登录鉴权，仅适用于可信内网；不要直接暴露到公网。
