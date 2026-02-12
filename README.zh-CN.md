# EviRAG Core 部署指南

## 环境要求

- Docker Engine 20.10+ 与 Compose v2（或 Docker Desktop）
- 一个 Ollama 实例，或兼容 OpenAI 的 `/chat/completions` 与 `/embeddings` 端点
- 端口 `8080`（控制台/API）；本地 Ollama 另需 `11434`

## 快速开始

```bash
cd deploy
cp .env.example .env
# 编辑 .env，填入 KB_AI_* 与 KB_EMBEDDING_* 的模型端点
docker compose up -d --build
```

启动后访问：

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

模型不随包提供。使用 Ollama 时确认目标机可访问其 `11434` 端口；容器内默认通过 `host.docker.internal` 访问宿主机。

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
