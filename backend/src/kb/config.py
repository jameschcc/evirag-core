# ==============================================================================
# EviRAG — Enterprise-grade High-performance Offline Knowledge Base with
# Traceable Retrieval and Generation Architecture
# Personal Research Project · Author: S.Chang (常帅)
# Paper: 常帅. 企业级高性能离网知识库可溯源检索与生成架构[J].
#        China Information World, 2026(2).
#        DOI: 10.3969/j.issn.1671-3370.2026.02.032
#        https://d.wanfangdata.com.cn/periodical/zgxxj202602032
# ==============================================================================

from functools import lru_cache
from uuid import UUID
from pydantic_settings import BaseSettings, SettingsConfigDict
class Settings(BaseSettings):
    git_tag: str = ""
    git_commit: str = ""
    app_name: str = "offline-knowledge-base"
    environment: str = "prototype"
    api_prefix: str = "/api/v1"
    model_service_url: str = "http://model-service:8100"
    ai_provider: str = "ollama"
    ai_base_url: str = ""
    ai_api_key: str = ""
    ai_model: str = ""
    # Legacy Ollama settings remain as fallbacks for older compose overrides.
    ollama_base_url: str = ""
    ollama_model: str = "qwen3:14b"
    embedding_provider: str = "ollama"
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_model: str = "bge-m3"
    embedding_timeout_seconds: float = 60.0
    embedding_batch_size: int = 16
    reranker_base_url: str = ""
    reranker_model: str = "bge-reranker-v2-m3"
    reranker_timeout_seconds: float = 20.0
    glossary_timeout_seconds: float = 120.0
    redis_url: str = "redis://redis:6379/0"
    postgres_dsn: str = "postgresql+asyncpg://kb:kb@postgres:5432/kb"
    elasticsearch_url: str = "http://elasticsearch:9200"
    pipeline_revision: str = "prototype-v1"
    max_upload_bytes: int = 50 * 1024 * 1024
    # 代码库批量入库：单文件、ZIP 体积、解压总量与条目上限。
    code_max_file_bytes: int = 2 * 1024 * 1024
    code_zip_max_bytes: int = 200 * 1024 * 1024
    code_zip_max_extracted_bytes: int = 500 * 1024 * 1024
    code_max_files_per_request: int = 300
    code_max_zip_entries: int = 2000
    max_context_chars: int = 12_000
    request_timeout_seconds: float = 20.0
    generation_timeout_seconds: float = 120.0
    ollama_num_predict: int = 4096
    # 发散型回答使用更高温度以获得更多探索方向；引用校验不因温度放松。
    divergent_temperature: float = 0.6
    structured_mode: str = "tool"
    tool_choice: str = "auto"
    tool_strict: bool = True
    retrieval_min_score: float = 0.15
    retrieval_dense_min_score: float = 0.45
    # 稠密召回后端：auto（pgvector 可用则用，否则内存）/ pgvector / memory。
    # 保留 memory 是为了做同库对照实验，不需要改代码即可回退。
    retrieval_backend: str = "auto"
    iterative_retrieval_enabled: bool = True
    iterative_max_rounds: int = 3
    neighbor_max_count: int = 3
    supplementary_keywords_max: int = 5
    api_internal_url: str = "http://api:8000"
    ingestion_dir: str = "/data/ingestion"
    default_user_id: UUID = UUID("00000000-0000-0000-0000-000000000010")
    auth_mode: str = "trusted_header"
    model_config = SettingsConfigDict(env_file=".env", env_prefix="KB_", extra="ignore")

    @property
    def effective_ai_base_url(self) -> str:
        return (self.ai_base_url or self.ollama_base_url).rstrip("/")

    @property
    def effective_ai_model(self) -> str:
        return self.ai_model or self.ollama_model

    @property
    def effective_embedding_base_url(self) -> str:
        return (self.embedding_base_url or self.ollama_base_url).rstrip("/")

    @property
    def effective_embedding_provider(self) -> str:
        return self.embedding_provider.strip().lower() or "ollama"

    @property
    def effective_ai_provider(self) -> str:
        return self.ai_provider.strip().lower() or "ollama"
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
