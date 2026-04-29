"""Centralized settings loaded from env (.env)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- MySQL ---
    mysql_host: Optional[str] = None
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = "root"
    mysql_database: str = "paperrag"
    mysql_url: Optional[str] = None

    # --- Qdrant ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    qdrant_collection: str = "paperrag"

    # --- Cloud LLM (MiniMax) ---
    llm_model: str = "MiniMax-M2.7"
    llm_api_base: str = "https://api.minimax.chat/v1"
    llm_api_key: Optional[str] = None

    # --- Cloud Embedding (Alibaba) ---
    embedding_model: str = "text-embedding-v4"
    embedding_api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_api_key: Optional[str] = None

    # --- RAG ---
    chunk_strategy: str = "v2"
    chunk_size: int = 800
    chunk_overlap: int = 100
    chunk_min_chars: int = 80
    chunk_noise_symbol_ratio: float = 0.35
    chunk_drop_references: bool = False
    retrieval_k: int = 8
    final_context_k: int = 3

    # --- Hybrid retrieval (vector oversample + BM25 fusion / rerank) ---
    hybrid_retrieval_enabled: bool = True
    hybrid_oversample: float = 2.5
    hybrid_alpha: float = 0.72
    hybrid_max_fetch: int = 64

    # --- Caching ---
    cache_retrieval_enabled: bool = True
    cache_retrieval_ttl_sec: int = 180
    cache_retrieval_max_entries: int = 256
    cache_embedding_enabled: bool = True
    cache_embedding_max_entries: int = 512

    # --- Retries (embedding HTTP) ---
    http_retry_max_attempts: int = 4
    http_retry_backoff_base_sec: float = 0.45

    # --- LLM ---
    llm_max_retries: int = 2

    # --- Observability ---
    observability_json_logs: bool = True

    # --- Data paths ---
    data_dir: str = str(PROJECT_ROOT / "data")
    pdf_dir: str = str(PROJECT_ROOT / "data" / "pdfs")
    metadata_json: str = str(PROJECT_ROOT / "data" / "metadata_filtered.json")

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:5173,http://localhost:8080"

    @property
    def sqlalchemy_url(self) -> str:
        """Build MySQL URL from MYSQL_URL or MYSQL_HOST + credentials."""
        if self.mysql_url:
            return self.mysql_url
        if self.mysql_host:
            return (
                f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
                f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}?charset=utf8mb4"
            )
        raise ValueError(
            "MySQL is required: set MYSQL_URL or MYSQL_HOST in .env (see .env.example)."
        )

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
