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

    # --- Ollama ---
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "gemma4:e4b"
    embedding_model: str = "bge-m3"

    # --- RAG ---
    chunk_size: int = 800
    chunk_overlap: int = 100
    retrieval_k: int = 8
    final_context_k: int = 3

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
