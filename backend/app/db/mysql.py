"""SQLAlchemy engine / Session / Base for MySQL."""
from __future__ import annotations

import logging

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import get_settings

log = logging.getLogger("app.db.mysql")

settings = get_settings()

engine = create_engine(settings.sqlalchemy_url, pool_pre_ping=True, echo=False)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    """Import models and create all tables."""
    from app.models import paper  # noqa: F401 register tables
    from app.models import chat_history  # noqa: F401 register tables
    from app.models import conversation  # noqa: F401 register tables
    from app.models import feedback  # noqa: F401 register tables
    from app.models import upload_job  # noqa: F401 register tables
    Base.metadata.create_all(bind=engine)
    _migrate_chat_history()
    _migrate_ingestion_schema()


def _migrate_chat_history() -> None:
    """Idempotently add new columns to chat_history if missing."""
    try:
        insp = inspect(engine)
        if "chat_history" not in insp.get_table_names():
            return
        existing = {c["name"] for c in insp.get_columns("chat_history")}
        with engine.begin() as conn:
            if "conversation_id" not in existing:
                conn.execute(text(
                    "ALTER TABLE chat_history ADD COLUMN conversation_id "
                    "VARCHAR(64) NOT NULL DEFAULT ''"
                ))
                conn.execute(text(
                    "CREATE INDEX ix_chat_history_conversation_id "
                    "ON chat_history (conversation_id)"
                ))
                # backfill conversation_id from session_id for legacy rows
                conn.execute(text(
                    "UPDATE chat_history SET conversation_id = session_id "
                    "WHERE conversation_id = ''"
                ))
            if "sources_json" not in existing:
                conn.execute(text(
                    "ALTER TABLE chat_history ADD COLUMN sources_json TEXT NULL"
                ))
            if "thinking_json" not in existing:
                conn.execute(text(
                    "ALTER TABLE chat_history ADD COLUMN thinking_json TEXT NULL"
                ))
    except Exception as e:
        log.warning("chat_history migration skipped: %s", e)


def _add_missing_columns(table: str, definitions: dict[str, str]) -> None:
    """Apply the project's lightweight, idempotent additive migrations."""
    insp = inspect(engine)
    if table not in insp.get_table_names():
        return
    existing = {column["name"] for column in insp.get_columns(table)}
    with engine.begin() as conn:
        for name, ddl in definitions.items():
            if name not in existing:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))


def _ensure_index(table: str, index_name: str, column: str) -> None:
    insp = inspect(engine)
    if table not in insp.get_table_names():
        return
    existing = {item["name"] for item in insp.get_indexes(table)}
    if index_name not in existing:
        with engine.begin() as conn:
            conn.execute(
                text(f"CREATE INDEX {index_name} ON {table} ({column})")
            )


def _migrate_ingestion_schema() -> None:
    """Add provenance and progress fields for heterogeneous ingestion."""
    try:
        _add_missing_columns(
            "papers",
            {
                "source_kind": "VARCHAR(32) NOT NULL DEFAULT 'arxiv'",
                "media_type": "VARCHAR(128) NULL",
                "content_hash": "VARCHAR(64) NULL",
                "original_filename": "VARCHAR(512) NULL",
                "ingest_metadata": "JSON NULL",
            },
        )
        _add_missing_columns(
            "chunks",
            {
                "modality": "VARCHAR(32) NOT NULL DEFAULT 'text'",
                "section": "VARCHAR(512) NULL",
                "source_locator": "JSON NULL",
            },
        )
        _add_missing_columns(
            "upload_jobs",
            {
                "stage": "VARCHAR(32) NOT NULL DEFAULT 'queued'",
                "progress": "INT NOT NULL DEFAULT 0",
                "source_kind": "VARCHAR(32) NOT NULL DEFAULT 'arxiv'",
                "media_type": "VARCHAR(128) NULL",
                "content_hash": "VARCHAR(64) NULL",
                "error_code": "VARCHAR(64) NULL",
                "warnings": "JSON NULL",
            },
        )
        _ensure_index("papers", "ix_papers_source_kind", "source_kind")
        _ensure_index("papers", "ix_papers_content_hash", "content_hash")
        _ensure_index("chunks", "ix_chunks_modality", "modality")
        _ensure_index("upload_jobs", "ix_upload_jobs_stage", "stage")
    except Exception as exc:
        log.warning("ingestion schema migration skipped: %s", exc)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
