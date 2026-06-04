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


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
