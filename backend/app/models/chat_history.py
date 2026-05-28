"""SQLAlchemy ORM model: ChatHistory."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Enum, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.mysql import Base


class ChatHistory(Base):
    __tablename__ = "chat_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # conversation_id is the authoritative grouping key.
    # session_id kept for legacy compatibility; mirrors conversation_id.
    conversation_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False, default="")
    session_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    role: Mapped[str] = mapped_column(Enum("user", "assistant", name="chat_role"), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # JSON-serialized sources/traces for full reload
    sources_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    thinking_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
