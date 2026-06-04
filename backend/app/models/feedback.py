"""SQLAlchemy ORM model: user answer feedback."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.mysql import Base


class AnswerFeedback(Base):
    __tablename__ = "answer_feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    message_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    vote: Mapped[str] = mapped_column(String(8), nullable=False)
    reason: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    comment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
