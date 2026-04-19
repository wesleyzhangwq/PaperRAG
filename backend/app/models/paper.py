"""SQLAlchemy ORM models: Paper & Chunk."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.mysql import Base


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    paper_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    title: Mapped[str] = mapped_column(String(1024), nullable=False)
    authors: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    year: Mapped[int] = mapped_column(Integer, index=True)
    primary_category: Mapped[str] = mapped_column(String(32), index=True)
    categories: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    doi: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    abstract: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    pdf_url: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    pdf_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    entry_id: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    published: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    updated: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    ingest_status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    ingest_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    num_chunks: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("paper_id", "chunk_index", name="uq_paper_chunkidx"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chunk_id: Mapped[str] = mapped_column(String(96), unique=True, index=True)
    paper_id: Mapped[str] = mapped_column(String(64), ForeignKey("papers.paper_id"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    chunk_text: Mapped[str] = mapped_column(Text)
    page_num: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    n_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    paper: Mapped["Paper"] = relationship(back_populates="chunks")
