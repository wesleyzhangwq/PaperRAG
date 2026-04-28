"""Regression: embedding failure must not leave MySQL chunks without Qdrant vectors."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.mysql import Base
from app.models.paper import Chunk, Paper
from app.services import ingest as ingest_mod
from app.services.ingest import _ingest_one
from app.utils.chunker import PaperChunk


@pytest.fixture
def memory_db() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _minimal_record(tmp_pdf: Path) -> dict:
    return {
        "paper_id": "test.paper.1",
        "title": "Title",
        "authors": None,
        "year": 2024,
        "primary_category": "cs.AI",
        "categories": None,
        "doi": None,
        "abstract": None,
        "pdf_url": None,
        "pdf_path": str(tmp_pdf),
        "entry_id": None,
        "published": None,
        "updated": None,
    }


def test_embed_failure_first_ingest_leaves_no_chunk_rows(memory_db: Session, tmp_path: Path) -> None:
    pdf = tmp_path / "stub.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    mock_vs = MagicMock()
    mock_vs.add_texts.side_effect = RuntimeError("ollama unavailable")

    chunks = [PaperChunk(chunk_index=0, text="x" * 100, page_num=1)]

    with patch.object(ingest_mod, "get_vector_store", return_value=mock_vs):
        with patch.object(ingest_mod, "extract_pages", return_value=[(1, "y" * 200)]):
            with patch.object(ingest_mod, "chunk_pages", return_value=chunks):
                pid, status = _ingest_one(memory_db, _minimal_record(pdf), force=True)

    assert status == "failed"
    assert pid == "test.paper.1"
    assert memory_db.query(Chunk).filter(Chunk.paper_id == "test.paper.1").count() == 0
    mock_vs.add_texts.assert_called_once()
    mock_vs.delete.assert_not_called()


def test_embed_failure_reingest_preserves_mysql_chunks(memory_db: Session, tmp_path: Path) -> None:
    pdf = tmp_path / "stub.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    paper = Paper(
        paper_id="test.paper.2",
        title="T",
        year=2024,
        primary_category="cs.AI",
        ingest_status="ok",
        num_chunks=1,
    )
    memory_db.add(paper)
    memory_db.add(
        Chunk(
            chunk_id="test.paper.2::0",
            paper_id="test.paper.2",
            chunk_index=0,
            chunk_text="old text " * 20,
            page_num=1,
            n_tokens=10,
        )
    )
    memory_db.commit()

    mock_vs = MagicMock()
    mock_vs.add_texts.side_effect = ConnectionError("qdrant down")

    chunks = [PaperChunk(chunk_index=0, text="z" * 100, page_num=1)]

    rec = _minimal_record(pdf)
    rec["paper_id"] = "test.paper.2"

    with patch.object(ingest_mod, "get_vector_store", return_value=mock_vs):
        with patch.object(ingest_mod, "extract_pages", return_value=[(1, "w" * 200)]):
            with patch.object(ingest_mod, "chunk_pages", return_value=chunks):
                _, status = _ingest_one(memory_db, rec, force=True)

    assert status == "failed"
    rows = memory_db.query(Chunk).filter(Chunk.paper_id == "test.paper.2").all()
    assert len(rows) == 1
    assert rows[0].chunk_text.startswith("old text")
    mock_vs.delete.assert_not_called()


def test_success_reingest_deletes_stale_qdrant_ids(memory_db: Session, tmp_path: Path) -> None:
    pdf = tmp_path / "stub.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    paper = Paper(
        paper_id="test.paper.3",
        title="T",
        year=2024,
        primary_category="cs.AI",
        ingest_status="ok",
        num_chunks=2,
    )
    memory_db.add(paper)
    memory_db.add(
        Chunk(
            chunk_id="test.paper.3::0",
            paper_id="test.paper.3",
            chunk_index=0,
            chunk_text="a" * 100,
            page_num=1,
            n_tokens=10,
        )
    )
    memory_db.add(
        Chunk(
            chunk_id="test.paper.3::1",
            paper_id="test.paper.3",
            chunk_index=1,
            chunk_text="b" * 100,
            page_num=1,
            n_tokens=10,
        )
    )
    memory_db.commit()

    mock_vs = MagicMock()

    chunks = [PaperChunk(chunk_index=0, text="n" * 100, page_num=1)]

    rec = _minimal_record(pdf)
    rec["paper_id"] = "test.paper.3"

    with patch.object(ingest_mod, "get_vector_store", return_value=mock_vs):
        with patch.object(ingest_mod, "extract_pages", return_value=[(1, "m" * 200)]):
            with patch.object(ingest_mod, "chunk_pages", return_value=chunks):
                _, status = _ingest_one(memory_db, rec, force=True)

    assert status == "ok"
    mock_vs.add_texts.assert_called_once()
    mock_vs.delete.assert_called_once()
    assert mock_vs.delete.call_args.kwargs["ids"] == ["test.paper.3::1"]
    assert memory_db.query(Chunk).filter(Chunk.paper_id == "test.paper.3").count() == 1
