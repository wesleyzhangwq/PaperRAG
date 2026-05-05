"""Ensure ingest does not drop MySQL chunks before Qdrant embed succeeds."""
from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.mysql import Base
from app.models.paper import Chunk, Paper  # noqa: F401 — register models
from app.services import ingest as ingest_mod
from app.services.ingest import _ingest_one
from app.utils.chunker import PaperChunk


class _FakeVS:
    def __init__(self, fail_on_add: bool = False) -> None:
        self.fail_on_add = fail_on_add
        self.deleted: list[str] = []
        self.add_calls = 0

    def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ) -> None:
        self.add_calls += 1
        if self.fail_on_add:
            raise RuntimeError("simulated embedding failure")

    def delete(self, ids: list[str]) -> None:
        self.deleted.extend(ids)


@pytest.fixture
def sqlite_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    try:
        yield s
    finally:
        s.close()


def test_embed_failure_leaves_existing_mysql_chunks(
    monkeypatch: pytest.MonkeyPatch, sqlite_session, tmp_path: Path
) -> None:
    fake = _FakeVS(fail_on_add=True)
    monkeypatch.setattr(ingest_mod, "get_vector_store", lambda: fake)

    monkeypatch.setattr(
        ingest_mod, "extract_pages", lambda _path: [(1, "body")]
    )
    monkeypatch.setattr(
        ingest_mod,
        "chunk_pages",
        lambda _pages: [
            PaperChunk(chunk_index=0, text="new chunk", page_num=1),
        ],
    )

    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    sqlite_session.add(
        Paper(
            paper_id="2301.00001",
            title="Old",
            year=2020,
            primary_category="cs.AI",
            ingest_status="ok",
            num_chunks=1,
        )
    )
    sqlite_session.add(
        Chunk(
            chunk_id="2301.00001::0",
            paper_id="2301.00001",
            chunk_index=0,
            chunk_text="preserved text",
            page_num=1,
        )
    )
    sqlite_session.commit()

    record = {
        "paper_id": "2301.00001",
        "title": "New title",
        "authors": [],
        "year": 2020,
        "primary_category": "cs.AI",
        "pdf_path": str(pdf),
    }

    pid, status = _ingest_one(sqlite_session, record, force=True)
    assert pid == "2301.00001"
    assert status == "failed"

    rows = sqlite_session.query(Chunk).filter_by(paper_id="2301.00001").all()
    assert len(rows) == 1
    assert rows[0].chunk_text == "preserved text"
    assert fake.add_calls == 1


def test_success_replaces_chunks_after_qdrant_write(
    monkeypatch: pytest.MonkeyPatch, sqlite_session, tmp_path: Path
) -> None:
    fake = _FakeVS(fail_on_add=False)
    monkeypatch.setattr(ingest_mod, "get_vector_store", lambda: fake)

    monkeypatch.setattr(
        ingest_mod, "extract_pages", lambda _path: [(1, "body")]
    )
    monkeypatch.setattr(
        ingest_mod,
        "chunk_pages",
        lambda _pages: [
            PaperChunk(chunk_index=0, text="fresh", page_num=1),
        ],
    )

    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    sqlite_session.add(
        Paper(
            paper_id="2301.00002",
            title="T",
            year=2021,
            primary_category="cs.LG",
            ingest_status="ok",
            num_chunks=1,
        )
    )
    sqlite_session.add(
        Chunk(
            chunk_id="2301.00002::0",
            paper_id="2301.00002",
            chunk_index=0,
            chunk_text="stale",
            page_num=1,
        )
    )
    sqlite_session.commit()

    record = {
        "paper_id": "2301.00002",
        "title": "T",
        "authors": [],
        "year": 2021,
        "primary_category": "cs.LG",
        "pdf_path": str(pdf),
    }

    pid, status = _ingest_one(sqlite_session, record, force=True)
    assert status == "ok"
    rows = sqlite_session.query(Chunk).filter_by(paper_id="2301.00002").all()
    assert len(rows) == 1
    assert rows[0].chunk_text == "fresh"
    assert fake.deleted == []
