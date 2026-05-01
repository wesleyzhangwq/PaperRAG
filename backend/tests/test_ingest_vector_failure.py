"""Regression: embedding/Qdrant failure must not commit chunk rows without vectors."""
from __future__ import annotations

import os

# app.db.mysql builds SQLAlchemy URL from env at import time.
os.environ.setdefault("MYSQL_HOST", "127.0.0.1")

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.mysql import Base
from app.models.paper import Chunk, Paper
from app.services.ingest import _ingest_one
from app.utils.chunker import PaperChunk


class _FakeVectorStore:
    def __init__(self, add_raises: bool) -> None:
        self.add_raises = add_raises
        self.deleted: list[list[str]] = []
        self.added: list[tuple] = []

    def delete(self, ids: list[str]) -> None:
        self.deleted.append(list(ids))

    def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ) -> None:
        self.added.append((list(texts), metadatas, ids))
        if self.add_raises:
            raise RuntimeError("embedding unavailable")


class TestIngestVectorFailure(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def test_embed_failure_preserves_existing_mysql_chunks(self) -> None:
        """If add_texts fails, session must not retain DELETE/INSERT of chunks."""
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "p.pdf"
            pdf_path.write_bytes(b"%PDF-1.4 minimal")

            db = self.Session()
            try:
                db.add(
                    Paper(
                        paper_id="paper_xyz",
                        title="t",
                        year=2024,
                        primary_category="cs.AI",
                        pdf_path=str(pdf_path),
                        ingest_status="ok",
                        num_chunks=1,
                    )
                )
                db.add(
                    Chunk(
                        chunk_id="paper_xyz::0",
                        paper_id="paper_xyz",
                        chunk_index=0,
                        chunk_text="old",
                        page_num=1,
                        n_tokens=1,
                    )
                )
                db.commit()

                record = {
                    "paper_id": "paper_xyz",
                    "title": "t",
                    "authors": [],
                    "year": 2024,
                    "primary_category": "cs.AI",
                    "categories": [],
                    "doi": None,
                    "abstract": None,
                    "pdf_url": None,
                    "pdf_path": str(pdf_path),
                    "entry_id": None,
                    "published": None,
                    "updated": None,
                }

                fake_vs = _FakeVectorStore(add_raises=True)

                pages = [(1, "hello world " * 50)]
                chunks = [
                    PaperChunk(chunk_index=0, text="new chunk text", page_num=1),
                ]

                with (
                    patch("app.services.ingest.get_vector_store", return_value=fake_vs),
                    patch("app.services.ingest.extract_pages", return_value=pages),
                    patch("app.services.ingest.chunk_pages", return_value=chunks),
                ):
                    pid, status = _ingest_one(db, record, force=True)
                    self.assertEqual(pid, "paper_xyz")
                    self.assertEqual(status, "failed")

                db.commit()
            finally:
                db.close()

        verify = self.Session()
        try:
            rows = verify.query(Chunk).filter(Chunk.paper_id == "paper_xyz").all()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].chunk_text, "old")
            paper = verify.query(Paper).filter_by(paper_id="paper_xyz").one()
            self.assertEqual(paper.ingest_status, "failed")
            self.assertIn("embed", paper.ingest_error or "")
        finally:
            verify.close()

    def test_success_replaces_chunks_after_vectors_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "p2.pdf"
            pdf_path.write_bytes(b"%PDF-1.4 minimal")

            db = self.Session()
            try:
                db.add(
                    Paper(
                        paper_id="paper_abc",
                        title="t",
                        year=2024,
                        primary_category="cs.AI",
                        pdf_path=str(pdf_path),
                        ingest_status="ok",
                        num_chunks=1,
                    )
                )
                db.add(
                    Chunk(
                        chunk_id="paper_abc::0",
                        paper_id="paper_abc",
                        chunk_index=0,
                        chunk_text="old",
                        page_num=1,
                        n_tokens=1,
                    )
                )
                db.commit()

                record = {
                    "paper_id": "paper_abc",
                    "title": "t",
                    "authors": [],
                    "year": 2024,
                    "primary_category": "cs.AI",
                    "categories": [],
                    "doi": None,
                    "abstract": None,
                    "pdf_url": None,
                    "pdf_path": str(pdf_path),
                    "entry_id": None,
                    "published": None,
                    "updated": None,
                }

                fake_vs = _FakeVectorStore(add_raises=False)
                pages = [(1, "hello world " * 50)]
                chunks = [
                    PaperChunk(chunk_index=0, text="fresh", page_num=1),
                ]

                with (
                    patch("app.services.ingest.get_vector_store", return_value=fake_vs),
                    patch("app.services.ingest.extract_pages", return_value=pages),
                    patch("app.services.ingest.chunk_pages", return_value=chunks),
                ):
                    pid, status = _ingest_one(db, record, force=True)
                    self.assertEqual(status, "ok")

                db.commit()
            finally:
                db.close()

        verify = self.Session()
        try:
            rows = verify.query(Chunk).filter(Chunk.paper_id == "paper_abc").all()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].chunk_text, "fresh")
        finally:
            verify.close()


if __name__ == "__main__":
    unittest.main()
