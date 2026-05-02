"""Ensure ingest does not drop Qdrant / MySQL chunk rows before vectors are live."""
from __future__ import annotations

import os

# Importing `app` loads mysql engine; tests do not need a real MySQL server.
os.environ.setdefault("MYSQL_URL", "sqlite+pysqlite:///:memory:")

import unittest
from unittest.mock import MagicMock, patch

from app.services.ingest import _ingest_one


class _FakeChunk:
    def __init__(self, chunk_id: str) -> None:
        self.chunk_id = chunk_id


class _FakePaperChunk:
    def __init__(self, chunk_index: int, text: str, page_num: int | None) -> None:
        self.chunk_index = chunk_index
        self.text = text
        self.page_num = page_num


def _make_db_with_old_chunks(paper_id: str, old_chunk_ids: list[str]) -> tuple[MagicMock, MagicMock]:
    db = MagicMock()
    chunk_q = MagicMock()
    chunk_q.filter.return_value.all.return_value = [_FakeChunk(cid) for cid in old_chunk_ids]
    chunk_q.filter.return_value.delete.return_value = None

    def query_side_effect(model):
        _ = model
        return chunk_q

    db.query.side_effect = query_side_effect
    return db, chunk_q


class TestIngestVectorOrder(unittest.TestCase):
    def test_embed_failure_does_not_call_vector_delete_or_mysql_chunk_delete(self) -> None:
        paper = MagicMock()
        paper.paper_id = "2401.00001"
        paper.title = "Title"
        paper.year = 2024
        paper.primary_category = "cs.AI"
        paper.doi = None
        paper.ingest_status = "ok"
        paper.num_chunks = 1

        vs = MagicMock()
        vs.add_texts.side_effect = RuntimeError("embedding API down")

        db, chunk_q = _make_db_with_old_chunks(paper.paper_id, [f"{paper.paper_id}::0"])
        record = {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "authors": [],
            "year": paper.year,
            "primary_category": paper.primary_category,
            "categories": [],
            "doi": None,
            "abstract": None,
            "pdf_url": None,
            "pdf_path": "/tmp/fake.pdf",
            "entry_id": None,
            "published": None,
            "updated": None,
        }
        pages = [(1, "x" * 400)]
        chunks = [_FakePaperChunk(0, "y" * 200, 1)]

        with patch("app.services.ingest._upsert_paper", return_value=paper):
            with patch("app.services.ingest.Path.exists", return_value=True):
                with patch("app.services.ingest.extract_pages", return_value=pages):
                    with patch("app.services.ingest.chunk_pages", return_value=chunks):
                        with patch(
                            "app.services.ingest.get_vector_store", return_value=vs
                        ):
                            pid, status = _ingest_one(db, record, force=True)

        self.assertEqual(pid, paper.paper_id)
        self.assertEqual(status, "failed")
        vs.add_texts.assert_called_once()
        vs.delete.assert_not_called()
        chunk_q.filter.return_value.delete.assert_not_called()

    def test_success_prunes_stale_vectors_then_mysql(self) -> None:
        paper = MagicMock()
        paper.paper_id = "2401.00002"
        paper.title = "T"
        paper.year = 2024
        paper.primary_category = "cs.AI"
        paper.doi = None
        paper.ingest_status = "ok"
        paper.num_chunks = 2

        vs = MagicMock()
        db, chunk_q = _make_db_with_old_chunks(
            paper.paper_id,
            [
                f"{paper.paper_id}::0",
                f"{paper.paper_id}::99",
            ],
        )

        record = {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "authors": [],
            "year": paper.year,
            "primary_category": paper.primary_category,
            "categories": [],
            "doi": None,
            "abstract": None,
            "pdf_url": None,
            "pdf_path": "/tmp/fake.pdf",
            "entry_id": None,
            "published": None,
            "updated": None,
        }
        pages = [(1, "x" * 400)]
        chunks = [_FakePaperChunk(0, "y" * 200, 1)]

        with patch("app.services.ingest._upsert_paper", return_value=paper):
            with patch("app.services.ingest.Path.exists", return_value=True):
                with patch("app.services.ingest.extract_pages", return_value=pages):
                    with patch("app.services.ingest.chunk_pages", return_value=chunks):
                        with patch(
                            "app.services.ingest.get_vector_store", return_value=vs
                        ):
                            pid, status = _ingest_one(db, record, force=True)

        self.assertEqual(status, "ok")
        vs.add_texts.assert_called_once()
        vs.delete.assert_called_once()
        args, kwargs = vs.delete.call_args
        self.assertEqual(kwargs.get("ids"), [f"{paper.paper_id}::99"])
        chunk_q.filter.return_value.delete.assert_called_once()


if __name__ == "__main__":
    unittest.main()
