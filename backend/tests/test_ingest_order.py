"""Ensure ingest persists vectors before mutating MySQL state."""
from __future__ import annotations

import os

# mysql.py creates the engine at import time; use SQLite for tests without MySQL.
os.environ.setdefault("MYSQL_URL", "sqlite+pysqlite:///:memory:")

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.models.paper import Chunk, Paper
from app.utils.chunker import PaperChunk


class TestIngestVectorMysqlOrder(unittest.TestCase):
    def setUp(self) -> None:
        self._pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        self._pdf.write(b"%PDF-1.4\n")
        self._pdf.close()
        self.pdf_path = Path(self._pdf.name)

    def tearDown(self) -> None:
        self.pdf_path.unlink(missing_ok=True)

    @patch("app.services.ingest.get_vector_store")
    @patch("app.services.ingest.chunk_pages")
    @patch("app.services.ingest.extract_pages")
    def test_add_texts_failure_skips_mysql_delete(
        self, mock_extract: MagicMock, mock_chunk: MagicMock, mock_get_vs: MagicMock
    ) -> None:
        mock_extract.return_value = [(1, "x" * 200)]
        mock_chunk.return_value = [PaperChunk(0, "y" * 200, 1)]

        vs = MagicMock()
        vs.add_texts.side_effect = RuntimeError("embedding unavailable")
        mock_get_vs.return_value = vs

        db = MagicMock()
        paper_q = MagicMock()
        paper_q.filter.return_value.one_or_none.return_value = None
        chunk_q = MagicMock()
        chunk_filter = MagicMock()
        chunk_q.filter.return_value = chunk_filter
        chunk_filter.all.return_value = [MagicMock(chunk_id="p::0")]

        def query_side_effect(model: object) -> MagicMock:
            if model is Chunk:
                return chunk_q
            if model is Paper:
                return paper_q
            raise AssertionError(model)

        db.query.side_effect = query_side_effect

        from app.services.ingest import _ingest_one

        record = {
            "paper_id": "p",
            "title": "t",
            "authors": [],
            "year": 0,
            "primary_category": "cs.AI",
            "categories": [],
            "doi": None,
            "abstract": None,
            "pdf_url": None,
            "pdf_path": str(self.pdf_path.resolve()),
            "entry_id": None,
            "published": None,
            "updated": None,
        }

        pid, status = _ingest_one(db, record, force=True)
        self.assertEqual(pid, "p")
        self.assertEqual(status, "failed")
        vs.add_texts.assert_called_once()
        vs.delete.assert_not_called()
        chunk_filter.delete.assert_not_called()

    @patch("app.services.ingest.get_vector_store")
    @patch("app.services.ingest.chunk_pages")
    @patch("app.services.ingest.extract_pages")
    def test_success_deletes_only_stale_vectors(
        self, mock_extract: MagicMock, mock_chunk: MagicMock, mock_get_vs: MagicMock
    ) -> None:
        mock_extract.return_value = [(1, "x" * 200)]
        mock_chunk.return_value = [PaperChunk(0, "y" * 200, 1)]

        vs = MagicMock()
        mock_get_vs.return_value = vs

        db = MagicMock()
        paper_q = MagicMock()
        paper_q.filter.return_value.one_or_none.return_value = None
        chunk_q = MagicMock()
        chunk_filter = MagicMock()
        chunk_q.filter.return_value = chunk_filter
        chunk_filter.all.return_value = [
            MagicMock(chunk_id="p::0"),
            MagicMock(chunk_id="p::1"),
        ]

        def query_side_effect(model: object) -> MagicMock:
            if model is Chunk:
                return chunk_q
            if model is Paper:
                return paper_q
            raise AssertionError(model)

        db.query.side_effect = query_side_effect

        from app.services.ingest import _ingest_one

        record = {
            "paper_id": "p",
            "title": "t",
            "authors": [],
            "year": 0,
            "primary_category": "cs.AI",
            "categories": [],
            "doi": None,
            "abstract": None,
            "pdf_url": None,
            "pdf_path": str(self.pdf_path.resolve()),
            "entry_id": None,
            "published": None,
            "updated": None,
        }

        pid, status = _ingest_one(db, record, force=True)
        self.assertEqual(status, "ok")
        vs.add_texts.assert_called_once()
        vs.delete.assert_called_once_with(ids=["p::1"])
        chunk_filter.delete.assert_called_once()


if __name__ == "__main__":
    unittest.main()
