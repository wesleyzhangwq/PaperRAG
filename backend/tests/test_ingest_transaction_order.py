"""Ensure ingest does not commit MySQL chunk removal before Qdrant vectors exist."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Importing app.db.mysql builds the engine from env; avoid failing in CI without .env.
os.environ.setdefault("MYSQL_HOST", "127.0.0.1")

import app.services.ingest as ingest_service  # noqa: E402


class TestIngestChunkOrder(unittest.TestCase):
    def _make_db_mocks(self, paper: MagicMock) -> tuple[MagicMock, MagicMock]:
        """Returns (db, chunk_filter_target) where chunk_filter_target has .all() and .delete()."""
        from app.models.paper import Chunk, Paper as PaperModel

        paper_q = MagicMock()
        paper_q.filter.return_value.one_or_none.return_value = paper

        chunk_filtered = MagicMock()
        chunk_filtered.all.return_value = []
        chunk_filtered.delete = MagicMock()

        chunk_q = MagicMock()
        chunk_q.filter.return_value = chunk_filtered

        def query_side(model):
            if model is PaperModel:
                return paper_q
            if model is Chunk:
                return chunk_q
            raise AssertionError(f"unexpected model {model}")

        db = MagicMock()
        db.query.side_effect = query_side
        return db, chunk_filtered

    @patch.object(ingest_service, "get_vector_store")
    @patch.object(ingest_service, "chunk_pages")
    @patch.object(ingest_service, "extract_pages")
    def test_embed_failure_does_not_touch_mysql_chunk_rows(
        self, mock_extract, mock_chunk, mock_vs
    ):
        from app.services.ingest import _ingest_one
        from app.utils.chunker import PaperChunk

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4\n")
            pdf_path = f.name
        try:
            mock_extract.return_value = [(1, "x" * 200)]
            mock_chunk.return_value = [
                PaperChunk(chunk_index=0, text="y" * 100, page_num=1),
            ]
            mock_vs.return_value.add_texts.side_effect = RuntimeError("ollama down")

            paper = MagicMock()
            paper.paper_id = "test.0001"
            paper.ingest_status = "pending"
            paper.num_chunks = 0
            paper.title = "Title"
            paper.year = 2024
            paper.primary_category = "cs.AI"
            paper.doi = None

            db, chunk_filtered = self._make_db_mocks(paper)
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
                "pdf_path": pdf_path,
                "entry_id": None,
                "published": None,
                "updated": None,
            }

            pid, status = _ingest_one(db, record, force=True)
            self.assertEqual(status, "failed")
            self.assertEqual(pid, paper.paper_id)
            chunk_filtered.delete.assert_not_called()
            db.add_all.assert_not_called()
        finally:
            Path(pdf_path).unlink(missing_ok=True)

    @patch.object(ingest_service, "get_vector_store")
    @patch.object(ingest_service, "chunk_pages")
    @patch.object(ingest_service, "extract_pages")
    def test_embed_success_updates_mysql_chunks(
        self, mock_extract, mock_chunk, mock_vs
    ):
        from app.services.ingest import _ingest_one
        from app.utils.chunker import PaperChunk

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4\n")
            pdf_path = f.name
        try:
            mock_extract.return_value = [(1, "x" * 200)]
            mock_chunk.return_value = [
                PaperChunk(chunk_index=0, text="y" * 100, page_num=1),
            ]

            paper = MagicMock()
            paper.paper_id = "test.0002"
            paper.ingest_status = "pending"
            paper.num_chunks = 0
            paper.title = "Title"
            paper.year = 2024
            paper.primary_category = "cs.AI"
            paper.doi = None

            db, chunk_filtered = self._make_db_mocks(paper)
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
                "pdf_path": pdf_path,
                "entry_id": None,
                "published": None,
                "updated": None,
            }

            pid, status = _ingest_one(db, record, force=True)
            self.assertEqual(status, "ok")
            chunk_filtered.delete.assert_called_once()
            db.add_all.assert_called_once()
        finally:
            Path(pdf_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
