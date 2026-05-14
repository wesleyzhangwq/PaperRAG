"""Ensure failed Qdrant deletes do not leave MySQL chunks wiped while vectors remain."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("MYSQL_URL", "sqlite+pysqlite:///:memory:")
os.environ.setdefault("EMBEDDING_API_KEY", "test-dummy-key")

from app.db.mysql import SessionLocal, init_db  # noqa: E402
from app.models.paper import Chunk  # noqa: E402
from app.services.ingest import _ingest_one  # noqa: E402
from app.utils.chunker import PaperChunk  # noqa: E402


class TestIngestVectorDeleteFailure(unittest.TestCase):
    def setUp(self) -> None:
        init_db()
        self._pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        self._pdf.write(b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF")
        self._pdf.flush()
        self.pdf_path = str(Path(self._pdf.name).resolve())

    def tearDown(self) -> None:
        self._pdf.close()
        Path(self.pdf_path).unlink(missing_ok=True)

    def _record(self) -> dict:
        return {
            "paper_id": "2501.00001",
            "title": "Test Paper",
            "authors": [],
            "year": 2025,
            "primary_category": "cs.TEST",
            "categories": [],
            "doi": None,
            "abstract": None,
            "pdf_path": self.pdf_path,
            "pdf_url": None,
            "entry_id": None,
            "published": None,
            "updated": None,
        }

    def test_vector_delete_failure_preserves_mysql_chunks(self) -> None:
        mock_vs = MagicMock()
        chunks = [
            PaperChunk(chunk_index=0, text="chunk zero " * 30, page_num=1),
            PaperChunk(chunk_index=1, text="chunk one " * 30, page_num=1),
        ]

        with patch("app.services.ingest.get_vector_store", return_value=mock_vs):
            with patch("app.services.ingest.extract_pages", return_value=[(1, "body " * 200)]):
                with patch("app.services.ingest.chunk_pages", return_value=chunks):
                    db = SessionLocal()
                    try:
                        pid, st = _ingest_one(db, self._record(), force=True)
                        self.assertEqual(st, "ok")
                        db.flush()
                        self.assertEqual(
                            db.query(Chunk).filter(Chunk.paper_id == pid).count(),
                            2,
                        )

                        mock_vs.delete.side_effect = RuntimeError("qdrant unavailable")
                        pid2, st2 = _ingest_one(db, self._record(), force=True)
                        self.assertEqual(pid2, pid)
                        self.assertEqual(st2, "failed")
                        db.flush()
                        self.assertEqual(
                            db.query(Chunk).filter(Chunk.paper_id == pid).count(),
                            2,
                        )
                    finally:
                        db.rollback()
                        db.close()


if __name__ == "__main__":
    unittest.main()
