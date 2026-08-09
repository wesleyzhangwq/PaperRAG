"""Transaction behavior for batch ingest."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# mysql.py builds the engine at import time; avoid requiring a real .env.
os.environ.setdefault("MYSQL_HOST", "127.0.0.1")

from app.services import ingest as ingest_mod


class TestRunIngestTransaction(unittest.TestCase):
    def test_run_ingest_rollback_on_failed_status(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump([{"paper_id": "p1", "pdf_path": "/x.pdf"}], f)
            meta_path = f.name
        try:
            sess = MagicMock()
            with patch.object(ingest_mod, "init_db"), patch.object(
                ingest_mod, "SessionLocal", return_value=sess
            ), patch.object(
                ingest_mod, "_ingest_one", return_value=("p1", "failed")
            ):
                ingest_mod.run_ingest(metadata_json=meta_path, force=False)

            sess.commit.assert_not_called()
            sess.rollback.assert_called_once()
            sess.close.assert_called_once()
        finally:
            Path(meta_path).unlink(missing_ok=True)

    def test_run_ingest_commits_on_ok(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump([{"paper_id": "p1", "pdf_path": "/x.pdf"}], f)
            meta_path = f.name
        try:
            sess = MagicMock()
            with patch.object(ingest_mod, "init_db"), patch.object(
                ingest_mod, "SessionLocal", return_value=sess
            ), patch.object(ingest_mod, "_ingest_one", return_value=("p1", "ok")):
                ingest_mod.run_ingest(metadata_json=meta_path, force=False)

            sess.commit.assert_called_once()
            sess.rollback.assert_not_called()
            sess.close.assert_called_once()
        finally:
            Path(meta_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
