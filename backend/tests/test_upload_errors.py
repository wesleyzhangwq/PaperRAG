from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.db.mysql import get_db
from app.main import app


def _mock_db():
    db = MagicMock()
    db.query.return_value.filter.return_value.one_or_none.return_value = None
    yield db


app.dependency_overrides[get_db] = _mock_db
client = TestClient(app)


def _settings(tmp_path, *, max_mb=1):
    return SimpleNamespace(
        upload_dir=str(tmp_path),
        ingest_max_file_mb=max_mb,
    )


def test_heterogeneous_file_upload_is_queued(tmp_path):
    with (
        patch("app.routers.upload.get_settings", return_value=_settings(tmp_path)),
        patch("app.routers.upload._run_file_ingest_job") as run_job,
    ):
        resp = client.post(
            "/upload/files",
            files={"files": ("notes.md", b"# RAG\n\nGrounded evidence.", "text/markdown")},
        )

    assert resp.status_code == 200
    item = resp.json()["items"][0]
    assert item["paper_id"].startswith("local-")
    assert item["status"] == "queued"
    assert item["message"] == "saved"
    run_job.assert_called_once()


def test_heterogeneous_upload_rejects_unsupported_extension(tmp_path):
    with patch("app.routers.upload.get_settings", return_value=_settings(tmp_path)):
        resp = client.post(
            "/upload/files",
            files={"files": ("payload.exe", b"MZ", "application/octet-stream")},
        )

    assert resp.status_code == 415
    assert resp.json()["detail"]["code"] == "unsupported_file_type"


def test_heterogeneous_upload_enforces_size_limit(tmp_path):
    with patch(
        "app.routers.upload.get_settings",
        return_value=_settings(tmp_path, max_mb=0),
    ):
        resp = client.post(
            "/upload/files",
            files={"files": ("notes.txt", b"not empty", "text/plain")},
        )

    assert resp.status_code == 413
    assert resp.json()["detail"]["code"] == "file_too_large"
    assert list(tmp_path.iterdir()) == []
