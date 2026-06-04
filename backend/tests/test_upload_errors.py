from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.db.mysql import get_db
from app.main import app


def _mock_db():
    db = MagicMock()
    yield db


app.dependency_overrides[get_db] = _mock_db
client = TestClient(app)


def test_upload_rejects_non_pdf_with_user_facing_error():
    resp = client.post(
        "/upload",
        files={"file": ("notes.txt", b"not a pdf", "text/plain")},
    )

    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "invalid_file_type"
    assert resp.json()["detail"]["retryable"] is False


def test_upload_returns_queued_job_without_inline_ingest():
    with patch("app.routers.upload._run_upload_ingest_job") as run_job, \
         patch("app.routers.upload._ingest_one") as ingest_one:
        resp = client.post(
            "/upload",
            files={"file": ("paper.pdf", b"%PDF-1.4\n%%EOF", "application/pdf")},
            data={"title": "Uploaded Paper"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "queued"
    assert data["job_id"]
    assert data["paper_id"].startswith("user_paper_")
    ingest_one.assert_not_called()
    run_job.assert_called_once()
