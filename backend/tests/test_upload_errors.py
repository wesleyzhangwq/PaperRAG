from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from app.db.mysql import get_db
from app.main import app


def _mock_db():
    db = MagicMock()
    yield db


app.dependency_overrides[get_db] = _mock_db
client = TestClient(app)


def test_pdf_upload_channel_is_disabled():
    resp = client.post(
        "/upload",
        files={"file": ("paper.pdf", b"%PDF-1.4\n%%EOF", "application/pdf")},
    )

    assert resp.status_code == 410
    assert resp.json()["code"] == "pdf_upload_disabled"
    assert resp.json()["retryable"] is False
