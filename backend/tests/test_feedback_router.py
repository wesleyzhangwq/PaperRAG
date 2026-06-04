from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.db.mysql import get_db
from app.main import app


def _mock_db():
    db = MagicMock()
    yield db


app.dependency_overrides[get_db] = _mock_db
client = TestClient(app)


def test_submit_answer_feedback_persists_vote():
    resp = client.post(
        "/feedback",
        json={
            "conversation_id": "conv-1",
            "message_id": 42,
            "vote": "down",
            "reason": "citation_mismatch",
            "comment": "The cited paper does not support the claim.",
        },
    )

    assert resp.status_code == 200
    assert resp.json()["status"] == "recorded"


def test_list_feedback_returns_recorded_signals():
    row = SimpleNamespace(
        id=7,
        conversation_id="conv-1",
        message_id=42,
        vote="down",
        reason="citation_mismatch",
        comment="The cited paper does not support the claim.",
        created_at="2026-06-04T00:00:00",
    )
    db = next(_mock_db())
    db.query.return_value.with_entities.return_value.scalar.return_value = 1
    db.query.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = [row]

    def override_db():
        yield db

    app.dependency_overrides[get_db] = override_db
    resp = client.get("/feedback?limit=10")

    assert resp.status_code == 200
    assert resp.json()["total"] == 1
    assert resp.json()["items"][0]["vote"] == "down"
