"""Test chat router endpoints."""
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.db.mysql import get_db
from app.schemas.chat import ChatResponse, Source


def _mock_db():
    db = MagicMock()
    db.query.return_value.filter.return_value.one_or_none.return_value = None
    db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []
    yield db


app.dependency_overrides[get_db] = _mock_db
client = TestClient(app)


def test_chat_sync_returns_200():
    mock_response = ChatResponse(
        answer="Test answer [arxiv:1706.03762]",
        sources=[Source(paper_id="1706.03762", title="Test", authors=[], year=2017, arxiv_url="https://arxiv.org/abs/1706.03762")],
        used_chunks=3,
    )
    with patch("app.routers.chat.run_agent_sync", return_value=mock_response):
        resp = client.post("/chat", json={"query": "what is attention", "session_id": "test"})

    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert data["answer"] == "Test answer [arxiv:1706.03762]"
