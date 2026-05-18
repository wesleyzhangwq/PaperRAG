"""Test chat router endpoints."""
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.chat import ChatResponse, Source

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
