"""Test chat router endpoints."""
import json
from contextlib import asynccontextmanager
from datetime import datetime
from types import SimpleNamespace
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


def test_chat_stream_uses_astream_events_and_thread_id():
    seen_config = {}

    class FakeGraph:
        async def astream_events(self, initial_state, config, version="v2"):
            seen_config.update(config)
            assert version == "v2"
            yield {"event": "on_custom_event", "name": "token", "data": {"t": "hello"}}
            yield {
                "event": "on_chain_stream",
                "name": "LangGraph",
                "data": {"chunk": {"intent": {"intent": {"type": "simple", "entities": [], "complexity": "low"}}}},
            }
            yield {
                "event": "on_chain_stream",
                "name": "LangGraph",
                "data": {"chunk": {"synthesis": {"final_answer": "hello"}}},
            }

    @asynccontextmanager
    async def fake_checkpointer():
        yield None

    with patch("app.routers.chat.build_agent_graph", return_value=FakeGraph()), \
         patch("app.routers.chat.open_async_checkpointer", fake_checkpointer), \
         patch("app.routers.chat.SessionLocal", return_value=MagicMock()), \
         patch("app.routers.chat._ensure_conversation"), \
         patch("app.routers.chat._load_history", return_value=[]), \
         patch("app.routers.chat._persist_messages"):
        resp = client.post(
            "/chat/stream",
            json={"query": "hello", "conversation_id": "conv-stream", "session_id": "conv-stream"},
        )

    assert resp.status_code == 200
    assert "event: conversation" in resp.text
    assert "event: token" in resp.text
    assert "event: intent" in resp.text
    assert "event: done" in resp.text
    assert seen_config["configurable"]["thread_id"] == "conv-stream"


def test_conversation_messages_expose_persisted_elapsed_ms():
    row = SimpleNamespace(
        id=5,
        role="assistant",
        content="answer",
        sources_json=None,
        thinking_json=json.dumps({"traces": [], "presentation": None, "elapsed_ms": 282000}),
        created_at=datetime(2026, 6, 5, 12, 0, 0),
    )
    db = MagicMock()
    db.query.return_value.filter.return_value.order_by.return_value.all.return_value = [row]

    def override_db():
        yield db

    app.dependency_overrides[get_db] = override_db
    resp = client.get("/conversations/conv-elapsed/messages")

    assert resp.status_code == 200
    assert resp.json()[0]["elapsed_ms"] == 282000
