"""Tests for the corpus overview endpoint."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.db.mysql import get_db
from app.main import app


@pytest.fixture(autouse=True)
def _clear_db_override():
    previous = app.dependency_overrides.get(get_db)
    yield
    if previous is None:
        app.dependency_overrides.pop(get_db, None)
    else:
        app.dependency_overrides[get_db] = previous


class FakePaperQuery:
    def __init__(self, papers: list[SimpleNamespace]):
        self._papers = papers

    def filter(self, *_args, **_kwargs) -> "FakePaperQuery":
        return self

    def all(self) -> list[SimpleNamespace]:
        return list(self._papers)


class FakeDB:
    def __init__(self, papers: list[SimpleNamespace]):
        self._papers = papers

    def query(self, *_args, **_kwargs) -> FakePaperQuery:
        return FakePaperQuery(self._papers)


def _paper(
    paper_id: str,
    title: str,
    *,
    year: int,
    categories: list[str],
    primary_category: str = "cs.CL",
    num_chunks: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        paper_id=paper_id,
        title=title,
        authors=[],
        year=year,
        primary_category=primary_category,
        categories=categories,
        doi=None,
        abstract=None,
        ingest_status="ok",
        num_chunks=num_chunks,
    )


def _client_with_papers(papers: list[SimpleNamespace]) -> TestClient:
    def _mock_db():
        yield FakeDB(papers)

    app.dependency_overrides[get_db] = _mock_db
    return TestClient(app)


def test_papers_overview_returns_corpus_shape() -> None:
    client = _client_with_papers([
        _paper(
            "2005.11401",
            "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            year=2020,
            categories=["cs.CL", "rag_ir_memory"],
            num_chunks=8,
        ),
        _paper(
            "1706.03762",
            "Attention Is All You Need",
            year=2017,
            categories=["cs.CL", "llm_transformer"],
            num_chunks=7,
        ),
        _paper(
            "2303.11366",
            "Reflexion: Language Agents with Verbal Reinforcement Learning",
            year=2023,
            categories=["cs.AI", "agents_reasoning"],
            num_chunks=3,
        ),
    ])

    resp = client.get("/papers/overview")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_papers"] == 3
    assert data["total_chunks"] == 18
    assert data["year_min"] == 2017
    assert data["year_max"] == 2023
    assert data["topic_buckets"][0]["key"] == "rag_ir_memory"
    assert data["topic_buckets"][0]["label"] == "RAG / Retrieval"
    assert data["topic_buckets"][0]["paper_count"] == 1
    assert data["topic_buckets"][0]["chunk_count"] == 8
    assert data["topic_buckets"][0]["representative_papers"][0]["paper_id"] == "2005.11401"
    assert "RAG / Retrieval 的技术路线是如何演进的？" in data["suggested_questions"]
    assert data["generated_at"]


def test_papers_overview_handles_empty_corpus() -> None:
    client = _client_with_papers([])

    resp = client.get("/papers/overview")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_papers"] == 0
    assert data["total_chunks"] == 0
    assert data["year_min"] is None
    assert data["year_max"] is None
    assert data["topic_buckets"] == []
    assert data["suggested_questions"] == ["如何上传第一篇论文？"]


def test_paper_summary_includes_topic_bucket_label() -> None:
    from app.routers.papers import _to_summary

    summary = _to_summary(_paper(
        "2511.16043",
        "Agent0: Unleashing Self-Evolving Agents from Zero Data",
        year=2025,
        categories=["cs.LG", "agents_reasoning"],
        primary_category="cs.LG",
        num_chunks=95,
    ))

    assert summary.topic_bucket_key == "agents_reasoning"
    assert summary.topic_bucket_label == "Agents / Reasoning"
