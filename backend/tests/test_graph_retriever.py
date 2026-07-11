from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from app.db.neo4j import GraphCandidate


def test_graph_second_pass_multiplies_semantic_and_path_scores() -> None:
    from app.services.graph_retriever import retrieve_graph_context

    seed = Document(
        page_content="seed",
        metadata={
            "paper_id": "A",
            "retrieval_score": 0.9,
            "retrieval_source": "local",
        },
    )
    candidate = GraphCandidate(
        paper_id="B",
        graph_score=0.25,
        paths=({"seed_paper_id": "A", "hops": 2, "relations": ["CITES", "AUTHORED_BY"]},),
    )
    graph_doc = Document(page_content="graph evidence", metadata={"paper_id": "B"})
    settings = SimpleNamespace(
        graph_rag_enabled=True,
        graph_seed_papers=4,
        graph_max_hops=2,
        graph_candidate_limit=12,
    )
    with patch(
        "app.services.graph_retriever.get_settings", return_value=settings
    ), patch(
        "app.services.graph_retriever.retrieve_graph_candidates",
        return_value=[candidate],
    ), patch(
        "app.services.graph_retriever.retrieve",
        return_value=[(graph_doc, 0.8)],
    ):
        documents, report = retrieve_graph_context(
            query="compare A and B",
            existing_context=[seed],
            top_k=5,
        )

    assert report.fallback_reason is None
    assert documents[0].metadata["semantic_score"] == pytest.approx(0.8)
    assert documents[0].metadata["graph_score"] == pytest.approx(0.25)
    assert documents[0].metadata["retrieval_score"] == pytest.approx(0.2)
