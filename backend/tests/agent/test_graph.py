"""Test agent graph compilation and basic flow (v2 orchestration)."""
from contextlib import nullcontext
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from app.agent.graph import (
    build_agent_graph,
    route_after_guard,
    route_after_reflection,
    run_agent_eval_sync,
    run_agent_sync,
)

EXPECTED_NODES = {
    "guard", "intent", "planner", "route", "executor", "evidence",
    "sufficiency", "synthesis", "groundedness", "re_planner",
    "citation_gate", "presentation",
}


def test_graph_compiles_with_pipeline_nodes():
    mock_db = MagicMock()
    graph = build_agent_graph(mock_db)
    assert graph is not None
    nodes = set(graph.get_graph().nodes.keys())
    assert EXPECTED_NODES <= nodes


def test_run_agent_sync_returns_response():
    mock_db = MagicMock()
    mock_paper = MagicMock()
    mock_paper.title = "Test"
    mock_paper.authors = ["A"]
    mock_paper.year = 2023
    mock_paper.primary_category = "cs.CL"
    mock_paper.doi = None
    mock_db.query.return_value.filter.return_value.one_or_none.return_value = mock_paper

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        # intent
        MagicMock(content='{"type": "simple", "entities": ["attention"], "complexity": "low"}'),
        # planner (structural steps are filtered out by the sanitizer)
        MagicMock(content='[{"action": "retrieve_local", "params": {"query": "attention", "top_k": 8}, "reason": "search"}]'),
        # sufficiency (evaluate_docs tool)
        MagicMock(content='{"sufficient": true, "reason": "ok", "missing_aspects": []}'),
        # groundedness
        MagicMock(content='{"passed": true, "citation_ok": true, "completeness_ok": true, "logic_ok": true, "issues": [], "fix_strategy": null}'),
    ]
    # Synthesis uses llm.stream() which yields chunk objects
    mock_llm.stream.return_value = [MagicMock(content="Attention 是一种机制 [arxiv:1706.03762]")]

    mock_docs = [(Document(page_content="attention text", metadata={"paper_id": "1706.03762"}), 0.9)]

    with patch("app.agent.nodes.intent._get_llm", return_value=mock_llm), \
         patch("app.agent.nodes.planner._get_llm", return_value=mock_llm), \
         patch("app.tools.evaluate_docs._get_llm", return_value=mock_llm), \
         patch("app.agent.nodes.synthesis._get_llm", return_value=mock_llm), \
         patch("app.agent.nodes.groundedness._get_llm", return_value=mock_llm), \
         patch("app.agent.nodes.executor.retrieve", return_value=mock_docs):

        result = run_agent_sync(mock_db, "what is attention", session_id="test-session")

    assert result.answer is not None
    assert "1706.03762" in result.answer


def test_run_agent_eval_sync_exposes_raw_and_final_context_paper_ids():
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "final_answer": "answer",
        "sources": [],
        "retrieved_paper_ids": ["p1", "p2"],
        "retrieval_context": [
            Document(page_content="kept", metadata={"paper_id": "p2"}),
            Document(page_content="not used for generation", metadata={"paper_id": "p3"}),
        ],
        "synthesis_context_count": 1,
        "synthesis_context_paper_ids": ["p2"],
        "step_traces": [],
        "reflection_result": None,
    }

    with (
        patch("app.agent.graph.build_agent_graph", return_value=mock_graph),
        patch("app.agent.graph.open_sync_checkpointer", return_value=nullcontext(None)),
    ):
        response, retrieved_paper_ids, context_paper_ids = run_agent_eval_sync(
            MagicMock(),
            "test query",
            session_id="eval-session",
        )

    assert response.answer == "answer"
    assert response.used_chunks == 1
    assert retrieved_paper_ids == ["p1", "p2"]
    assert context_paper_ids == ["p2"]


def test_run_agent_sync_blocked_query_skips_llm_entirely():
    """Guard-blocked queries must short-circuit to presentation with a refusal."""
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.one_or_none.return_value = None

    mock_llm = MagicMock()  # would raise if invoked (no side_effect set ⇒ returns MagicMock)
    with patch("app.agent.nodes.intent._get_llm", return_value=mock_llm), \
         patch("app.agent.nodes.planner._get_llm", return_value=mock_llm):
        result = run_agent_sync(mock_db, "   ", session_id="blocked-session")

    assert "请输入" in result.answer
    mock_llm.invoke.assert_not_called()


# ------------------------------------------------------------------- routing

def test_route_after_guard_allows():
    assert route_after_guard({"guard_result": {"allowed": True}}) == "intent"


def test_route_after_guard_blocks_to_presentation():
    assert route_after_guard({"guard_result": {"allowed": False}}) == "presentation"


def test_route_after_reflection_passes_to_citation_gate():
    assert route_after_reflection({"reflection_result": {"passed": True}, "reflection_count": 0}, 2) == "citation_gate"


def test_route_after_reflection_re_generate_to_synthesis():
    state = {
        "reflection_result": {"passed": False, "fix_strategy": "re_generate"},
        "reflection_count": 1,
    }

    assert route_after_reflection(state, 2) == "synthesis"


def test_route_after_reflection_re_retrieve_to_re_planner():
    state = {
        "reflection_result": {"passed": False, "fix_strategy": "re_retrieve"},
        "reflection_count": 1,
    }

    assert route_after_reflection(state, 2) == "re_planner"


def test_route_after_reflection_stops_at_retry_budget():
    state = {
        "reflection_result": {"passed": False, "fix_strategy": "re_retrieve"},
        "reflection_count": 2,
    }

    assert route_after_reflection(state, 2) == "citation_gate"
