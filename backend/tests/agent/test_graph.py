"""Test agent graph compilation and basic flow."""
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from app.agent.graph import build_agent_graph, run_agent_sync


def test_graph_compiles():
    mock_db = MagicMock()
    graph = build_agent_graph(mock_db)
    assert graph is not None


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
        # planner
        MagicMock(content='[{"action": "retrieve_local", "params": {"query": "attention", "top_k": 8}, "reason": "search"}, {"action": "evaluate_docs", "params": {}, "reason": "check"}, {"action": "reasoning_synthesis", "params": {}, "reason": "answer"}]'),
        # evaluate_docs
        MagicMock(content='{"sufficient": true, "reason": "ok", "missing_aspects": []}'),
        # reflection
        MagicMock(content='{"passed": true, "citation_ok": true, "completeness_ok": true, "logic_ok": true, "issues": [], "fix_strategy": null}'),
    ]
    # Synthesis uses llm.stream() which yields chunk objects
    mock_llm.stream.return_value = [MagicMock(content="Attention 是一种机制 [arxiv:1706.03762]")]

    mock_docs = [(Document(page_content="attention text", metadata={"paper_id": "1706.03762"}), 0.9)]

    with patch("app.agent.nodes.intent._get_llm", return_value=mock_llm), \
         patch("app.agent.nodes.planner._get_llm", return_value=mock_llm), \
         patch("app.tools.evaluate_docs._get_llm", return_value=mock_llm), \
         patch("app.agent.nodes.synthesis._get_llm", return_value=mock_llm), \
         patch("app.agent.nodes.reflection._get_llm", return_value=mock_llm), \
         patch("app.agent.nodes.executor.retrieve", return_value=mock_docs):

        result = run_agent_sync(mock_db, "what is attention", session_id="test-session")

    assert result.answer is not None
    assert "1706.03762" in result.answer
