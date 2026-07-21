"""Test executor node."""
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from app.agent.state import AgentState, StepSpec
from app.agent.nodes.executor import _parse_arxiv_to_documents, executor_node


def _base_state(**overrides) -> AgentState:
    defaults = {
        "messages": [],
        "intent": {"type": "simple", "entities": [], "complexity": "low"},
        "plan": [],
        "plan_step_index": 0,
        "retrieval_context": [],
        "step_traces": [],
        "reflection_count": 0,
        "final_answer": None,
    }
    defaults.update(overrides)
    return defaults


def test_executor_retrieve_local():
    plan = [StepSpec(action="retrieve_local", params={"query": "attention", "top_k": 8}, reason="test")]
    state = _base_state(plan=plan, plan_step_index=0)

    mock_docs = [
        (Document(page_content="attention content", metadata={"paper_id": "1706.03762"}), 0.9),
    ]
    with patch("app.agent.nodes.executor.retrieve", return_value=mock_docs):
        result = executor_node(state, db=MagicMock())

    assert result["plan_step_index"] == 1
    assert len(result["retrieval_context"]) == 1
    assert len(result["step_traces"]) == 1


def test_executor_tracks_unique_paper_ids_before_evidence_compression():
    plan = [StepSpec(action="retrieve_local", params={"query": "attention", "top_k": 8}, reason="test")]
    state = _base_state(plan=plan, plan_step_index=0)
    mock_docs = [
        (Document(page_content="first", metadata={"paper_id": "p1"}), 0.9),
        (Document(page_content="second", metadata={"paper_id": "p1"}), 0.8),
        (Document(page_content="third", metadata={"paper_id": "p2"}), 0.7),
    ]

    with patch("app.agent.nodes.executor.retrieve", return_value=mock_docs):
        result = executor_node(state, db=MagicMock())

    assert result["retrieved_paper_ids"] == ["p1", "p2"]


def test_executor_query_rewrite():
    plan = [StepSpec(action="query_rewrite", params={"original_query": "test"}, reason="decompose")]
    state = _base_state(plan=plan, plan_step_index=0)

    with patch("app.agent.nodes.executor.rewrite_query", return_value=["sub query 1", "sub query 2"]):
        result = executor_node(state, db=MagicMock())

    assert result["plan_step_index"] == 1
    assert len(result["step_traces"]) == 1


def test_executor_query_rewrite_replaces_defaulted_later_query():
    plan = [
        StepSpec(action="query_rewrite", params={"original_query": "original"}, reason="decompose"),
        StepSpec(
            action="retrieve_local",
            params={"query": "original", "top_k": 8, "_query_defaulted": True},
            reason="search",
        ),
    ]
    state = _base_state(
        messages=[HumanMessage(content="original")],
        plan=plan,
        plan_step_index=0,
    )

    with patch("app.agent.nodes.executor.rewrite_query", return_value=["specific sub-query"]):
        executor_node(state, db=MagicMock())

    assert state["plan"][1]["params"]["query"] == "specific sub-query"
    assert "_query_defaulted" not in state["plan"][1]["params"]


def test_executor_advances_index():
    plan = [
        StepSpec(action="retrieve_local", params={"query": "q1", "top_k": 8}, reason="first"),
        StepSpec(action="retrieve_local", params={"query": "q2", "top_k": 8}, reason="second"),
    ]
    state = _base_state(plan=plan, plan_step_index=0)

    with patch("app.agent.nodes.executor.retrieve", return_value=[]):
        result = executor_node(state, db=MagicMock())

    assert result["plan_step_index"] == 1


def test_executor_search_web_unavailable_does_not_add_context():
    plan = [
        StepSpec(
            action="search_web",
            params={"query": "latest transformer history", "max_results": 3},
            reason="web supplement",
        )
    ]
    state = _base_state(
        messages=[HumanMessage(content="explain transformer history")],
        plan=plan,
        plan_step_index=0,
        retrieval_context=[],
    )

    mock_tool = MagicMock()
    mock_tool.invoke.return_value = "Web search unavailable: SSLError: unexpected eof"
    with patch("app.agent.nodes.executor.search_web_tool", mock_tool):
        result = executor_node(state, db=MagicMock())

    assert result["retrieval_context"] == []
    assert result["step_traces"][0]["output_summary"] == "web unavailable"
    assert result["step_traces"][0]["detail"]["total"] == 0
    assert result["step_traces"][0]["detail"]["error"]


def test_executor_retrieve_arxiv_failure_degrades_to_warning_trace():
    plan = [
        StepSpec(
            action="retrieve_arxiv",
            params={"query": "transformers", "max_results": 3},
            reason="live supplement",
        )
    ]
    state = _base_state(
        messages=[HumanMessage(content="latest transformer work")],
        plan=plan,
        plan_step_index=0,
        retrieval_context=[],
    )

    mock_tool = MagicMock()
    mock_tool.invoke.side_effect = RuntimeError("arXiv timeout")
    with patch("app.agent.nodes.executor.retrieve_arxiv_tool", mock_tool):
        result = executor_node(state, db=MagicMock())

    assert result["plan_step_index"] == 1
    assert result["retrieval_context"] == []
    assert result["step_traces"][0]["output_summary"] == "arXiv unavailable"
    assert result["step_traces"][0]["detail"]["total"] == 0
    assert result["step_traces"][0]["detail"]["error"] == "RuntimeError"
    assert "arxiv_service_unavailable" in result["fallback_telemetry"]["failure_classes"]


def test_parse_arxiv_tool_native_output_preserves_id_and_title():
    raw = (
        "[arxiv:2401.01234v2 | cs.CL | 2024]\n"
        "title: Native Tool Format\n"
        "authors: A, B\n"
        "abstract: Evidence"
    )

    docs = _parse_arxiv_to_documents(raw)

    assert len(docs) == 1
    assert docs[0].metadata["paper_id"] == "2401.01234"
    assert docs[0].metadata["title"] == "Native Tool Format"


def test_executor_arxiv_no_results_is_not_a_fallback_failure():
    state = _base_state(
        plan=[StepSpec(action="retrieve_arxiv", params={"query": "none"}, reason="test")],
        plan_step_index=0,
    )
    mock_tool = MagicMock()
    mock_tool.invoke.return_value = "No papers found on arXiv for this query."

    with patch("app.agent.nodes.executor.retrieve_arxiv_tool", mock_tool):
        result = executor_node(state, db=MagicMock())

    assert result["retrieval_context"] == []
    assert result["step_traces"][0]["detail"]["status"] == "no_results"
    assert "fallback_telemetry" not in result


def test_executor_intentional_external_disable_is_not_a_fallback_failure():
    state = _base_state(
        plan=[StepSpec(action="search_web", params={"query": "fixture"}, reason="test")],
        plan_step_index=0,
    )
    mock_tool = MagicMock()
    mock_tool.invoke.return_value = "Evaluation external web intentionally disabled."

    with patch("app.agent.nodes.executor.search_web_tool", mock_tool):
        result = executor_node(state, db=MagicMock())

    assert result["step_traces"][0]["detail"]["status"] == "intentional_disabled"
    assert "fallback_telemetry" not in result
