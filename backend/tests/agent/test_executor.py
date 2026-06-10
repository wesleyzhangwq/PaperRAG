"""Test executor node."""
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from app.agent.state import AgentState, StepSpec
from app.agent.nodes.executor import executor_node


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


def test_executor_query_rewrite():
    plan = [StepSpec(action="query_rewrite", params={"original_query": "test"}, reason="decompose")]
    state = _base_state(plan=plan, plan_step_index=0)

    with patch("app.agent.nodes.executor.rewrite_query", return_value=["sub query 1", "sub query 2"]):
        result = executor_node(state, db=MagicMock())

    assert result["plan_step_index"] == 1
    assert len(result["step_traces"]) == 1


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
