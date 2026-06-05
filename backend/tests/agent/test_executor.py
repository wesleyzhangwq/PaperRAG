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
        StepSpec(action="evaluate_docs", params={}, reason="second"),
    ]
    state = _base_state(plan=plan, plan_step_index=0)

    with patch("app.agent.nodes.executor.retrieve", return_value=[]):
        result = executor_node(state, db=MagicMock())

    assert result["plan_step_index"] == 1


def test_executor_inserts_web_search_when_docs_insufficient_and_tavily_configured():
    plan = [
        StepSpec(action="evaluate_docs", params={}, reason="check"),
        StepSpec(action="reasoning_synthesis", params={}, reason="answer"),
    ]
    state = _base_state(
        messages=[HumanMessage(content="explain recent RAG products")],
        plan=plan,
        plan_step_index=0,
        retrieval_context=[Document(page_content="partial", metadata={"paper_id": "2604.00001"})],
    )
    settings = MagicMock(tavily_api_key="test-key")

    with patch("app.agent.nodes.executor.evaluate_docs", return_value={
        "sufficient": False,
        "missing_aspects": ["recent industry background", "product usage"],
        "reason": "local papers are incomplete",
        "parse_failed": False,
    }), patch("app.agent.nodes.executor.get_settings", return_value=settings):
        result = executor_node(state, db=MagicMock())

    actions = [step["action"] for step in result["plan"]]
    assert actions[1] == "search_web"
    assert result["plan"][1]["params"]["query"] == "recent industry background"
    assert "retrieve_local" in actions


def test_executor_does_not_insert_web_search_when_tavily_missing():
    plan = [
        StepSpec(action="evaluate_docs", params={}, reason="check"),
        StepSpec(action="reasoning_synthesis", params={}, reason="answer"),
    ]
    state = _base_state(
        messages=[HumanMessage(content="explain recent RAG products")],
        plan=plan,
        plan_step_index=0,
        retrieval_context=[Document(page_content="partial", metadata={"paper_id": "2604.00001"})],
    )
    settings = MagicMock(tavily_api_key=None)

    with patch("app.agent.nodes.executor.evaluate_docs", return_value={
        "sufficient": False,
        "missing_aspects": ["recent industry background"],
        "reason": "local papers are incomplete",
        "parse_failed": False,
    }), patch("app.agent.nodes.executor.get_settings", return_value=settings):
        result = executor_node(state, db=MagicMock())

    actions = [step["action"] for step in result["plan"]]
    assert "search_web" not in actions


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


def test_executor_evaluate_docs_sufficient_does_not_add_supplementary_steps():
    plan = [
        StepSpec(action="evaluate_docs", params={}, reason="check"),
        StepSpec(action="reasoning_synthesis", params={}, reason="answer"),
    ]
    state = _base_state(
        messages=[HumanMessage(content="explain attention")],
        plan=plan,
        plan_step_index=0,
        retrieval_context=[Document(page_content="enough", metadata={"paper_id": "1706.03762"})],
    )

    with patch("app.agent.nodes.executor.evaluate_docs", return_value={
        "sufficient": True,
        "missing_aspects": [],
        "reason": "enough",
        "parse_failed": False,
    }):
        result = executor_node(state, db=MagicMock())

    assert "plan" not in result
