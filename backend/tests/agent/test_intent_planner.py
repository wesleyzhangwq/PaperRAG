"""Test intent and planner nodes."""
import json
from unittest.mock import patch, MagicMock

from app.agent.state import AgentState, StepSpec
from app.agent.nodes.intent import intent_node
from app.agent.nodes.planner import planner_node, re_planner_node


def _base_state(**overrides) -> AgentState:
    defaults = {
        "messages": [],
        "intent": None,
        "plan": [],
        "plan_step_index": 0,
        "retrieval_context": [],
        "step_traces": [],
        "reflection_count": 0,
        "final_answer": None,
    }
    defaults.update(overrides)
    return defaults


def test_intent_node_simple_question():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='{"type": "simple", "entities": ["attention"], "complexity": "low"}'
    )
    state = _base_state()
    with patch("app.agent.nodes.intent._get_llm", return_value=mock_llm):
        result = intent_node(state, query="what is attention mechanism")

    assert result["intent"]["type"] == "simple"
    assert result["intent"]["complexity"] == "low"
    assert len(result["step_traces"]) == 1


def test_intent_node_comparison():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='{"type": "comparison", "entities": ["BERT", "GPT"], "complexity": "high"}'
    )
    state = _base_state()
    with patch("app.agent.nodes.intent._get_llm", return_value=mock_llm):
        result = intent_node(state, query="compare BERT and GPT")

    assert result["intent"]["type"] == "comparison"
    assert "BERT" in result["intent"]["entities"]


def test_planner_node_generates_plan_filtering_structural_steps():
    """evaluate_docs / reasoning_synthesis are graph stages now — the planner
    must filter them out of any LLM-emitted plan."""
    mock_llm = MagicMock()
    plan_json = json.dumps([
        {"action": "retrieve_local", "params": {"query": "attention", "top_k": 8}, "reason": "search locally"},
        {"action": "evaluate_docs", "params": {}, "reason": "check sufficiency"},
        {"action": "reasoning_synthesis", "params": {}, "reason": "generate answer"},
    ])
    mock_llm.invoke.return_value = MagicMock(content=plan_json)

    state = _base_state(intent={"type": "simple", "entities": ["attention"], "complexity": "low"})
    with patch("app.agent.nodes.planner._get_llm", return_value=mock_llm):
        result = planner_node(state, query="what is attention")

    assert [s["action"] for s in result["plan"]] == ["retrieve_local"]
    assert result["plan_step_index"] == 0


def test_planner_node_falls_back_when_no_executable_steps():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="not valid json")

    state = _base_state(intent={"type": "simple", "entities": [], "complexity": "low"})
    with patch("app.agent.nodes.planner._get_llm", return_value=mock_llm):
        result = planner_node(state, query="what is attention")

    assert [s["action"] for s in result["plan"]] == ["retrieve_local"]


def test_planner_node_normalizes_query_list_params_for_live_retrieval():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content=json.dumps([
        {
            "action": "retrieve_arxiv",
            "params": {"queries": ["voice agent benchmark", "MoE flow matching"]},
            "reason": "supplement live search",
        }
    ]))

    state = _base_state(intent={"type": "comparison", "entities": [], "complexity": "high"})
    with patch("app.agent.nodes.planner._get_llm", return_value=mock_llm):
        result = planner_node(state, query="compare the two papers")

    assert result["plan"][0]["action"] == "retrieve_arxiv"
    assert result["plan"][0]["params"] == {
        "query": "voice agent benchmark",
        "max_results": 5,
    }


def test_planner_node_marks_defaulted_retrieval_query_for_rewrite():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content=json.dumps([
        {
            "action": "retrieve_local",
            "params": {"top_k": 4},
            "reason": "search after rewrite",
        }
    ]))
    state = _base_state(intent={"type": "simple", "entities": [], "complexity": "low"})

    with patch("app.agent.nodes.planner._get_llm", return_value=mock_llm):
        result = planner_node(state, query="original question")

    assert result["plan"][0]["params"] == {
        "query": "original question",
        "top_k": 4,
        "_query_defaulted": True,
    }


def test_re_planner_sanitizes_structural_steps_and_fills_missing_query():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content=json.dumps([
        {"action": "reasoning_synthesis", "params": {}, "reason": "answer"},
        {"action": "retrieve_local", "params": {}, "reason": "fallback retry"},
        {"action": "evaluate_docs", "params": {}, "reason": "check"},
    ]))
    old_plan = [
        StepSpec(action="retrieve_local", params={"query": "original", "top_k": 8}, reason="search"),
    ]
    state = _base_state(plan=old_plan, plan_step_index=len(old_plan))

    with patch("app.agent.nodes.planner._get_llm", return_value=mock_llm):
        result = re_planner_node(
            state,
            query="tell me about Dr RTL",
            issues=["missing details"],
            missing_aspects=["Dr. RTL tool calling mechanism"],
        )

    appended = result["plan"][len(old_plan):]
    assert result["plan_step_index"] == len(old_plan)
    assert [step["action"] for step in appended] == ["retrieve_local"]
    assert appended[0]["params"]["query"] == "Dr. RTL tool calling mechanism"


def test_planner_and_replanner_filter_external_steps_in_local_only_mode():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content=json.dumps([
        {"action": "retrieve_arxiv", "params": {"query": "latest"}, "reason": "live"},
        {"action": "search_web", "params": {"query": "latest"}, "reason": "web"},
        {"action": "retrieve_local", "params": {"query": "local"}, "reason": "local"},
    ]))
    settings = MagicMock()
    settings.agent_max_plan_steps = 7
    settings.agent_external_retrieval_enabled = False
    state = _base_state(intent={"type": "comparison", "entities": [], "complexity": "high"})

    with patch("app.agent.nodes.planner._get_llm", return_value=mock_llm), patch(
        "app.agent.nodes.planner.get_settings", return_value=settings
    ):
        initial = planner_node(state, query="compare locally")
        replanned = re_planner_node(
            {**state, "plan": initial["plan"], "plan_step_index": len(initial["plan"])},
            query="compare locally",
            issues=["missing"],
            missing_aspects=[],
        )

    assert [step["action"] for step in initial["plan"]] == ["retrieve_local"]
    appended = replanned["plan"][len(initial["plan"]):]
    assert [step["action"] for step in appended] == ["retrieve_local"]
