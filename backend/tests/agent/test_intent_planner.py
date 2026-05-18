"""Test intent and planner nodes."""
import json
from unittest.mock import patch, MagicMock

from app.agent.state import AgentState, StepSpec
from app.agent.nodes.intent import intent_node
from app.agent.nodes.planner import planner_node


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


def test_planner_node_generates_plan():
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

    assert len(result["plan"]) == 3
    assert result["plan"][0]["action"] == "retrieve_local"
    assert result["plan_step_index"] == 0
