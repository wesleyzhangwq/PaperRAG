"""Truth-table tests for the explainable complexity router."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.graph import route_after_complexity
from app.agent.nodes.complexity_router import (
    COMPLEXITY_ROUTER_POLICY_VERSION,
    complexity_router_node,
)
from app.agent.state import AgentState


def _state(
    query: str,
    *,
    intent_type: str = "simple",
    complexity: str = "low",
    entities: list[str] | None = None,
    intent_status: str = "ok",
    history: bool = False,
) -> AgentState:
    messages = []
    if history:
        messages.extend(
            [
                HumanMessage(content="previous question"),
                AIMessage(content="previous answer"),
            ]
        )
    messages.append(HumanMessage(content=query))
    return {
        "messages": messages,
        "intent": {
            "type": intent_type,
            "complexity": complexity,
            "entities": list(entities or []),
        },
        "intent_status": intent_status,
        "plan": [],
        "plan_step_index": 0,
        "retrieval_context": [],
        "step_traces": [],
        "reflection_count": 0,
        "sufficiency_round": 0,
    }


def _route(state: AgentState, query: str, *, mode: str = "auto") -> dict:
    settings = SimpleNamespace(agent_routing_mode=mode, retrieval_k=20)
    with patch(
        "app.agent.nodes.complexity_router.get_settings",
        return_value=settings,
    ):
        return complexity_router_node(state, query=query)


def test_simple_low_risk_query_uses_deterministic_fast_local_plan() -> None:
    query = "什么是注意力机制？"
    result = _route(_state(query, entities=["注意力机制"]), query)

    assert result["execution_path"] == "fast_local"
    assert result["plan"] == [
        {
            "action": "retrieve_local",
            "params": {"query": query, "top_k": 20},
            "reason": "complexity_router_fast_local",
        }
    ]
    assert result["plan_step_index"] == 0
    assert result["complexity_decision"]["policy_version"] == COMPLEXITY_ROUTER_POLICY_VERSION
    assert result["complexity_decision"]["initial_path"] == "fast_local"
    assert result["complexity_decision"]["final_path"] == "fast_local"
    assert "intent_simple" in result["complexity_decision"]["reason_codes"]
    assert route_after_complexity({**state_or_empty(), **result}) == "route"


def state_or_empty() -> AgentState:
    return {
        "messages": [],
        "intent": None,
        "plan": [],
        "plan_step_index": 0,
        "retrieval_context": [],
        "step_traces": [],
        "reflection_count": 0,
        "sufficiency_round": 0,
    }


@pytest.mark.parametrize(
    ("query", "intent_type", "complexity", "entities", "intent_status", "history", "veto"),
    [
        ("比较 BERT 与 GPT", "comparison", "high", ["BERT", "GPT"], "ok", False, "intent_not_simple"),
        ("比较 BERT 与 GPT", "simple", "low", ["BERT"], "ok", False, "comparison_query"),
        ("2026 年最新 Agent 研究", "simple", "low", ["Agent"], "ok", False, "recency_query"),
        ("请总结这些论文的共同趋势", "simple", "low", ["论文"], "ok", False, "multi_paper_query"),
        ("语料库中是否有蛋白质折叠论文", "simple", "low", ["蛋白质折叠"], "ok", False, "scope_check_query"),
        ("有哪些论文研究深海热液喷口？", "simple", "low", ["深海热液喷口"], "ok", False, "scope_check_query"),
        ("解释这个方法", "simple", "low", ["方法"], "ok", True, "history_present"),
        ("A 和 B 分别是什么", "simple", "low", ["A", "B"], "ok", False, "multiple_entities"),
        ("什么是注意力", "simple", "low", ["注意力"], "fallback", False, "intent_fallback"),
    ],
)
def test_risk_or_uncertainty_routes_full_agentic(
    query: str,
    intent_type: str,
    complexity: str,
    entities: list[str],
    intent_status: str,
    history: bool,
    veto: str,
) -> None:
    result = _route(
        _state(
            query,
            intent_type=intent_type,
            complexity=complexity,
            entities=entities,
            intent_status=intent_status,
            history=history,
        ),
        query,
    )

    assert result["execution_path"] == "full_agentic"
    assert result["plan"] == []
    assert veto in result["complexity_decision"]["vetoes"]
    assert route_after_complexity({**state_or_empty(), **result}) == "planner"


def test_full_agentic_mode_disables_fast_path() -> None:
    query = "什么是注意力机制？"
    result = _route(
        _state(query, entities=["注意力机制"]),
        query,
        mode="full_agentic",
    )

    assert result["execution_path"] == "full_agentic"
    assert result["complexity_decision"]["vetoes"] == ["mode_full_agentic"]


def test_missing_routing_setting_fails_closed_to_full_agentic() -> None:
    query = "什么是注意力机制？"
    settings = SimpleNamespace(retrieval_k=20)
    with patch(
        "app.agent.nodes.complexity_router.get_settings",
        return_value=settings,
    ):
        result = complexity_router_node(
            _state(query, entities=["注意力机制"]),
            query=query,
        )

    assert result["execution_path"] == "full_agentic"
    assert result["complexity_decision"]["mode"] == "full_agentic"
    assert result["complexity_decision"]["vetoes"] == ["mode_full_agentic"]


def test_complexity_decision_does_not_persist_raw_query_or_reasoning() -> None:
    query = "PRIVATE raw query with hidden chain of thought"
    result = _route(_state(query, entities=["private"]), query)

    serialized = json.dumps(result["complexity_decision"], ensure_ascii=False)
    assert query not in serialized
    assert "chain of thought" not in serialized
    assert result["complexity_decision"]["features"]["query_chars"] == len(query)
