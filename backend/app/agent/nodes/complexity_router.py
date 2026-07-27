"""Explainable routing between the safe fast path and full Agentic planning.

The router deliberately does not generate answers or bypass any evidence
checks.  A high-confidence simple query receives one deterministic local
retrieval step; every uncertain or risky query keeps the full planner.  Both
paths converge before evidence sufficiency, synthesis, groundedness, and the
citation gate.

Only bounded features and reason codes are persisted.  Raw query text is used
to construct the executable retrieval step but is never copied into the
``complexity_decision`` audit payload.
"""
from __future__ import annotations

import re
import time

from langchain_core.messages import HumanMessage

from app.agent.stages import emit_plan, stage
from app.agent.state import AgentState, StepSpec, StepTrace
from app.core.config import get_settings

COMPLEXITY_ROUTER_POLICY_VERSION = "complexity-router-v1"

_RECENCY_RE = re.compile(
    r"(最新|最近|新进展|前沿|今年|去年|这两年|"
    r"state[- ]of[- ]the[- ]art|sota|latest|recent|2024|2025|2026)",
    re.IGNORECASE,
)
_COMPARISON_RE = re.compile(
    r"(比较|对比|区别|差异|异同|本质不同|"
    r"\bvs\.?\b|\bversus\b|\bcompare\b|\bdifference(?:s)?\b)",
    re.IGNORECASE,
)
_MULTI_PAPER_RE = re.compile(
    r"(这些论文|这组论文|多篇论文|跨论文|共同趋势|共同特征|"
    r"综述|趋势综合|综合分析|分别采取|演变趋势|"
    r"\bsurvey\b|\breview\b|\btrend(?:s)?\b|\bsynthesi[sz]e\b)",
    re.IGNORECASE,
)
_SCOPE_CHECK_RE = re.compile(
    r"(语料库中|本语料库|这些论文中|是否有.{0,12}论文|"
    r"有没有.{0,12}论文|有哪些论文|哪些论文|"
    r"是否.{0,24}论文|论文.{0,24}(?:吗|？)|不包含|未涵盖|"
    r"out[- ]of[- ]scope|in (?:the )?corpus)",
    re.IGNORECASE,
)
_EXTERNAL_RE = re.compile(
    r"(新闻|公司|产品|行业|市场|官网|网页|网站|实时|"
    r"\bnews\b|\bcompany\b|\bproduct\b|\bmarket\b|\bwebsite\b)",
    re.IGNORECASE,
)


def _history_turns(state: AgentState) -> int:
    human_messages = sum(
        1
        for message in state.get("messages") or []
        if isinstance(message, HumanMessage)
        or getattr(message, "type", "") == "human"
        or (isinstance(message, tuple) and len(message) == 2 and message[0] == "user")
    )
    return max(0, human_messages - 1)


def _feature_snapshot(state: AgentState, query: str) -> dict:
    intent = state.get("intent") or {}
    entities = intent.get("entities") or []
    if not isinstance(entities, list):
        entities = []
    return {
        "intent_status": str(state.get("intent_status") or "unknown"),
        "intent_type": str(intent.get("type") or "unknown"),
        "intent_complexity": str(intent.get("complexity") or "unknown"),
        "entity_count": len(entities),
        "history_turns": _history_turns(state),
        "query_chars": len(query or ""),
        "has_recency": bool(_RECENCY_RE.search(query or "")),
        "has_comparison": bool(_COMPARISON_RE.search(query or "")),
        "has_multi_paper": bool(_MULTI_PAPER_RE.search(query or "")),
        "has_scope_check": bool(_SCOPE_CHECK_RE.search(query or "")),
        "has_external_need": bool(_EXTERNAL_RE.search(query or "")),
    }


def _vetoes(features: dict, *, mode: str) -> list[str]:
    if mode == "full_agentic":
        return ["mode_full_agentic"]

    vetoes: list[str] = []
    if features["intent_status"] != "ok":
        vetoes.append("intent_fallback")
    if features["intent_type"] != "simple":
        vetoes.append("intent_not_simple")
    if features["intent_complexity"] != "low":
        vetoes.append("complexity_not_low")
    if features["entity_count"] > 1:
        vetoes.append("multiple_entities")
    if features["history_turns"] > 0:
        vetoes.append("history_present")
    if features["has_recency"]:
        vetoes.append("recency_query")
    if features["has_comparison"]:
        vetoes.append("comparison_query")
    if features["has_multi_paper"]:
        vetoes.append("multi_paper_query")
    if features["has_scope_check"]:
        vetoes.append("scope_check_query")
    if features["has_external_need"]:
        vetoes.append("external_information_query")
    return vetoes


def complexity_router_node(state: AgentState, *, query: str) -> dict:
    """Choose and explain the initial execution path.

    ``fast_local`` means exactly one deterministic local retrieval step before
    joining the normal evidence/safety chain.  Every other decision is
    ``full_agentic`` and leaves planning to the existing LLM planner.
    """
    t0 = time.perf_counter()
    settings = get_settings()
    mode = str(getattr(settings, "agent_routing_mode", "full_agentic"))
    features = _feature_snapshot(state, query)
    vetoes = _vetoes(features, mode=mode)
    fast = not vetoes

    if fast:
        execution_path = "fast_local"
        reason_codes = [
            "intent_simple",
            "complexity_low",
            "no_risk_veto",
            "deterministic_local_plan",
        ]
        plan = [
            StepSpec(
                action="retrieve_local",
                params={"query": query, "top_k": settings.retrieval_k},
                reason="complexity_router_fast_local",
            )
        ]
        emit_plan(plan, revision=0)
    else:
        execution_path = "full_agentic"
        reason_codes = ["conservative_full_agentic", *vetoes]
        plan = []

    decision = {
        "policy_version": COMPLEXITY_ROUTER_POLICY_VERSION,
        "mode": mode,
        "initial_path": execution_path,
        "final_path": execution_path,
        "confidence": "high" if fast else "conservative",
        "reason_codes": reason_codes,
        "vetoes": vetoes,
        "features": features,
        "escalated": False,
    }

    with stage("complexity", detail={"policy_version": COMPLEXITY_ROUTER_POLICY_VERSION}) as s:
        if fast:
            s.done("使用单次本地检索快路径", detail=decision)
        else:
            s.done("使用完整 Agentic 规划路径", detail=decision)

    duration = round((time.perf_counter() - t0) * 1000, 2)
    trace = StepTrace(
        node="complexity_router_node",
        action="complexity_route",
        input_summary=(
            f"intent={features['intent_type']}, "
            f"complexity={features['intent_complexity']}, "
            f"history_turns={features['history_turns']}"
        ),
        output_summary=f"path={execution_path}, vetoes={len(vetoes)}",
        duration_ms=duration,
        detail=decision,
    )
    return {
        "execution_path": execution_path,
        "fast_path_escalated": False,
        "complexity_decision": decision,
        "plan": plan,
        "plan_step_index": 0,
        "step_traces": state["step_traces"] + [trace],
    }


def mark_fast_path_escalated(
    value: dict | None,
    *,
    reason_code: str,
) -> dict:
    """Return an immutable-style audit update for a fast-path escalation."""
    decision = dict(value or {})
    reason_codes = list(decision.get("reason_codes") or [])
    if reason_code not in reason_codes:
        reason_codes.append(reason_code)
    decision.update(
        {
            "final_path": "fast_escalated",
            "confidence": "revised",
            "escalated": True,
            "reason_codes": reason_codes,
        }
    )
    return decision


__all__ = [
    "COMPLEXITY_ROUTER_POLICY_VERSION",
    "complexity_router_node",
    "mark_fast_path_escalated",
]
