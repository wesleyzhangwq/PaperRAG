"""Retrieval router node: deterministic source-routing policy.

Maps to the enterprise pipeline's 检索路由 stage. The planner proposes steps;
this node enforces routing policy and records an explainable decision:

- a plan with no retrieval at all gets a local-retrieval step injected
- recency-sensitive queries get arXiv/web supplements (when configured)
- the chosen source mix is recorded as ``route_decision`` for the UI

Deterministic by design — policy is data (intent + plan), not another LLM call.
"""
from __future__ import annotations

import re
import time

from app.agent.stages import emit_plan, stage
from app.agent.state import AgentState, StepSpec, StepTrace
from app.core.config import get_settings

_RETRIEVAL_ACTIONS = {"retrieve_local", "retrieve_arxiv", "search_web"}

_RECENCY_RE = re.compile(
    r"(最新|最近|新进展|前沿|今年|去年|这两年|state[- ]of[- ]the[- ]art|sota|latest|recent|2024|2025|2026)",
    re.IGNORECASE,
)

_SOURCE_LABELS = {
    "retrieve_local": "本地论文库",
    "retrieve_arxiv": "arXiv 在线",
    "search_web": "网络搜索",
}


def route_node(state: AgentState, *, query: str) -> dict:
    """Validate and adjust the plan's retrieval routing; record the decision."""
    t0 = time.perf_counter()
    with stage("route") as s:
        plan = list(state.get("plan") or [])
        adjustments: list[str] = []
        settings = get_settings()
        external_enabled = bool(
            getattr(settings, "agent_external_retrieval_enabled", True)
        )

        if not external_enabled:
            before = len(plan)
            plan = [
                step
                for step in plan
                if step["action"] not in {"retrieve_arxiv", "search_web"}
            ]
            if len(plan) != before:
                adjustments.append("dropped_external_retrieval_local_only")

        used_actions = {step["action"] for step in plan}
        retrieval_steps = [p for p in plan if p["action"] in _RETRIEVAL_ACTIONS]

        # Policy 1: never run the pipeline without at least one retrieval source.
        if not retrieval_steps:
            plan.insert(0, StepSpec(
                action="retrieve_local",
                params={"query": query, "top_k": 8},
                reason="路由策略：计划缺少检索步骤，注入本地检索",
            ))
            used_actions.add("retrieve_local")
            adjustments.append("injected_retrieve_local")

        # Policy 2: recency-sensitive queries need a live source.
        needs_recency = bool(_RECENCY_RE.search(query or ""))
        if external_enabled and needs_recency and "retrieve_arxiv" not in used_actions and "search_web" not in used_actions:
            insert_at = next(
                (i + 1 for i in range(len(plan) - 1, -1, -1) if plan[i]["action"] in _RETRIEVAL_ACTIONS),
                len(plan),
            )
            plan.insert(insert_at, StepSpec(
                action="retrieve_arxiv",
                params={"query": query, "max_results": settings.arxiv_max_results},
                reason="路由策略：问题涉及最新进展，补充 arXiv 在线检索",
            ))
            used_actions.add("retrieve_arxiv")
            adjustments.append("injected_retrieve_arxiv_for_recency")

        # Policy 3: drop web steps when the web tool is not configured
        # (executor would emit noisy 'web unavailable' otherwise).
        if external_enabled and "search_web" in used_actions and not settings.tavily_api_key:
            plan = [p for p in plan if p["action"] != "search_web"]
            used_actions.discard("search_web")
            adjustments.append("dropped_search_web_unconfigured")

        sources = [a for a in ("retrieve_local", "retrieve_arxiv", "search_web") if a in {p["action"] for p in plan}]
        route_decision = {
            "sources": sources,
            "source_labels": [_SOURCE_LABELS[s_] for s_ in sources],
            "needs_recency": needs_recency,
            "adjustments": adjustments,
        }

        # Plan changed → re-publish with stable ids so the frontend stays in sync.
        if adjustments:
            emit_plan(plan, revision=0)

        summary = "来源：" + "、".join(route_decision["source_labels"]) if sources else "无可用检索来源"
        if adjustments:
            s.warning(summary + f"（策略调整 {len(adjustments)} 项）", detail=route_decision)
        else:
            s.done(summary, detail=route_decision)

    duration = round((time.perf_counter() - t0) * 1000, 2)
    trace = StepTrace(
        node="route_node",
        action="route",
        input_summary=f"plan={len(state.get('plan') or [])} steps",
        output_summary=f"sources={','.join(sources)}" + (f", adjustments={adjustments}" if adjustments else ""),
        duration_ms=duration,
        detail=route_decision,
    )
    return {
        "plan": plan,
        "route_decision": route_decision,
        "step_traces": state["step_traces"] + [trace],
    }


__all__ = ["route_node"]
