"""Planner and re-planner nodes."""
from __future__ import annotations

import json
import time

from langchain_openai import ChatOpenAI

from app.agent.prompts.planner import PLANNER_PROMPT, RE_PLANNER_PROMPT
from app.agent.state import AgentState, StepSpec, StepTrace
from app.core.config import get_settings


def _get_llm() -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        model=s.planner_model or s.llm_model,
        base_url=s.llm_api_base,
        api_key=s.llm_api_key,
        temperature=0.3,
        max_retries=2,
        request_timeout=120,
    )


def _parse_plan(content: str, max_steps: int) -> list[StepSpec]:
    try:
        steps = json.loads(content)
        if isinstance(steps, list):
            return [
                StepSpec(
                    action=s.get("action", ""),
                    params=s.get("params", {}),
                    reason=s.get("reason", ""),
                )
                for s in steps[:max_steps]
            ]
    except (json.JSONDecodeError, TypeError):
        pass
    return [
        StepSpec(action="retrieve_local", params={"query": "", "top_k": 8}, reason="fallback"),
        StepSpec(action="evaluate_docs", params={}, reason="check sufficiency"),
        StepSpec(action="reasoning_synthesis", params={}, reason="generate answer"),
    ]


def planner_node(state: AgentState, *, query: str) -> dict:
    """Generate structured execution plan based on intent."""
    t0 = time.perf_counter()
    settings = get_settings()
    llm = _get_llm()

    intent = state["intent"] or {"type": "simple", "entities": [], "complexity": "low"}
    prompt = PLANNER_PROMPT.format(
        query=query,
        intent=json.dumps(intent, ensure_ascii=False),
        max_steps=settings.agent_max_plan_steps,
    )
    response = llm.invoke(prompt)
    plan = _parse_plan(response.content, settings.agent_max_plan_steps)

    duration = round((time.perf_counter() - t0) * 1000, 2)
    trace = StepTrace(
        node="planner_node",
        action="planning",
        input_summary=f"intent={intent.get('type')}, complexity={intent.get('complexity')}",
        output_summary=f"generated {len(plan)} steps",
        duration_ms=duration,
    )
    return {
        "plan": plan,
        "plan_step_index": 0,
        "step_traces": state["step_traces"] + [trace],
    }


def re_planner_node(state: AgentState, *, query: str, issues: list[str], missing_aspects: list[str]) -> dict:
    """Generate supplementary plan after reflection failure."""
    t0 = time.perf_counter()
    settings = get_settings()
    llm = _get_llm()

    prompt = RE_PLANNER_PROMPT.format(
        query=query,
        issues=json.dumps(issues, ensure_ascii=False),
        missing_aspects=json.dumps(missing_aspects, ensure_ascii=False),
    )
    response = llm.invoke(prompt)
    new_steps = _parse_plan(response.content, 3)

    duration = round((time.perf_counter() - t0) * 1000, 2)
    trace = StepTrace(
        node="re_planner_node",
        action="re_planning",
        input_summary=f"issues: {', '.join(issues[:2])}",
        output_summary=f"generated {len(new_steps)} supplementary steps",
        duration_ms=duration,
    )
    return {
        "plan": state["plan"] + new_steps,
        "step_traces": state["step_traces"] + [trace],
    }
