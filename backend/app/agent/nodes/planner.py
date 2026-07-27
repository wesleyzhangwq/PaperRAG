"""Planner and re-planner nodes (查询理解与查询规划 / 补充规划).

Plans contain ONLY executable tool steps. Sufficiency evaluation and answer
synthesis are structural graph nodes in the v2 orchestration — the planner
no longer schedules them, and any such steps an LLM emits are filtered out.
"""
from __future__ import annotations

import json
import time

from langchain_openai import ChatOpenAI

from app.agent.nodes.complexity_router import mark_fast_path_escalated
from app.agent.prompts.planner import PLANNER_PROMPT, RE_PLANNER_PROMPT
from app.agent.stages import emit_plan, stage
from app.agent.state import AgentState, StepSpec, StepTrace
from app.agent.telemetry import classify_failure, record_fallback
from app.core.config import get_settings
from app.observability.llm_usage import invoke_with_usage
from app.utils.llm_json import extract_json

# Actions the executor can dispatch. evaluate_docs / reasoning_synthesis are
# graph-level stages now and must never reach the executor.
EXECUTABLE_ACTIONS = {
    "query_rewrite", "retrieve_local", "retrieve_arxiv", "search_web",
    "get_paper_detail", "get_paper_chunks",
}
QUERY_ACTIONS = {"retrieve_local", "retrieve_arxiv", "search_web"}
EXTERNAL_RETRIEVAL_ACTIONS = {"retrieve_arxiv", "search_web"}


def _filter_external_steps(steps: list[StepSpec], *, enabled: bool) -> list[StepSpec]:
    if enabled:
        return steps
    return [step for step in steps if step["action"] not in EXTERNAL_RETRIEVAL_ACTIONS]


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


def _first_query(params: dict) -> str:
    query = str(params.get("query") or "").strip()
    if query:
        return query
    queries = params.get("queries")
    if isinstance(queries, list):
        for item in queries:
            text = str(item or "").strip()
            if text:
                return text
    if isinstance(queries, str):
        return queries.strip()
    return ""


def _normalize_params(action: str, params: dict, default_query: str) -> dict:
    cleaned = dict(params or {})
    if action in QUERY_ACTIONS:
        supplied_query = _first_query(cleaned)
        query = supplied_query or default_query
        cleaned.pop("queries", None)
        cleaned["query"] = query
        if supplied_query:
            cleaned.pop("_query_defaulted", None)
        else:
            cleaned["_query_defaulted"] = True
        if action == "retrieve_local":
            cleaned["top_k"] = cleaned.get("top_k") or 8
        elif action in {"retrieve_arxiv", "search_web"}:
            cleaned["max_results"] = cleaned.get("max_results") or 5
    return cleaned


def _parse_plan_with_status(
    content: str, max_steps: int, *, default_query: str = ""
) -> tuple[list[StepSpec], bool]:
    steps = extract_json(content)
    if isinstance(steps, list):
        parsed = [
            StepSpec(
                action=s.get("action", ""),
                params=_normalize_params(s.get("action", ""), s.get("params", {}), default_query),
                reason=s.get("reason", ""),
            )
            for s in steps
            if isinstance(s, dict) and s.get("action") in EXECUTABLE_ACTIONS
        ]
        if parsed:
            return parsed[:max_steps], False
    return (
        [StepSpec(action="retrieve_local", params={"query": "", "top_k": 8}, reason="fallback")],
        True,
    )


def _parse_plan(content: str, max_steps: int, *, default_query: str = "") -> list[StepSpec]:
    """Compatibility wrapper retained for focused planner tests."""
    return _parse_plan_with_status(content, max_steps, default_query=default_query)[0]


def _supplement_query(query: str, issues: list[str], missing_aspects: list[str]) -> str:
    for item in missing_aspects + issues + [query]:
        text = str(item or "").strip()
        if text:
            return text
    return query


def _sanitize_supplementary_steps(
    steps: list[StepSpec],
    *,
    query: str,
    issues: list[str],
    missing_aspects: list[str],
) -> list[StepSpec]:
    """Keep only executable retrieval steps for the supplement loop."""
    fallback = _supplement_query(query, issues, missing_aspects)
    external_enabled = bool(
        getattr(get_settings(), "agent_external_retrieval_enabled", True)
    )
    sanitized: list[StepSpec] = []
    seen: set[tuple[str, str]] = set()
    for step in steps:
        action = step.get("action", "")
        if action not in EXECUTABLE_ACTIONS:
            continue
        if not external_enabled and action in EXTERNAL_RETRIEVAL_ACTIONS:
            continue
        params = dict(step.get("params") or {})
        if action in {"retrieve_local", "retrieve_arxiv", "search_web"}:
            text = str(params.get("query") or "").strip() or fallback
            params["query"] = text
            if action == "retrieve_local":
                params["top_k"] = params.get("top_k") or 4
            if action == "search_web":
                params["max_results"] = params.get("max_results") or 3
        key = (action, str(params.get("query") or ""))
        if key in seen:
            continue
        seen.add(key)
        sanitized.append(StepSpec(
            action=action,
            params=params,
            reason=step.get("reason", "") or f"supplement: {params.get('query', fallback)}",
        ))

    if not any(step["action"].startswith("retrieve_") or step["action"] == "search_web" for step in sanitized):
        sanitized.insert(0, StepSpec(
            action="retrieve_local",
            params={"query": fallback, "top_k": 4},
            reason=f"supplement: {fallback}",
        ))
    return sanitized[:3]


def planner_node(state: AgentState, *, query: str) -> dict:
    """Generate structured execution plan based on intent."""
    t0 = time.perf_counter()
    settings = get_settings()

    with stage("plan") as s:
        llm = _get_llm()
        intent = state["intent"] or {"type": "simple", "entities": [], "complexity": "low"}
        prompt = PLANNER_PROMPT.format(
            query=query,
            intent=json.dumps(intent, ensure_ascii=False),
            max_steps=settings.agent_max_plan_steps,
        )
        telemetry = None
        try:
            response = invoke_with_usage(
                llm,
                prompt,
                node="planner",
                model=settings.planner_model or settings.llm_model,
                api_base=settings.llm_api_base,
            )
            plan, parse_failed = _parse_plan_with_status(
                response.content,
                settings.agent_max_plan_steps,
                default_query=query,
            )
        except Exception as exc:
            plan, parse_failed = _parse_plan_with_status(
                "", settings.agent_max_plan_steps, default_query=query
            )
            telemetry = record_fallback(
                state,
                failure_class=classify_failure(exc, default="planner_llm_failure"),
                stage="planner",
                outcome="fallback_local_plan",
            )
        if parse_failed and telemetry is None:
            telemetry = record_fallback(
                state,
                failure_class="planner_output_unparseable",
                stage="planner",
                outcome="fallback_local_plan",
            )
        plan = _filter_external_steps(
            plan,
            enabled=bool(getattr(settings, "agent_external_retrieval_enabled", True)),
        )
        if not plan:
            plan = [StepSpec(
                action="retrieve_local",
                params={"query": query, "top_k": settings.retrieval_k},
                reason="本地检索模式：使用语料库检索",
            )]
        emit_plan(plan, revision=1 if state.get("fast_path_escalated") else 0)
        detail = {"steps": [p["action"] for p in plan], "fallback_plan": bool(telemetry)}
        if telemetry:
            s.warning(f"{len(plan)} 个检索步骤（安全回退计划）", detail=detail)
        else:
            s.done(f"{len(plan)} 个检索步骤", detail=detail)

    duration = round((time.perf_counter() - t0) * 1000, 2)
    trace = StepTrace(
        node="planner_node",
        action="planning",
        input_summary=f"intent={intent.get('type')}, complexity={intent.get('complexity')}",
        output_summary=f"generated {len(plan)} steps",
        duration_ms=duration,
    )
    out = {
        "plan": plan,
        "plan_step_index": 0,
        "step_traces": state["step_traces"] + [trace],
    }
    if telemetry:
        out["fallback_telemetry"] = telemetry
    return out


def re_planner_node(state: AgentState, *, query: str, issues: list[str], missing_aspects: list[str]) -> dict:
    """Generate supplementary retrieval plan after a sufficiency or
    groundedness failure."""
    t0 = time.perf_counter()

    with stage("plan", stage_id=f"plan:r{int(state.get('sufficiency_round', 0)) + int(state.get('reflection_count', 0))}",
               title="补充检索计划") as s:
        llm = _get_llm()
        settings = get_settings()
        prompt = RE_PLANNER_PROMPT.format(
            query=query,
            issues=json.dumps(issues, ensure_ascii=False),
            missing_aspects=json.dumps(missing_aspects, ensure_ascii=False),
        )
        telemetry = None
        try:
            response = invoke_with_usage(
                llm,
                prompt,
                node="re_planner",
                model=settings.planner_model or settings.llm_model,
                api_base=settings.llm_api_base,
            )
            raw_steps, parse_failed = _parse_plan_with_status(
                response.content,
                3,
                default_query=_supplement_query(query, issues, missing_aspects),
            )
        except Exception as exc:
            raw_steps, parse_failed = _parse_plan_with_status(
                "", 3, default_query=_supplement_query(query, issues, missing_aspects)
            )
            telemetry = record_fallback(
                state,
                failure_class=classify_failure(exc, default="re_planner_llm_failure"),
                stage="re_planner",
                outcome="deterministic_supplement_plan",
            )
        if parse_failed and telemetry is None:
            telemetry = record_fallback(
                state,
                failure_class="re_planner_output_unparseable",
                stage="re_planner",
                outcome="deterministic_supplement_plan",
            )
        new_steps = _sanitize_supplementary_steps(
            raw_steps,
            query=query,
            issues=issues,
            missing_aspects=missing_aspects,
        )
        full_plan = state["plan"] + new_steps
        emit_plan(full_plan, revision=1)
        detail = {"steps": [p["action"] for p in new_steps], "fallback_plan": bool(telemetry)}
        if telemetry:
            s.warning(f"补充 {len(new_steps)} 个检索步骤（确定性回退）", detail=detail)
        else:
            s.done(f"补充 {len(new_steps)} 个检索步骤", detail=detail)

    duration = round((time.perf_counter() - t0) * 1000, 2)
    trace = StepTrace(
        node="re_planner_node",
        action="re_planning",
        input_summary=f"issues: {', '.join(issues[:2])}",
        output_summary=f"generated {len(new_steps)} supplementary steps",
        duration_ms=duration,
    )
    out = {
        "plan": full_plan,
        "plan_step_index": len(state["plan"]),
        "step_traces": state["step_traces"] + [trace],
    }
    if state.get("execution_path") == "fast_local":
        out.update(
            {
                "execution_path": "fast_escalated",
                "fast_path_escalated": True,
                "complexity_decision": mark_fast_path_escalated(
                    state.get("complexity_decision"),
                    reason_code="groundedness_retrieval_escalation",
                ),
            }
        )
    if telemetry:
        out["fallback_telemetry"] = telemetry
    return out
