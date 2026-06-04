"""Self-reflection node: verify answer quality."""
from __future__ import annotations

import json
import time

from langchain_openai import ChatOpenAI

from app.agent.prompts.reflection import REFLECTION_PROMPT
from app.agent.state import AgentState, ReflectionResult, StepTrace
from app.core.config import get_settings


def _get_llm() -> ChatOpenAI:
    s = get_settings()
    model = s.reflection_model or s.llm_model
    return ChatOpenAI(
        model=model,
        base_url=s.llm_api_base,
        api_key=s.llm_api_key,
        temperature=0.1,
        max_retries=2,
        request_timeout=120,
    )


def reflection_node(state: AgentState, *, query: str) -> dict:
    """Verify answer along 3 dimensions: citation, completeness, logic."""
    t0 = time.perf_counter()

    evaluator = state.get("evaluator_result") or {}
    if evaluator and not evaluator.get("parse_failed") and evaluator.get("sufficient") is False:
        missing = [str(x) for x in evaluator.get("missing_aspects", []) if str(x).strip()]
        reason = str(evaluator.get("reason", "")).strip()
        issues = []
        if missing:
            issues.append("Evaluator reported missing aspects: " + "; ".join(missing))
        if reason:
            issues.append("Evaluator reason: " + reason)
        if not issues:
            issues.append("Evaluator reported that retrieved context is insufficient.")
        reflection = ReflectionResult(
            passed=False,
            citation_ok=True,
            completeness_ok=False,
            logic_ok=True,
            issues=issues,
            fix_strategy="re_retrieve",
        )
        duration = round((time.perf_counter() - t0) * 1000, 2)
        trace = StepTrace(
            node="reflection_node",
            action="self_reflection",
            input_summary="using evaluator insufficiency signal",
            output_summary=f"passed=False, strategy={reflection.get('fix_strategy')}",
            duration_ms=duration,
        )
        return {
            "reflection_result": reflection,
            "reflection_count": state["reflection_count"] + 1,
            "step_traces": state["step_traces"] + [trace],
        }

    llm = _get_llm()

    paper_ids = set()
    for d in state["retrieval_context"]:
        pid = (d.metadata or {}).get("paper_id")
        if pid:
            paper_ids.add(pid)

    prompt = REFLECTION_PROMPT.format(
        query=query,
        available_paper_ids=", ".join(sorted(paper_ids)),
        answer=state["final_answer"] or "",
    )
    response = llm.invoke(prompt)

    try:
        result = json.loads(response.content)
        reflection = ReflectionResult(
            passed=bool(result.get("passed", False)),
            citation_ok=bool(result.get("citation_ok", True)),
            completeness_ok=bool(result.get("completeness_ok", True)),
            logic_ok=bool(result.get("logic_ok", True)),
            issues=list(result.get("issues", [])),
            fix_strategy=result.get("fix_strategy"),
        )
    except (json.JSONDecodeError, TypeError):
        reflection = ReflectionResult(
            passed=False,
            citation_ok=False,
            completeness_ok=False,
            logic_ok=False,
            issues=["Reflection output was not valid JSON; answer quality could not be verified."],
            fix_strategy="re_retrieve",
        )

    duration = round((time.perf_counter() - t0) * 1000, 2)
    trace = StepTrace(
        node="reflection_node",
        action="self_reflection",
        input_summary=f"verifying answer ({len(state.get('final_answer', '') or '')} chars)",
        output_summary=f"passed={reflection['passed']}, strategy={reflection.get('fix_strategy')}",
        duration_ms=duration,
    )

    # Only increment reflection_count on failure (to track retry budget)
    new_count = state["reflection_count"] if reflection["passed"] else state["reflection_count"] + 1

    return {
        "reflection_result": reflection,
        "reflection_count": new_count,
        "step_traces": state["step_traces"] + [trace],
    }
