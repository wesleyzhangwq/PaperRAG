"""Intent analysis node."""
from __future__ import annotations

import time

from langchain_openai import ChatOpenAI

from app.agent.prompts.intent import INTENT_PROMPT
from app.agent.stages import stage
from app.agent.state import AgentState, StepTrace
from app.agent.telemetry import classify_failure, record_fallback
from app.core.config import get_settings
from app.observability.llm_usage import invoke_with_usage
from app.utils.llm_json import extract_json

_TYPE_LABELS = {"simple": "事实型", "complex": "综合型", "comparison": "对比型"}


def _get_llm() -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        model=s.planner_model or s.llm_model,
        base_url=s.llm_api_base,
        api_key=s.llm_api_key,
        temperature=0.1,
        max_retries=2,
        request_timeout=120,
    )


def intent_node(state: AgentState, *, query: str) -> dict:
    """Analyze user intent: type, entities, complexity."""
    t0 = time.perf_counter()
    with stage("intent") as s:
        llm = _get_llm()
        prompt = INTENT_PROMPT.format(query=query)
        settings = get_settings()
        telemetry = None
        try:
            response = invoke_with_usage(
                llm,
                prompt,
                node="intent",
                model=settings.planner_model or settings.llm_model,
                api_base=settings.llm_api_base,
            )
            intent = extract_json(response.content)
        except Exception as exc:
            intent = {"type": "simple", "entities": [], "complexity": "low"}
            telemetry = record_fallback(
                state,
                failure_class=classify_failure(exc, default="intent_llm_failure"),
                stage="intent",
                outcome="default_intent",
            )
        else:
            if not isinstance(intent, dict):
                intent = {"type": "simple", "entities": [], "complexity": "low"}
                telemetry = record_fallback(
                    state,
                    failure_class="intent_output_unparseable",
                    stage="intent",
                    outcome="default_intent",
                )

        type_label = _TYPE_LABELS.get(str(intent.get("type")), str(intent.get("type") or ""))
        entities = [str(e) for e in (intent.get("entities") or [])[:3]]
        summary = f"{type_label}问题" + (f"，关键实体：{'、'.join(entities)}" if entities else "")
        if telemetry:
            s.warning(summary + "（使用安全默认意图）", detail=intent)
        else:
            s.done(summary, detail=intent)

    duration = round((time.perf_counter() - t0) * 1000, 2)
    trace = StepTrace(
        node="intent_node",
        action="intent_analysis",
        input_summary=query[:100],
        output_summary=f"type={intent.get('type')}, complexity={intent.get('complexity')}",
        duration_ms=duration,
    )
    out = {"intent": intent, "step_traces": state["step_traces"] + [trace]}
    if telemetry:
        out["fallback_telemetry"] = telemetry
    return out
