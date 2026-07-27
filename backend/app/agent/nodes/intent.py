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
_INTENT_TYPES = frozenset(_TYPE_LABELS)
_COMPLEXITIES = frozenset({"low", "medium", "high"})
_FAIL_CLOSED_INTENT = {"type": "complex", "entities": [], "complexity": "high"}


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


def _valid_intent(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    entities = value.get("entities")
    return (
        value.get("type") in _INTENT_TYPES
        and value.get("complexity") in _COMPLEXITIES
        and isinstance(entities, list)
        and all(isinstance(entity, str) for entity in entities)
    )


def intent_node(state: AgentState, *, query: str) -> dict:
    """Analyze user intent: type, entities, complexity."""
    t0 = time.perf_counter()
    with stage("intent") as s:
        prompt = INTENT_PROMPT.format(query=query)
        telemetry = None
        try:
            llm = _get_llm()
            settings = get_settings()
            response = invoke_with_usage(
                llm,
                prompt,
                node="intent",
                model=settings.planner_model or settings.llm_model,
                api_base=settings.llm_api_base,
            )
            intent = extract_json(response.content)
        except Exception as exc:
            # Fail closed for complexity routing: an unavailable classifier
            # must never be mistaken for a high-confidence fast-path signal.
            intent = dict(_FAIL_CLOSED_INTENT)
            telemetry = record_fallback(
                state,
                failure_class=classify_failure(exc, default="intent_llm_failure"),
                stage="intent",
                outcome="default_intent",
            )
        else:
            if not _valid_intent(intent):
                intent = dict(_FAIL_CLOSED_INTENT)
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
    out = {
        "intent": intent,
        "intent_status": "fallback" if telemetry else "ok",
        "step_traces": state["step_traces"] + [trace],
    }
    if telemetry:
        out["fallback_telemetry"] = telemetry
    return out
