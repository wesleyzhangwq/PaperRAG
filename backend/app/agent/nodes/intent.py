"""Intent analysis node."""
from __future__ import annotations

import json
import time

from langchain_openai import ChatOpenAI

from app.agent.prompts.intent import INTENT_PROMPT
from app.agent.state import AgentState, StepTrace
from app.core.config import get_settings


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
    llm = _get_llm()
    prompt = INTENT_PROMPT.format(query=query)
    response = llm.invoke(prompt)

    try:
        intent = json.loads(response.content)
    except (json.JSONDecodeError, TypeError):
        intent = {"type": "simple", "entities": [], "complexity": "low"}

    duration = round((time.perf_counter() - t0) * 1000, 2)
    trace = StepTrace(
        node="intent_node",
        action="intent_analysis",
        input_summary=query[:100],
        output_summary=f"type={intent.get('type')}, complexity={intent.get('complexity')}",
        duration_ms=duration,
    )
    return {"intent": intent, "step_traces": state["step_traces"] + [trace]}
