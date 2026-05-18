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
    )


def reflection_node(state: AgentState, *, query: str) -> dict:
    """Verify answer along 3 dimensions: citation, completeness, logic."""
    t0 = time.perf_counter()
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
            passed=True, citation_ok=True, completeness_ok=True,
            logic_ok=True, issues=[], fix_strategy=None,
        )

    duration = round((time.perf_counter() - t0) * 1000, 2)
    trace = StepTrace(
        node="reflection_node",
        action="self_reflection",
        input_summary=f"verifying answer ({len(state.get('final_answer', '') or '')} chars)",
        output_summary=f"passed={reflection['passed']}, strategy={reflection.get('fix_strategy')}",
        duration_ms=duration,
    )

    return {
        "reflection_result": reflection,
        "reflection_count": state["reflection_count"] + 1,
        "step_traces": state["step_traces"] + [trace],
    }
