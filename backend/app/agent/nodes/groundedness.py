"""Groundedness node: hallucination & quality verification of the answer.

Maps to the enterprise pipeline's groundedness 检查/幻觉检查 stage (Reflexion
pattern). Two-layer check:

1. deterministic precheck — every ``[arxiv:ID]`` cited in the answer must
   exist in the retrieval context (no LLM needed to catch a fabricated id)
2. LLM verification     — citation faithfulness / completeness / logical
   consistency, with a fix strategy on failure

Fix strategies: ``re_generate`` (context fine, answer flawed) re-runs
synthesis with issues as constraints; ``re_retrieve`` goes through re_planner
for supplementary retrieval. Budget: ``AGENT_MAX_REFLECTIONS``.
"""
from __future__ import annotations

import re
import time

from langchain_openai import ChatOpenAI

from app.agent.prompts.reflection import REFLECTION_PROMPT
from app.agent.stages import stage
from app.agent.state import AgentState, ReflectionResult, StepTrace
from app.core.config import get_settings
from app.utils.llm_json import extract_json

_CITATION_RE = re.compile(r"\[arxiv:([0-9]{4}\.[0-9]{4,6})\]")


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


def _context_paper_ids(state: AgentState) -> set[str]:
    synthesis_ids = state.get("synthesis_context_paper_ids")
    if synthesis_ids is not None:
        return {str(pid) for pid in synthesis_ids if str(pid or "").strip()}

    ids = set()
    for d in state.get("retrieval_context") or []:
        pid = (d.metadata or {}).get("paper_id")
        if pid:
            ids.add(pid)
    return ids


def _precheck_citations(answer: str, available_ids: set[str]) -> list[str]:
    """Return cited ids NOT present in the retrieval context (fabrications)."""
    cited = {m.group(1) for m in _CITATION_RE.finditer(answer or "")}
    return sorted(cited - available_ids)


def groundedness_node(state: AgentState, *, query: str) -> dict:
    """Verify the synthesized answer is grounded, complete and consistent."""
    t0 = time.perf_counter()
    with stage("groundedness") as s:
        answer = state.get("final_answer") or ""
        available_ids = _context_paper_ids(state)

        # Layer 1: deterministic fabricated-citation check.
        fabricated = _precheck_citations(answer, available_ids)
        if fabricated:
            reflection = ReflectionResult(
                passed=False,
                citation_ok=False,
                completeness_ok=True,
                logic_ok=True,
                issues=[f"答案引用了检索结果中不存在的论文: {', '.join(fabricated)}"],
                fix_strategy="re_generate",
            )
            s.warning(f"发现 {len(fabricated)} 个虚构引用，要求重新生成", detail=dict(reflection))
        else:
            # Layer 2: LLM 3-dimension verification.
            llm = _get_llm()
            prompt = REFLECTION_PROMPT.format(
                query=query,
                available_paper_ids=", ".join(sorted(available_ids)),
                answer=answer,
            )
            response = llm.invoke(prompt)
            result = extract_json(response.content)
            if isinstance(result, dict):
                reflection = ReflectionResult(
                    passed=bool(result.get("passed", False)),
                    citation_ok=bool(result.get("citation_ok", True)),
                    completeness_ok=bool(result.get("completeness_ok", True)),
                    logic_ok=bool(result.get("logic_ok", True)),
                    issues=list(result.get("issues", [])),
                    fix_strategy=result.get("fix_strategy"),
                )
            else:
                reflection = ReflectionResult(
                    passed=False,
                    citation_ok=False,
                    completeness_ok=False,
                    logic_ok=False,
                    issues=["校验输出无法解析，无法确认答案质量。"],
                    fix_strategy="re_generate",
                )
            if reflection["passed"]:
                s.done("通过（引用/完整性/逻辑）", detail=dict(reflection))
            else:
                s.warning(
                    f"未通过，策略={reflection.get('fix_strategy')}",
                    detail=dict(reflection),
                )

    duration = round((time.perf_counter() - t0) * 1000, 2)
    trace = StepTrace(
        node="groundedness_node",
        action="groundedness_check",
        input_summary=f"verifying answer ({len(answer)} chars, {len(available_ids)} papers in context)",
        output_summary=f"passed={reflection['passed']}, strategy={reflection.get('fix_strategy')}",
        duration_ms=duration,
        detail=dict(reflection),
    )

    # Only count failures against the retry budget.
    new_count = state["reflection_count"] if reflection["passed"] else state["reflection_count"] + 1

    return {
        "reflection_result": reflection,
        "reflection_count": new_count,
        "step_traces": state["step_traces"] + [trace],
    }


__all__ = ["groundedness_node"]
