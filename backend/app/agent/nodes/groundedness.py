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

import time

from langchain_openai import ChatOpenAI

from app.agent.prompts.reflection import REFLECTION_PROMPT
from app.agent.stages import stage
from app.agent.state import AgentState, ReflectionResult, StepTrace
from app.agent.telemetry import classify_failure, record_fallback
from app.core.config import get_settings
from app.observability.llm_usage import invoke_with_usage
from app.utils.citations import extract_arxiv_ids
from app.utils.llm_json import extract_json


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
        return {str(pid).strip() for pid in synthesis_ids if str(pid).strip()}
    # Compatibility fallback for checkpoints created before the synthesis
    # boundary was persisted.
    ids = set()
    for d in state.get("retrieval_context") or []:
        pid = (d.metadata or {}).get("paper_id")
        if pid:
            ids.add(pid)
    return ids


def _precheck_citations(answer: str, available_ids: set[str]) -> list[str]:
    """Return cited ids NOT present in the retrieval context (fabrications)."""
    cited = set(extract_arxiv_ids(answer))
    return sorted(cited - available_ids)


def groundedness_node(state: AgentState, *, query: str) -> dict:
    """Verify the synthesized answer is grounded, complete and consistent."""
    t0 = time.perf_counter()
    telemetry = None
    force_degraded = False
    failure_class = ""
    with stage("groundedness") as s:
        answer = state.get("final_answer") or ""
        available_ids = _context_paper_ids(state)

        if state.get("synthesis_failed"):
            fabricated = []
            force_degraded = True
            failure_class = "synthesis_llm_failure"
            reflection = ReflectionResult(
                passed=False,
                citation_ok=True,
                completeness_ok=False,
                logic_ok=True,
                issues=["生成服务不可用，未产出可验证答案。"],
                fix_strategy=None,
            )
            s.warning("生成阶段已安全降级，跳过额外模型校验", detail=dict(reflection))
        else:
            # Layer 1: deterministic fabricated-citation check.
            fabricated = _precheck_citations(answer, available_ids)
        if not state.get("synthesis_failed") and fabricated:
            failure_class = "citation_outside_synthesis_context"
            reflection = ReflectionResult(
                passed=False,
                citation_ok=False,
                completeness_ok=True,
                logic_ok=True,
                issues=[f"答案引用了检索结果中不存在的论文: {', '.join(fabricated)}"],
                fix_strategy="re_generate",
            )
            s.warning(f"发现 {len(fabricated)} 个虚构引用，要求重新生成", detail=dict(reflection))
        elif not state.get("synthesis_failed"):
            # Layer 2: LLM 3-dimension verification.
            llm = _get_llm()
            settings = get_settings()
            prompt = REFLECTION_PROMPT.format(
                query=query,
                available_paper_ids=", ".join(sorted(available_ids)),
                answer=answer,
            )
            try:
                response = invoke_with_usage(
                    llm,
                    prompt,
                    node="groundedness",
                    model=settings.reflection_model or settings.llm_model,
                    api_base=settings.llm_api_base,
                )
                result = extract_json(response.content)
            except Exception as exc:
                result = None
                force_degraded = True
                failure_class = classify_failure(exc, default="groundedness_llm_failure")
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
                if not failure_class:
                    failure_class = "groundedness_output_unparseable"
                reflection = ReflectionResult(
                    passed=False,
                    citation_ok=False,
                    completeness_ok=False,
                    logic_ok=False,
                    issues=["校验输出无法解析，无法确认答案质量。"],
                    fix_strategy=None if force_degraded else "re_generate",
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

    if not reflection["passed"]:
        max_reflections = get_settings().agent_max_reflections
        strategy = reflection.get("fix_strategy")
        if force_degraded or new_count >= max_reflections:
            force_degraded = True
            telemetry = record_fallback(
                state,
                failure_class=failure_class or "groundedness_retry_budget_exhausted",
                stage="groundedness",
                outcome="safe_degraded_answer",
                degraded=True,
            )
        elif strategy == "re_generate":
            telemetry = record_fallback(
                state,
                failure_class=failure_class or "groundedness_check_failed",
                stage="groundedness",
                outcome="regenerate_answer",
                re_generate_delta=1,
            )
        else:
            telemetry = record_fallback(
                state,
                failure_class=failure_class or "groundedness_check_failed",
                stage="groundedness",
                outcome="supplement_retrieval",
                re_retrieve_delta=1,
            )

    out = {
        "reflection_result": reflection,
        "reflection_count": new_count,
        "step_traces": state["step_traces"] + [trace],
    }
    if telemetry is not None:
        out["fallback_telemetry"] = telemetry
    if force_degraded:
        out["degraded"] = True
    return out


__all__ = ["groundedness_node"]
