"""Sufficiency node: graph-level evidence sufficiency gate.

Maps to the enterprise pipeline's 证据充分性判断 stage. Previously
``evaluate_docs`` ran as an executor plan step with inline plan-injection;
promoting it to a graph node makes the control flow explicit:

    sufficient / parse_failed / budget exhausted → synthesis
    insufficient + budget left                   → re_planner (supplement)

``parse_failed`` deliberately does NOT loop — an unreliable evaluator should
downgrade confidence (presentation layer), not burn retrieval budget.
"""
from __future__ import annotations

import time

from app.agent.nodes.complexity_router import mark_fast_path_escalated
from app.agent.stages import stage
from app.agent.state import AgentState, StepTrace
from app.agent.telemetry import record_fallback
from app.tools.evaluate_docs import evaluate_docs

MAX_SUFFICIENCY_ROUNDS = 1


def sufficiency_node(state: AgentState, *, query: str) -> dict:
    """Evaluate whether the (already filtered) evidence can answer the query."""
    t0 = time.perf_counter()
    with stage("sufficiency") as s:
        context = state.get("retrieval_context") or []
        result = evaluate_docs(query, [d.page_content for d in context])

        sufficient = bool(result.get("sufficient"))
        parse_failed = bool(result.get("parse_failed"))
        round_used = int(state.get("sufficiency_round", 0))

        out: dict = {
            "sufficiency_result": result,
            # mirror kept for the presentation layer & persisted-history compat
            "evaluator_result": result,
        }
        if parse_failed:
            out["evaluator_parse_failed"] = True
            out["degraded"] = True
            out["fallback_telemetry"] = record_fallback(
                state,
                failure_class=str(result.get("failure_class") or "sufficiency_output_unparseable"),
                stage="sufficiency",
                outcome="safe_degraded_answer",
                degraded=True,
            )
            s.warning("充分性评估解析失败，按低可信度继续", detail=result)
        elif sufficient:
            s.done("证据足以支撑回答", detail=result)
        else:
            out["sufficiency_round"] = round_used + 1
            if (
                state.get("execution_path") == "fast_local"
                and not state.get("fast_path_escalated", False)
            ):
                decision = mark_fast_path_escalated(
                    state.get("complexity_decision"),
                    reason_code="evidence_insufficient_escalation",
                )
                out.update(
                    {
                        "execution_path": "fast_escalated",
                        "fast_path_escalated": True,
                        "complexity_decision": decision,
                    }
                )
            if round_used >= MAX_SUFFICIENCY_ROUNDS:
                out["degraded"] = True
                out["fallback_telemetry"] = record_fallback(
                    state,
                    failure_class="supplement_retrieval_budget_exhausted",
                    stage="sufficiency",
                    outcome="safe_degraded_answer",
                    degraded=True,
                )
                s.warning("补充检索后证据仍不足，将注明局限后作答", detail=result)
            else:
                out["fallback_telemetry"] = record_fallback(
                    state,
                    failure_class="evidence_insufficient",
                    stage="sufficiency",
                    outcome="supplement_retrieval",
                    re_retrieve_delta=1,
                )
                missing = result.get("missing_aspects") or []
                s.warning(
                    "证据不足，准备补充检索" + (f"（缺少：{'、'.join(str(m) for m in missing[:2])}）" if missing else ""),
                    detail=result,
                )

    duration = round((time.perf_counter() - t0) * 1000, 2)
    trace = StepTrace(
        node="sufficiency_node",
        action="sufficiency_check",
        input_summary=f"{len(context)} chunks evaluated",
        output_summary=(
            "parse_failed" if parse_failed else f"sufficient={sufficient}"
            + (", degraded" if out.get("degraded") else "")
        ),
        duration_ms=duration,
        detail=result,
    )
    out["step_traces"] = state["step_traces"] + [trace]
    return out


def after_sufficiency(state: AgentState, max_rounds: int = MAX_SUFFICIENCY_ROUNDS) -> str:
    """Conditional edge: supplement retrieval or proceed to synthesis."""
    result = state.get("sufficiency_result") or {}
    if result.get("parse_failed") or result.get("sufficient"):
        return "synthesis"
    if (
        state.get("execution_path") == "fast_escalated"
        and state.get("fast_path_escalated", False)
        and int(state.get("sufficiency_round", 0)) == 1
        and "evidence_insufficient_escalation"
        in list(
            (state.get("complexity_decision") or {}).get("reason_codes") or []
        )
    ):
        return "planner"
    # state.sufficiency_round was already incremented for this failure
    if int(state.get("sufficiency_round", 0)) > max_rounds:
        return "synthesis"
    return "re_planner"


__all__ = ["sufficiency_node", "after_sufficiency", "MAX_SUFFICIENCY_ROUNDS"]
