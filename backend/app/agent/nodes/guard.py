"""Guard node: deterministic safety & validity checks before any LLM call.

Maps to the enterprise pipeline's 安全与权限校验 stage. Request-level
protections (rate limit, API key) live in middleware; this node covers
content-level checks on the query itself:

- empty / oversized queries → blocked (cheap DoS & garbage protection)
- prompt-injection heuristics → flagged (not blocked; recorded for
  observability and shown in presentation as a notice)

Deterministic by design: <1ms, no LLM, fully unit-testable.
"""
from __future__ import annotations

import re
import time

from app.agent.stages import stage
from app.agent.state import AgentState, StepTrace

MAX_QUERY_CHARS = 2000

# Conservative, high-precision patterns only — flags, not blocks.
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|system)\s+", re.IGNORECASE),
    re.compile(r"(reveal|print|show|output)\s+(your\s+)?(system\s+prompt|instructions)", re.IGNORECASE),
    re.compile(r"忽略(之前|以上|前面)的?(所有)?(指令|提示|设定)"),
    re.compile(r"(显示|输出|打印)(你的)?(系统提示词|系统指令)"),
]

_REFUSAL_EMPTY = "请输入想要询问的问题。"
_REFUSAL_TOO_LONG = f"问题过长（超过 {MAX_QUERY_CHARS} 字符），请精简后重试。"


def guard_node(state: AgentState, *, query: str) -> dict:
    """Validate the incoming query. Blocked queries short-circuit to presentation."""
    t0 = time.perf_counter()
    with stage("guard") as s:
        flags: list[str] = []
        allowed = True
        reason = "ok"
        refusal = ""

        stripped = (query or "").strip()
        if not stripped:
            allowed, reason, refusal = False, "empty_query", _REFUSAL_EMPTY
        elif len(stripped) > MAX_QUERY_CHARS:
            allowed, reason, refusal = False, "query_too_long", _REFUSAL_TOO_LONG
        else:
            for pat in _INJECTION_PATTERNS:
                if pat.search(stripped):
                    flags.append("possible_prompt_injection")
                    break

        guard_result = {"allowed": allowed, "reason": reason, "flags": flags}
        if allowed:
            if flags:
                s.warning("检测到可疑指令注入特征，已标记", detail=guard_result)
            else:
                s.done("通过", detail=guard_result)
        else:
            s.warning(f"已拦截：{reason}", detail=guard_result)

    duration = round((time.perf_counter() - t0) * 1000, 2)
    trace = StepTrace(
        node="guard_node",
        action="guard",
        input_summary=(stripped[:80] if stripped else "(empty)"),
        output_summary=("passed" + (f", flags={flags}" if flags else "")) if allowed else f"blocked: {reason}",
        duration_ms=duration,
        detail=guard_result,
    )

    out: dict = {
        "guard_result": guard_result,
        "step_traces": state["step_traces"] + [trace],
    }
    if not allowed:
        out["final_answer"] = refusal
    return out


__all__ = ["guard_node", "MAX_QUERY_CHARS"]
