"""Structured fallback telemetry shared by runtime and evaluation."""
from __future__ import annotations

from typing import Any, Mapping


DEFAULT_FALLBACK_TELEMETRY: dict[str, Any] = {
    "fallback_attempted": False,
    "fallback_recovered": False,
    "re_retrieve_count": 0,
    "re_generate_count": 0,
    "degraded_answer": False,
    "terminal_failure": False,
    "failure_class": None,
    "failure_classes": [],
    "events": [],
}


def fallback_telemetry(state: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return an independent, normalized telemetry mapping."""
    raw = (state or {}).get("fallback_telemetry") or {}
    value = {
        **DEFAULT_FALLBACK_TELEMETRY,
        **dict(raw),
    }
    value["failure_classes"] = list(raw.get("failure_classes") or [])
    value["events"] = [dict(item) for item in raw.get("events") or []]
    return value


def classify_failure(exc: BaseException, *, default: str) -> str:
    """Map exceptions to stable, non-sensitive metric labels."""
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if "timeout" in name or "timeout" in text or "timed out" in text:
        return "llm_timeout"
    if any(token in name or token in text for token in ("connection", "connecterror", "network")):
        return "llm_connection_failure"
    return default


def record_fallback(
    state: Mapping[str, Any],
    *,
    failure_class: str,
    stage: str,
    outcome: str,
    attempted: bool = True,
    degraded: bool = False,
    terminal: bool = False,
    re_retrieve_delta: int = 0,
    re_generate_delta: int = 0,
) -> dict[str, Any]:
    telemetry = fallback_telemetry(state)
    telemetry["fallback_attempted"] = bool(telemetry["fallback_attempted"] or attempted)
    telemetry["degraded_answer"] = bool(telemetry["degraded_answer"] or degraded)
    telemetry["terminal_failure"] = bool(telemetry["terminal_failure"] or terminal)
    telemetry["re_retrieve_count"] = int(telemetry["re_retrieve_count"]) + int(re_retrieve_delta)
    telemetry["re_generate_count"] = int(telemetry["re_generate_count"]) + int(re_generate_delta)
    classes = telemetry["failure_classes"]
    if failure_class and failure_class not in classes:
        classes.append(failure_class)
    telemetry["failure_class"] = failure_class or telemetry.get("failure_class")
    telemetry["events"].append(
        {
            "stage": stage,
            "failure_class": failure_class,
            "outcome": outcome,
        }
    )
    return telemetry


def finalize_fallback_telemetry(state: Mapping[str, Any]) -> dict[str, Any]:
    """Compute an operational recovery result at the presentation boundary.

    Offline evaluation tightens ``fallback_recovered`` further by requiring
    strict task success against the dataset gold fields.
    """
    telemetry = fallback_telemetry(state)
    degraded = bool(state.get("degraded") or telemetry["degraded_answer"])
    terminal = bool(telemetry["terminal_failure"])
    reflection = state.get("reflection_result") or {}
    sufficient = state.get("sufficiency_result") or {}
    operationally_sound = bool(
        state.get("final_answer")
        and not degraded
        and not terminal
        and not sufficient.get("parse_failed")
        and reflection.get("passed", True)
    )
    telemetry["degraded_answer"] = degraded
    telemetry["fallback_recovered"] = bool(
        telemetry["fallback_attempted"] and operationally_sound
    )
    if terminal:
        status = "terminal_failure"
    elif degraded:
        status = "safe_degraded"
    elif telemetry["fallback_recovered"]:
        status = "recovered"
    elif telemetry["fallback_attempted"]:
        status = "fallback_incomplete"
    else:
        status = "normal"
    telemetry["outcome"] = status
    return telemetry


__all__ = [
    "DEFAULT_FALLBACK_TELEMETRY",
    "classify_failure",
    "fallback_telemetry",
    "finalize_fallback_telemetry",
    "record_fallback",
]
