"""LangGraph streaming event helpers."""
from __future__ import annotations

import json
from typing import Any

from langchain_core.callbacks.manager import dispatch_custom_event


def emit(event_type: str, data: Any) -> None:
    """Emit a custom event visible to ``graph.astream_events``.

    Outside a LangGraph/LangChain run there is no parent run id, so this becomes
    a no-op. That preserves sync endpoint and unit-test behavior.
    """
    try:
        dispatch_custom_event(event_type, data)
    except RuntimeError:
        return


def encode_sse(event_type: str, data: Any) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def graph_event_to_sse_events(
    event: dict,
    final_state: dict,
    runtime: dict,
) -> list[dict]:
    """Map LangGraph ``astream_events`` records to the existing SSE protocol."""
    if event.get("event") == "on_custom_event":
        return [{"type": event.get("name", ""), "data": event.get("data", {})}]

    if event.get("event") != "on_chain_stream" or event.get("name") != "LangGraph":
        return []

    chunk = (event.get("data") or {}).get("chunk") or {}
    out: list[dict] = []
    for node_name, node_state in chunk.items():
        if not isinstance(node_state, dict):
            continue
        final_state.update(node_state)

        if node_name == "executor":
            plan = final_state.get("plan", [])
            step_idx = node_state.get("plan_step_index", 0) - 1
            if 0 <= step_idx < len(plan):
                step_spec = plan[step_idx]
                out.append({
                    "type": "step_start",
                    "data": {
                        "index": step_idx,
                        "action": step_spec.get("action", ""),
                        "reason": step_spec.get("reason", ""),
                    },
                })

        traces = node_state.get("step_traces", [])
        prev_traces_len = int(runtime.get("prev_traces_len", 0))
        for trace in traces[prev_traces_len:]:
            out.append({"type": "step_done", "data": trace})
            runtime["steps_count"] = int(runtime.get("steps_count", 0)) + 1
        runtime["prev_traces_len"] = len(final_state.get("step_traces", []))

        if node_name == "intent" and node_state.get("intent"):
            out.append({"type": "intent", "data": node_state["intent"]})

        if node_name == "planner" and node_state.get("plan"):
            out.append({
                "type": "plan",
                "data": {
                    "steps": node_state["plan"],
                    "total_steps": len(node_state["plan"]),
                },
            })

        if node_name == "reflection" and node_state.get("reflection_result"):
            runtime["reflections_count"] = int(runtime.get("reflections_count", 0)) + 1
            out.append({"type": "reflection", "data": node_state["reflection_result"]})

        if node_name == "re_planner" and node_state.get("plan"):
            out.append({
                "type": "re_plan",
                "data": {"new_steps": node_state["plan"][-3:]},
            })

    return out
