"""LangGraph streaming event helpers.

v2 protocol: every pipeline node self-reports lifecycle ``stage`` events
(stable ids) and the planner publishes ``plan`` events via ``emit``. The
adapter therefore only has two jobs:

1. forward custom events (stage / plan / answer_start / token) as SSE
2. fold ``on_chain_stream`` node-state updates into ``final_state`` so the
   router can persist the result after the stream ends

No node-state diffing, no index reconstruction — the old failure modes are
structurally gone.
"""
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
    """Map LangGraph ``astream_events`` records to the SSE protocol."""
    if event.get("event") == "on_custom_event":
        name = event.get("name", "")
        data = event.get("data", {})
        _update_runtime(name, data, runtime)
        return [{"type": name, "data": data}]

    if event.get("event") != "on_chain_stream" or event.get("name") != "LangGraph":
        return []

    # Fold node-state updates into final_state for post-stream persistence.
    chunk = (event.get("data") or {}).get("chunk") or {}
    for node_state in chunk.values():
        if isinstance(node_state, dict):
            final_state.update(node_state)
    return []


def _update_runtime(name: str, data: Any, runtime: dict) -> None:
    """Track step/reflection counters for the final ``done`` event."""
    if name != "stage" or not isinstance(data, dict):
        return
    status = data.get("status")
    if data.get("stage") == "retrieve_step" and status in ("done", "warning", "failed"):
        runtime["steps_count"] = int(runtime.get("steps_count", 0)) + 1
    if data.get("stage") == "groundedness" and status in ("done", "warning"):
        detail = data.get("detail") or {}
        if detail.get("passed") is False:
            runtime["reflections_count"] = int(runtime.get("reflections_count", 0)) + 1
