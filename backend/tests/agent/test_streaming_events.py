"""Streaming adapter tests for the v2 stage-event protocol."""
import asyncio
import inspect

from langgraph.graph import END, StateGraph

from app.agent.streaming import emit, graph_event_to_sse_events
from app.routers import chat as chat_router


def test_emit_is_noop_outside_langgraph_run():
    emit("token", {"t": "outside"})


def test_emit_surfaces_as_langgraph_custom_event():
    def node(state: dict) -> dict:
        emit("token", {"t": "hello"})
        return {"done": True}

    async def collect() -> list[dict]:
        builder = StateGraph(dict)
        builder.add_node("node", node)
        builder.set_entry_point("node")
        builder.add_edge("node", END)
        graph = builder.compile()
        return [event async for event in graph.astream_events({}, version="v2")]

    events = asyncio.run(collect())

    assert any(
        event["event"] == "on_custom_event"
        and event["name"] == "token"
        and event["data"] == {"t": "hello"}
        for event in events
    )


def test_custom_events_are_forwarded_verbatim():
    raw_event = {
        "event": "on_custom_event",
        "name": "stage",
        "data": {"id": "intent", "stage": "intent", "status": "done", "title": "理解问题"},
    }
    runtime = {"steps_count": 0, "reflections_count": 0}

    events = graph_event_to_sse_events(raw_event, {}, runtime)

    assert events == [{"type": "stage", "data": raw_event["data"]}]


def test_chain_stream_updates_final_state_without_emitting():
    """Node-state diffing is gone: chain_stream chunks only fold into
    final_state for post-stream persistence."""
    raw_event = {
        "event": "on_chain_stream",
        "name": "LangGraph",
        "data": {
            "chunk": {
                "synthesis": {"final_answer": "hello", "step_traces": [{"action": "x"}]},
            }
        },
    }
    final_state: dict = {}
    runtime = {"steps_count": 0, "reflections_count": 0}

    events = graph_event_to_sse_events(raw_event, final_state, runtime)

    assert events == []
    assert final_state["final_answer"] == "hello"


def test_runtime_counts_retrieve_steps_and_failed_groundedness():
    runtime = {"steps_count": 0, "reflections_count": 0}

    graph_event_to_sse_events({
        "event": "on_custom_event",
        "name": "stage",
        "data": {"id": "step:0", "stage": "retrieve_step", "status": "done"},
    }, {}, runtime)
    graph_event_to_sse_events({
        "event": "on_custom_event",
        "name": "stage",
        "data": {"id": "step:1", "stage": "retrieve_step", "status": "start"},
    }, {}, runtime)
    graph_event_to_sse_events({
        "event": "on_custom_event",
        "name": "stage",
        "data": {
            "id": "groundedness", "stage": "groundedness", "status": "warning",
            "detail": {"passed": False},
        },
    }, {}, runtime)

    assert runtime["steps_count"] == 1
    assert runtime["reflections_count"] == 1


def test_chat_stream_uses_langgraph_events_not_thread_queue_bridge():
    source = inspect.getsource(chat_router)

    assert "astream_events" in source
    assert "threading.Thread" not in source
    assert "queue_module.Queue" not in source
    assert "stream_queue_ctx" not in source
