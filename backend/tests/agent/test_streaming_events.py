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


def test_graph_update_event_maps_to_existing_sse_protocol():
    raw_event = {
        "event": "on_chain_stream",
        "name": "LangGraph",
        "data": {
            "chunk": {
                "intent": {
                    "intent": {"type": "simple", "entities": [], "complexity": "low"},
                    "step_traces": [],
                }
            }
        },
    }

    final_state: dict = {"step_traces": []}
    runtime = {"prev_traces_len": 0, "steps_count": 0, "reflections_count": 0}

    events = graph_event_to_sse_events(raw_event, final_state, runtime)

    assert events == [
        {
            "type": "intent",
            "data": {"type": "simple", "entities": [], "complexity": "low"},
        }
    ]


def test_chat_stream_uses_langgraph_events_not_thread_queue_bridge():
    source = inspect.getsource(chat_router)

    assert "astream_events" in source
    assert "threading.Thread" not in source
    assert "queue_module.Queue" not in source
    assert "stream_queue_ctx" not in source
