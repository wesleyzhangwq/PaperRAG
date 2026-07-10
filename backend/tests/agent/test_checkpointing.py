import tempfile
import asyncio
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, StateGraph

from app.agent.checkpoint import agent_run_config, open_async_checkpointer, open_sync_checkpointer


class CounterState(TypedDict):
    count: int


def test_agent_run_config_uses_conversation_id_as_thread_id():
    config = agent_run_config("conv-123")

    assert config["recursion_limit"] == 60
    assert config["configurable"]["thread_id"] == "conv-123"


def test_sqlite_checkpointer_persists_checkpoint_between_instances():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "checkpoints.sqlite")

        builder = StateGraph(CounterState)
        builder.add_node("inc", lambda state: {"count": state["count"] + 1})
        builder.set_entry_point("inc")
        builder.add_edge("inc", END)

        config = agent_run_config("thread-a")
        with open_sync_checkpointer(db_path) as saver:
            graph = builder.compile(checkpointer=saver)
            assert graph.invoke({"count": 0}, config=config)["count"] == 1

        with open_sync_checkpointer(db_path) as saver:
            checkpoint = saver.get_tuple(config)

        assert checkpoint is not None
        assert checkpoint.config["configurable"]["thread_id"] == "thread-a"


def test_async_sqlite_checkpointer_supports_astream_events():
    async def run() -> list[dict]:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "checkpoints.sqlite")

            builder = StateGraph(CounterState)
            builder.add_node("inc", lambda state: {"count": state["count"] + 1})
            builder.set_entry_point("inc")
            builder.add_edge("inc", END)

            async with open_async_checkpointer(db_path) as saver:
                graph = builder.compile(checkpointer=saver)
                return [
                    event
                    async for event in graph.astream_events(
                        {"count": 0},
                        config=agent_run_config("thread-async"),
                        version="v2",
                    )
                ]

    events = asyncio.run(run())

    assert any(event.get("event") == "on_chain_stream" for event in events)
