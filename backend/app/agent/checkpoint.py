"""LangGraph checkpoint helpers."""
from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager, nullcontext
from pathlib import Path
from typing import AsyncIterator, Iterator

import aiosqlite
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.core.config import get_settings


def agent_run_config(thread_id: str, *, recursion_limit: int = 25) -> dict:
    """Build the LangGraph run config shared by sync and streaming paths."""
    return {
        "recursion_limit": recursion_limit,
        "configurable": {"thread_id": thread_id or "default"},
    }


def _resolve_checkpoint_path(path: str | None = None) -> str:
    raw = path or get_settings().agent_checkpoint_path
    if raw == ":memory:":
        return raw
    p = Path(raw)
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


@contextmanager
def open_sync_checkpointer(path: str | None = None) -> Iterator[SqliteSaver | None]:
    settings = get_settings()
    if path is None and not settings.agent_checkpoint_enabled:
        with nullcontext(None) as saver:
            yield saver
        return
    with SqliteSaver.from_conn_string(_resolve_checkpoint_path(path)) as saver:
        yield saver


@asynccontextmanager
async def open_async_checkpointer(path: str | None = None) -> AsyncIterator[AsyncSqliteSaver | None]:
    settings = get_settings()
    if path is None and not settings.agent_checkpoint_enabled:
        yield None
        return
    async with aiosqlite.connect(_resolve_checkpoint_path(path)) as conn:
        if not hasattr(conn, "is_alive"):
            conn.is_alive = conn._thread.is_alive  # type: ignore[attr-defined]
        saver = AsyncSqliteSaver(conn)
        yield saver
