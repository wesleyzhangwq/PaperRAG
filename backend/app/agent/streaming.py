"""Streaming context: per-request callback queue for real-time SSE.

Used by synthesis_node to push tokens (and other live events) directly to the
SSE generator without waiting for the node to finish.
"""
from __future__ import annotations

from contextvars import ContextVar
from queue import Queue
from typing import Any, Optional

# Per-request event queue. None when not inside a streaming request.
stream_queue_ctx: ContextVar[Optional[Queue]] = ContextVar("stream_queue", default=None)


def emit(event_type: str, data: Any) -> None:
    """Push an SSE event onto the current request's stream queue.

    No-op when there is no queue bound (e.g. sync /chat endpoint or tests).
    """
    q = stream_queue_ctx.get()
    if q is None:
        return
    try:
        q.put_nowait({"type": event_type, "data": data})
    except Exception:
        pass
