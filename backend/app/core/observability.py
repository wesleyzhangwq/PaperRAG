"""Structured logging and app-level observability helpers."""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from app.core.context import request_id_ctx


class JsonLogFormatter(logging.Formatter):
    """One JSON object per line; merges `extra={...}` whitelisted keys."""

    _EXTRA_KEYS = frozenset(
        {
            "event",
            "request_id",
            "ms",
            "chunks",
            "phase",
            "error_kind",
            "top_k",
            "final_k",
            "path",
            "status_code",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        rid = getattr(record, "request_id", None) or request_id_ctx.get()
        if rid:
            payload["request_id"] = rid
        for key in self._EXTRA_KEYS:
            if key == "request_id":
                continue
            if hasattr(record, key):
                val = getattr(record, key)
                if val is not None:
                    payload[key] = val
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(*, json_logs: bool) -> None:
    """Attach a single handler to the `app` logger tree (does not reconfigure uvicorn)."""
    app_log = logging.getLogger("app")
    app_log.handlers.clear()
    app_log.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    if json_logs:
        h.setFormatter(JsonLogFormatter())
    else:
        h.setFormatter(
            logging.Formatter("%(levelname)s %(name)s %(message)s")
        )
    app_log.addHandler(h)
    app_log.propagate = False
