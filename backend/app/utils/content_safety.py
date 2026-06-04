"""Content filters for model output before it reaches product surfaces."""
from __future__ import annotations

import re

_THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think>[\s\S]*$", re.IGNORECASE)
_ESCAPED_THINK_BLOCK_RE = re.compile(r"&lt;think&gt;[\s\S]*?&lt;/think&gt;", re.IGNORECASE)
_ESCAPED_THINK_OPEN_RE = re.compile(r"&lt;think&gt;[\s\S]*$", re.IGNORECASE)


def strip_hidden_reasoning(text: str | None) -> str:
    """Remove hidden reasoning blocks from persisted or streamed answer text."""
    if not text:
        return ""
    cleaned = _THINK_BLOCK_RE.sub("", text)
    cleaned = _THINK_OPEN_RE.sub("", cleaned)
    cleaned = _ESCAPED_THINK_BLOCK_RE.sub("", cleaned)
    cleaned = _ESCAPED_THINK_OPEN_RE.sub("", cleaned)
    return cleaned.lstrip("\n").rstrip()
