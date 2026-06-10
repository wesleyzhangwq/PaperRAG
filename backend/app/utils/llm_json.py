"""Tolerant JSON extraction from LLM output.

Reasoning models (MiniMax-M2.7, DeepSeek-R1, …) may prefix output with
<think>…</think> blocks or wrap JSON in code fences. Every node that parses
LLM JSON must go through this helper — a bare ``json.loads`` silently degrades
the pipeline (plan parse failure ⇒ single-step fallback retrieval).
"""
from __future__ import annotations

import json
import re
from typing import Any

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_OPEN_THINK_RE = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.DOTALL)


def strip_think(text: str) -> str:
    """Remove closed (and trailing unterminated) <think> blocks."""
    text = _THINK_RE.sub("", text or "")
    return _OPEN_THINK_RE.sub("", text).strip()


def extract_json(text: str) -> Any | None:
    """Best-effort JSON extraction: strips think blocks and code fences, then
    falls back to the first balanced {...} or [...] span. Returns None when
    nothing parses."""
    if not text:
        return None
    text = strip_think(str(text)).strip()
    if text.startswith("```"):
        text = _FENCE_RE.sub("", text).strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    for pattern in (r"\[[\s\S]*\]", r"\{[\s\S]*\}"):
        m = re.search(pattern, text)
        if m:
            try:
                return json.loads(m.group(0))
            except (json.JSONDecodeError, TypeError):
                continue
    return None


__all__ = ["extract_json", "strip_think"]
