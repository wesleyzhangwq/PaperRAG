"""Per-run capture of provider-reported LLM usage.

The collector deliberately never estimates tokens.  A call is ``known`` only
when the provider response contains both input and output token counts.  Cost
attribution is an evaluation concern because it additionally needs an explicit
billing origin and a versioned pricing catalog.
"""
from __future__ import annotations

import re
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator
from urllib.parse import urlparse


def _plain_int(value: Any) -> int | None:
    """Return an exact non-negative integer without numeric coercion.

    Provider usage is a billing input.  Accept native integers and canonical
    decimal integer strings, but reject booleans, floats, exponents, signs,
    whitespace, and other values rather than letting ``int(...)`` truncate or
    reinterpret them.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if not isinstance(value, str) or re.fullmatch(r"(?:0|[1-9][0-9]*)", value) is None:
        return None
    return int(value)


def _coalesced_plain_int(*values: Any) -> tuple[int | None, bool]:
    """Resolve equivalent provider aliases and reject invalid/conflicting data."""
    present = [value for value in values if value is not None]
    if not present:
        return None, False
    parsed = [_plain_int(value) for value in present]
    if any(value is None for value in parsed) or len(set(parsed)) != 1:
        return None, True
    return parsed[0], True


def _plain_text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def infer_provider(api_base: str | None) -> str:
    """Return a stable provider label without retaining endpoint details."""
    host = (urlparse(api_base or "").hostname or "").lower()
    if "minimax" in host:
        return "minimax"
    if "openai" in host:
        return "openai"
    if "anthropic" in host:
        return "anthropic"
    return "openai_compatible" if host else "unknown"


def _usage_mapping(message: Any) -> tuple[dict[str, Any], str]:
    usage = getattr(message, "usage_metadata", None)
    if isinstance(usage, dict) and usage:
        return usage, "usage_metadata"
    metadata = getattr(message, "response_metadata", None)
    if isinstance(metadata, dict):
        for key in ("token_usage", "usage"):
            value = metadata.get(key)
            if isinstance(value, dict) and value:
                return value, f"response_metadata.{key}"
    if isinstance(message, dict):
        for key in ("usage_metadata", "token_usage", "usage"):
            value = message.get(key)
            if isinstance(value, dict) and value:
                return value, key
    return {}, "missing"


def normalize_provider_usage(message: Any) -> dict[str, Any]:
    """Normalize LangChain/OpenAI-compatible provider usage fields."""
    raw, source = _usage_mapping(message)
    input_tokens, _input_present = _coalesced_plain_int(
        raw.get("input_tokens"), raw.get("prompt_tokens")
    )
    output_tokens, _output_present = _coalesced_plain_int(
        raw.get("output_tokens"), raw.get("completion_tokens")
    )
    total_tokens, total_present = _coalesced_plain_int(raw.get("total_tokens"))

    input_details = raw.get("input_token_details") or raw.get("prompt_tokens_details") or {}
    if not isinstance(input_details, dict):
        input_details = {}
    cached_read, _cached_read_present = _coalesced_plain_int(
        input_details.get("cache_read"),
        input_details.get("cached_tokens"),
        raw.get("cache_read_input_tokens"),
    )
    cache_write, _cache_write_present = _coalesced_plain_int(
        input_details.get("cache_creation"),
        input_details.get("cache_write"),
        raw.get("cache_creation_input_tokens"),
    )

    known = input_tokens is not None and output_tokens is not None
    if known:
        expected_total = input_tokens + output_tokens
        if not total_present:
            total_tokens = expected_total
        elif total_tokens != expected_total:
            total_tokens = None
            known = False
    return {
        "usage_status": "known" if known else "unknown",
        "usage_source": source,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_read_tokens": cached_read,
        "cache_write_tokens": cache_write,
    }


def _llm_model(llm: Any, explicit: str | None) -> str:
    if explicit:
        return explicit
    for name in ("model_name", "model"):
        value = getattr(llm, name, None)
        if isinstance(value, str) and value:
            return value
    return "unknown"


@dataclass
class LlmUsageCollector:
    records: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        *,
        node: str,
        llm: Any,
        message: Any,
        duration_ms: float,
        model: str | None,
        api_base: str | None,
    ) -> None:
        normalized = normalize_provider_usage(message)
        self.records.append(
            {
                "call_index": len(self.records) + 1,
                "node": node,
                "provider": infer_provider(api_base),
                "model": _llm_model(llm, model),
                "duration_ms": round(float(duration_ms), 3),
                "outcome": "completed",
                **normalized,
            }
        )

    def record_error(
        self,
        *,
        node: str,
        llm: Any,
        duration_ms: float,
        model: str | None,
        api_base: str | None,
        exc: BaseException,
    ) -> None:
        self.records.append(
            {
                "call_index": len(self.records) + 1,
                "node": node,
                "provider": infer_provider(api_base),
                "model": _llm_model(llm, model),
                "duration_ms": round(float(duration_ms), 3),
                "outcome": "error",
                "error_class": type(exc).__name__,
                "usage_status": "unknown",
                "usage_source": "call_error",
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "cached_read_tokens": None,
                "cache_write_tokens": None,
            }
        )

    def snapshot(self, start: int = 0) -> list[dict[str, Any]]:
        return [dict(record) for record in self.records[start:]]


_ACTIVE_COLLECTOR: ContextVar[LlmUsageCollector | None] = ContextVar(
    "paperrag_llm_usage_collector", default=None
)


def current_collector() -> LlmUsageCollector | None:
    return _ACTIVE_COLLECTOR.get()


@contextmanager
def collect_llm_usage() -> Iterator[LlmUsageCollector]:
    collector = LlmUsageCollector()
    token = _ACTIVE_COLLECTOR.set(collector)
    try:
        yield collector
    finally:
        _ACTIVE_COLLECTOR.reset(token)


def invoke_with_usage(
    llm: Any,
    prompt: Any,
    *,
    node: str,
    model: str | None = None,
    api_base: str | None = None,
) -> Any:
    """Invoke an LLM and attach only provider-returned usage to the active run."""
    started = time.perf_counter()
    collector = current_collector()
    try:
        response = llm.invoke(prompt)
    except Exception as exc:
        if collector is not None:
            collector.record_error(
                node=node,
                llm=llm,
                duration_ms=(time.perf_counter() - started) * 1000,
                model=model,
                api_base=api_base,
                exc=exc,
            )
        raise
    if collector is not None:
        collector.record(
            node=node,
            llm=llm,
            message=response,
            duration_ms=(time.perf_counter() - started) * 1000,
            model=model,
            api_base=api_base,
        )
    return response


def stream_with_usage(
    llm: Any,
    prompt: Any,
    *,
    node: str,
    model: str | None = None,
    api_base: str | None = None,
) -> Iterator[Any]:
    """Yield streaming chunks and record the final provider usage payload."""
    started = time.perf_counter()
    collector = current_collector()
    usage_message: Any = None
    try:
        for chunk in llm.stream(prompt):
            normalized = normalize_provider_usage(chunk)
            if normalized["usage_status"] == "known":
                usage_message = chunk
            yield chunk
    except Exception as exc:
        if collector is not None:
            collector.record_error(
                node=node,
                llm=llm,
                duration_ms=(time.perf_counter() - started) * 1000,
                model=model,
                api_base=api_base,
                exc=exc,
            )
        raise
    if collector is not None:
        collector.record(
            node=node,
            llm=llm,
            message=usage_message,
            duration_ms=(time.perf_counter() - started) * 1000,
            model=model,
            api_base=api_base,
        )


__all__ = [
    "LlmUsageCollector",
    "collect_llm_usage",
    "current_collector",
    "infer_provider",
    "invoke_with_usage",
    "normalize_provider_usage",
    "stream_with_usage",
]
