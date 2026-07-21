"""Versioned official-pricing attribution for provider-reported LLM usage."""
from __future__ import annotations

import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any


DEFAULT_CATALOG_PATH = (
    Path(__file__).resolve().parent / "pricing" / "llm-pricing-2026-07-21.json"
)


def load_pricing_catalog(path: str | Path = DEFAULT_CATALOG_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value))


def _usd(value: Decimal) -> float:
    return float(value.quantize(Decimal("0.000000001")))


def _exact_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str) and re.fullmatch(r"(?:0|[1-9][0-9]*)", value):
        return int(value)
    return None


def _unknown_call(record: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        **dict(record),
        "cost_status": "unknown",
        "cost_usd": None,
        "cost_unknown_reason": reason,
    }


def attribute_llm_costs(
    records: list[dict[str, Any]],
    *,
    billing_origin: str,
    catalog: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attribute pay-as-you-go list cost without estimating missing usage.

    ``billing_origin`` is deliberately explicit.  A MiniMax-compatible host or
    API-key presence cannot prove that calls are billed via official paygo
    rather than a token plan, proxy, or custom contract.
    """
    catalog = catalog or load_pricing_catalog()
    expected_origin = str(catalog.get("billing_origin") or "")
    expected_provider = str(catalog.get("provider") or "")
    models = catalog.get("models") or {}
    calls: list[dict[str, Any]] = []
    known_total = Decimal("0")
    input_total = 0
    output_total = 0
    cached_read_total = 0
    cache_write_total = 0
    usage_unknown = 0
    cached_read_unknown = 0
    cache_write_unknown = 0
    cached_read_known_calls = 0
    cache_write_known_calls = 0

    for raw in records:
        record = dict(raw)
        input_tokens = _exact_nonnegative_int(record.get("input_tokens"))
        output_tokens = _exact_nonnegative_int(record.get("output_tokens"))
        if record.get("usage_status") != "known":
            calls.append(_unknown_call(record, "provider_usage_missing"))
            usage_unknown += 1
            continue
        if input_tokens is None or output_tokens is None:
            calls.append(_unknown_call(record, "provider_usage_invalid"))
            usage_unknown += 1
            continue
        cached_read_raw = record.get("cached_read_tokens")
        cache_write_raw = record.get("cache_write_tokens")
        cached_read_value = _exact_nonnegative_int(cached_read_raw)
        cache_write_value = _exact_nonnegative_int(cache_write_raw)
        cached_read_known = cached_read_value is not None
        cache_write_known = cache_write_value is not None
        cached_read = cached_read_value or 0
        cache_write = cache_write_value or 0
        input_total += input_tokens
        output_total += output_tokens
        if cached_read_known:
            cached_read_total += cached_read
            cached_read_known_calls += 1
        else:
            cached_read_unknown += 1
        if cache_write_known:
            cache_write_total += cache_write
            cache_write_known_calls += 1
        else:
            cache_write_unknown += 1

        if billing_origin != expected_origin:
            calls.append(_unknown_call(record, "billing_origin_unverified"))
            continue
        if record.get("provider") != expected_provider:
            calls.append(_unknown_call(record, "provider_not_in_catalog"))
            continue
        prices = models.get(record.get("model"))
        if not isinstance(prices, dict):
            calls.append(_unknown_call(record, "model_not_in_catalog"))
            continue
        if not cached_read_known or not cache_write_known:
            # The official catalog prices both cache dimensions separately.
            # A missing provider dimension is not evidence of zero usage.
            calls.append(_unknown_call(record, "cache_usage_missing"))
            continue
        if cached_read + cache_write > input_tokens:
            calls.append(_unknown_call(record, "cache_token_semantics_ambiguous"))
            continue

        standard_input = input_tokens - cached_read - cache_write
        cost = (
            _decimal(standard_input) * _decimal(prices["input"])
            + _decimal(output_tokens) * _decimal(prices["output"])
            + _decimal(cached_read) * _decimal(prices["cache_read"])
            + _decimal(cache_write) * _decimal(prices["cache_write"])
        ) / Decimal("1000000")
        known_total += cost
        calls.append(
            {
                **record,
                "standard_input_tokens": standard_input,
                "cost_status": "known",
                "cost_usd": _usd(cost),
                "cost_unknown_reason": None,
            }
        )

    unknown_cost_calls = sum(1 for call in calls if call["cost_status"] == "unknown")
    known_cost_calls = len(calls) - unknown_cost_calls
    all_known = unknown_cost_calls == 0
    return {
        "provider_models": sorted(
            {
                f"{record.get('provider')}:{record.get('model')}"
                for record in records
                if record.get("provider") or record.get("model")
            }
        ),
        "input_tokens": input_total if usage_unknown == 0 else None,
        "output_tokens": output_total if usage_unknown == 0 else None,
        "cached_read_tokens": (
            cached_read_total
            if usage_unknown == 0 and cached_read_unknown == 0
            else None
        ),
        "cache_write_tokens": (
            cache_write_total
            if usage_unknown == 0 and cache_write_unknown == 0
            else None
        ),
        "total_tokens": input_total + output_total if usage_unknown == 0 else None,
        "known_partial_input_tokens": input_total,
        "known_partial_output_tokens": output_total,
        "known_partial_cached_read_tokens": (
            cached_read_total if cached_read_known_calls else None
        ),
        "known_partial_cache_write_tokens": (
            cache_write_total if cache_write_known_calls else None
        ),
        "usage_status": "known" if usage_unknown == 0 else "unknown",
        "usage_unknown_call_count": usage_unknown,
        "cache_usage_status": (
            "known"
            if usage_unknown == 0
            and cached_read_unknown == 0
            and cache_write_unknown == 0
            else "unknown"
        ),
        "cached_read_unknown_call_count": cached_read_unknown,
        "cache_write_unknown_call_count": cache_write_unknown,
        "cost_usd": _usd(known_total) if all_known else None,
        "known_cost_call_count": known_cost_calls,
        "known_partial_cost_usd": _usd(known_total) if known_cost_calls else None,
        "cost_status": "known" if all_known else "unknown",
        "cost_unknown_call_count": unknown_cost_calls,
        "pricing_catalog_version": catalog.get("catalog_version"),
        "pricing_source": (catalog.get("source") or {}).get("url"),
        "billing_origin": billing_origin,
        "cost_scope": "llm_only",
        "calls": calls,
    }


__all__ = ["DEFAULT_CATALOG_PATH", "attribute_llm_costs", "load_pricing_catalog"]
