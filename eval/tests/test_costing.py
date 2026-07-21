from eval.costing import attribute_llm_costs, load_pricing_catalog


def _usage(**overrides):
    row = {
        "node": "planner",
        "provider": "minimax",
        "model": "MiniMax-M2.7",
        "usage_status": "known",
        "input_tokens": 1000,
        "output_tokens": 100,
        "cached_read_tokens": 200,
        "cache_write_tokens": 100,
    }
    row.update(overrides)
    return row


def test_costing_uses_exact_official_paygo_model_and_cache_prices() -> None:
    result = attribute_llm_costs(
        [_usage()],
        billing_origin="minimax_paygo",
        catalog=load_pricing_catalog(),
    )

    # 700 standard input * .3/M + 100 output * 1.2/M
    # + 200 cache-read * .06/M + 100 cache-write * .375/M.
    assert result["cost_status"] == "known"
    assert result["cost_usd"] == 0.0003795
    assert result["cached_read_tokens"] == 200
    assert result["cache_write_tokens"] == 100


def test_costing_is_unknown_for_unverified_origin_or_exact_model() -> None:
    origin_unknown = attribute_llm_costs([_usage()], billing_origin="unknown")
    model_unknown = attribute_llm_costs(
        [_usage(model="MiniMax-M3")], billing_origin="minimax_paygo"
    )

    assert origin_unknown["cost_status"] == "unknown"
    assert origin_unknown["cost_usd"] is None
    assert model_unknown["cost_status"] == "unknown"
    assert model_unknown["calls"][0]["cost_unknown_reason"] == "model_not_in_catalog"


def test_partial_missing_usage_nulls_task_totals_but_keeps_labeled_partial() -> None:
    result = attribute_llm_costs(
        [_usage(), _usage(usage_status="unknown", input_tokens=None, output_tokens=None)],
        billing_origin="minimax_paygo",
    )

    assert result["usage_status"] == "unknown"
    assert result["input_tokens"] is None
    assert result["output_tokens"] is None
    assert result["total_tokens"] is None
    assert result["known_partial_input_tokens"] == 1000
    assert result["cost_status"] == "unknown"


def test_missing_cache_dimension_is_unknown_not_assumed_zero() -> None:
    result = attribute_llm_costs(
        [_usage(cache_write_tokens=None)],
        billing_origin="minimax_paygo",
    )

    assert result["usage_status"] == "known"
    assert result["input_tokens"] == 1000
    assert result["cached_read_tokens"] == 200
    assert result["cache_write_tokens"] is None
    assert result["known_partial_cache_write_tokens"] is None
    assert result["cache_usage_status"] == "unknown"
    assert result["cost_status"] == "unknown"
    assert result["cost_usd"] is None
    assert result["calls"][0]["cost_unknown_reason"] == "cache_usage_missing"


def test_costing_rejects_fractional_or_boolean_usage_without_truncation() -> None:
    for invalid in (1.9, True, "1.9", " 1"):
        result = attribute_llm_costs(
            [_usage(input_tokens=invalid)],
            billing_origin="minimax_paygo",
        )

        assert result["usage_status"] == "unknown"
        assert result["cost_status"] == "unknown"
        assert result["cost_usd"] is None
        assert result["calls"][0]["cost_unknown_reason"] == "provider_usage_invalid"


def test_costing_rejects_fractional_cache_usage_as_unknown() -> None:
    result = attribute_llm_costs(
        [_usage(cached_read_tokens=2.5)],
        billing_origin="minimax_paygo",
    )

    assert result["usage_status"] == "known"
    assert result["cache_usage_status"] == "unknown"
    assert result["cost_status"] == "unknown"
    assert result["calls"][0]["cost_unknown_reason"] == "cache_usage_missing"
