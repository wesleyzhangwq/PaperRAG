from __future__ import annotations

from collections import Counter

import pytest

from eval.agentic_compare_metrics import summarize_answer_cases
from eval.costing import load_pricing_catalog
from eval.run_failure_injection_eval import SCENARIOS, run_scenario


@pytest.fixture(scope="module")
def failure_rows() -> dict[str, dict]:
    catalog = load_pricing_catalog()
    return {
        scenario["id"]: run_scenario(scenario, catalog=catalog)
        for scenario in SCENARIOS
    }


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda item: item["id"])
def test_production_graph_failure_scenario_contract(
    scenario: dict, failure_rows: dict[str, dict]
) -> None:
    row = failure_rows[scenario["id"]]

    assert row["scenario_passed"] is True
    assert row["actual_outcome"] == scenario["expected_outcome"]
    assert row["cost_status"] == "unknown"
    assert row["cost_usd"] is None


def test_failure_suite_aggregate_contract(failure_rows: dict[str, dict]) -> None:
    rows = list(failure_rows.values())
    outcomes = Counter(row["actual_outcome"] for row in rows)
    metrics = summarize_answer_cases(rows)

    assert len(rows) == 10
    assert outcomes == {"recovered": 7, "safe_degraded": 2, "terminal_failure": 1}
    assert metrics["task_success_rate"] == 0.7
    assert metrics["fallback_attempted_count"] == 9
    assert metrics["fallback_recovered_count"] == 7
    assert metrics["fallback_recovery_rate"] == pytest.approx(7 / 9, abs=1e-4)
    assert metrics["terminal_failure_rate"] == 0.1
