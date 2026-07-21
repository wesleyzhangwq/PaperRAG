from __future__ import annotations

import pytest

from eval.scripts.compare_rag_runs import compare_metric_rows, compare_runs


def test_compare_metric_rows_reports_paired_delta_and_outcomes() -> None:
    baseline = [
        {"qid": "q1", "ndcg_at_5": 0.0},
        {"qid": "q2", "ndcg_at_5": 0.5},
        {"qid": "q3", "ndcg_at_5": 1.0},
    ]
    candidate = [
        {"qid": "q1", "ndcg_at_5": 0.5},
        {"qid": "q2", "ndcg_at_5": 0.5},
        {"qid": "q3", "ndcg_at_5": 0.5},
    ]

    result = compare_metric_rows(
        baseline,
        candidate,
        metric="ndcg_at_5",
        bootstrap_samples=500,
        seed=7,
    )

    assert result["count"] == 3
    assert result["mean_delta"] == 0.0
    assert result["wins"] == 1
    assert result["ties"] == 1
    assert result["losses"] == 1
    assert result["ci95_low"] <= 0.0 <= result["ci95_high"]


def test_compare_runs_rejects_mismatched_question_ids() -> None:
    with pytest.raises(ValueError, match="Question id mismatch"):
        compare_runs(
            [{"qid": "q1", "ndcg_at_5": 0.1}],
            [{"qid": "q2", "ndcg_at_5": 0.2}],
            metrics=["ndcg_at_5"],
            bootstrap_samples=10,
            seed=1,
        )


def test_latency_treats_lower_candidate_value_as_a_win() -> None:
    result = compare_metric_rows(
        [{"qid": "q1", "latency_s": 1.0}],
        [{"qid": "q1", "latency_s": 0.5}],
        metric="latency_s",
        bootstrap_samples=10,
        seed=1,
    )

    assert result["mean_delta"] == -0.5
    assert result["wins"] == 1
    assert result["losses"] == 0
