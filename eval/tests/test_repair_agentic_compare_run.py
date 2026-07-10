from __future__ import annotations

import pytest


def test_retry_failed_rows_replaces_only_recovered_rows_and_keeps_order() -> None:
    try:
        from eval.scripts.repair_agentic_compare_run import retry_failed_rows
    except ModuleNotFoundError as exc:
        pytest.fail(f"repair runner is not implemented: {exc}")

    rows = [
        {"qid": "q001", "error": None, "answer": "already valid"},
        {"qid": "q002", "error": "APIConnectionError: Connection error.", "answer": ""},
    ]
    calls: list[tuple[str, int]] = []

    def rerun(qid: str, attempt: int) -> dict:
        calls.append((qid, attempt))
        if attempt == 1:
            return {"qid": qid, "error": "APIConnectionError: Connection error."}
        return {"qid": qid, "error": None, "answer": "recovered"}

    updated, repair_rows = retry_failed_rows(
        rows,
        rerun_case=rerun,
        max_attempts=3,
        sleep_seconds=0,
    )

    assert [row["qid"] for row in updated] == ["q001", "q002"]
    assert updated[0] == rows[0]
    assert updated[1]["answer"] == "recovered"
    assert updated[1]["repair_attempts"] == 2
    assert calls == [("q002", 1), ("q002", 2)]
    assert repair_rows == [
        {
            "qid": "q002",
            "attempts": 2,
            "recovered": True,
            "final_error": None,
        }
    ]


def test_recompute_case_rows_applies_updated_abstention_rules() -> None:
    try:
        from eval.scripts.repair_agentic_compare_run import recompute_case_rows
    except ImportError as exc:
        pytest.fail(f"repair runner cannot recompute persisted metrics: {exc}")

    rows = [
        {
            "qid": "n001",
            "type": "negative",
            "difficulty": "easy",
            "expected_pids": [],
            "expected_mode": "insufficient",
            "answer": "参考资料不足以回答该问题。",
            "answer_for_metrics": "参考资料不足以回答该问题。",
            "source_pids": [],
            "latency_s": 1.0,
            "used_chunks": 0,
            "step_count": 2,
            "retrieval_step_count": 1,
            "mode_correct": False,
        }
    ]

    updated = recompute_case_rows(rows)

    assert updated[0]["mode_correct"] is True
    assert updated[0]["answer_abstained"] is True
