from __future__ import annotations

import pytest

from eval.rag_metrics import (
    compare_metric,
    evaluate_retrieval_case,
    render_markdown_report,
    summarize_retrieval_cases,
)


def test_evaluate_retrieval_case_scores_unique_ranking_and_context_noise() -> None:
    case = evaluate_retrieval_case(
        qid="x001",
        query="compare two papers",
        expected_paper_ids=["A", "B"],
        expected_mode="answer",
        difficulty="hard",
        qtype="comparison",
        retrieved_chunks=[
            {"paper_id": "A", "score": 0.91},
            {"paper_id": "A", "score": 0.89},
            {"paper_id": "C", "score": 0.81},
            {"paper_id": "B", "score": 0.72},
        ],
        k_values=[1, 3],
        context_k=4,
        latency_s=0.42,
    )

    assert case["ranked_pids"] == ["A", "C", "B"]
    assert case["hit_at_1"] == 1.0
    assert case["recall_at_1"] == 0.5
    assert case["recall_at_3"] == 1.0
    assert case["precision_at_3"] == pytest.approx(2 / 3)
    assert case["ndcg_at_3"] == pytest.approx(0.9197, abs=0.0001)
    assert case["mrr"] == 1.0
    assert case["first_relevant_rank"] == 1
    assert case["context_chunk_precision"] == 0.75
    assert case["context_recall"] == 1.0
    assert case["context_noise_rate"] == 0.25


def test_negative_retrieval_case_records_exposure_without_polluting_positive_metrics() -> None:
    case = evaluate_retrieval_case(
        qid="n001",
        query="out of corpus question",
        expected_paper_ids=[],
        expected_mode="insufficient",
        difficulty="easy",
        qtype="negative",
        retrieved_chunks=[
            {"paper_id": "Z", "score": 0.77},
            {"paper_id": "Y", "score": 0.61},
        ],
        k_values=[1, 3],
        context_k=2,
        latency_s=0.2,
    )

    assert case["has_expected"] is False
    assert case["is_negative"] is True
    assert case["negative_context_count"] == 2
    assert case["negative_max_score"] == 0.77
    assert "ndcg_at_3" not in case

    summary = summarize_retrieval_cases([case], k_values=[1, 3])
    assert summary["count"] == 1
    assert summary["count_positive"] == 0
    assert summary["count_negative"] == 1
    assert summary["negative_with_context_rate"] == 1.0
    assert summary["ndcg_at_3"] is None


def test_summarize_retrieval_cases_breaks_down_quality_and_latency() -> None:
    cases = [
        evaluate_retrieval_case(
            qid="c001",
            query="single paper",
            expected_paper_ids=["A"],
            expected_mode="answer",
            difficulty="easy",
            qtype="concept_locate",
            retrieved_chunks=[{"paper_id": "A", "score": 0.9}],
            k_values=[1, 3],
            context_k=3,
            latency_s=0.1,
        ),
        evaluate_retrieval_case(
            qid="x001",
            query="two papers",
            expected_paper_ids=["B", "C"],
            expected_mode="answer",
            difficulty="hard",
            qtype="comparison",
            retrieved_chunks=[
                {"paper_id": "B", "score": 0.8},
                {"paper_id": "D", "score": 0.7},
            ],
            k_values=[1, 3],
            context_k=3,
            latency_s=0.3,
        ),
    ]

    summary = summarize_retrieval_cases(cases, k_values=[1, 3])

    assert summary["count_positive"] == 2
    assert summary["hit_at_1"] == 1.0
    assert summary["recall_at_3"] == 0.75
    assert summary["context_recall"] == 0.75
    assert summary["latency_p90"] == 0.3
    assert summary["by_difficulty"]["hard"]["recall_at_3"] == 0.5
    assert summary["by_type"]["concept_locate"]["ndcg_at_3"] == 1.0


def test_compare_metric_and_markdown_report_are_resume_safe() -> None:
    comparison = compare_metric("ndcg_at_5", baseline=0.333, candidate=0.35)
    assert comparison == {
        "metric": "ndcg_at_5",
        "baseline": 0.333,
        "candidate": 0.35,
        "absolute_delta": 0.017,
        "relative_delta_pct": 5.11,
    }

    report = render_markdown_report(
        run_id="rag-v1",
        dataset_name="questions_v2.jsonl",
        summary={
            "count": 65,
            "count_positive": 55,
            "count_negative": 10,
            "ndcg_at_5": 0.35,
            "recall_at_5": 0.341,
            "mrr": 0.385,
            "context_chunk_precision": 0.62,
            "context_recall": 0.49,
            "negative_with_context_rate": 1.0,
            "negative_max_score_mean": 0.73,
            "generation_count": 5,
            "mode_accuracy": 0.8,
            "citation_support_rate": 1.0,
            "latency_p90": 0.61,
        },
        comparisons=[comparison],
    )

    assert "Resume-safe metrics" in report
    assert "NDCG@5: 0.3500" in report
    assert "Negative with context rate: 1.0000" in report
    assert "Generation cases: 5" in report
    assert "Mode accuracy: 0.8000" in report
    assert "ndcg_at_5: 0.3330 -> 0.3500 (+5.11%)" in report
