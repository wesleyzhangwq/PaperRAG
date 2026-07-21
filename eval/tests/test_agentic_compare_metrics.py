from __future__ import annotations

import pytest

from eval.agentic_compare_metrics import (
    answer_case_metrics,
    compare_system_summaries,
    detect_abstention,
    select_proportional_questions,
    select_stratified_questions,
    strip_thinking_for_metrics,
    summarize_answer_cases,
)


def test_answer_case_metrics_scores_sources_citations_and_mode() -> None:
    row = answer_case_metrics(
        qid="x001",
        qtype="comparison",
        difficulty="hard",
        expected_paper_ids=["2604.14920", "2604.15184"],
        expected_mode="answer",
        answer="该结论来自 [arxiv:2604.14920] 和 [arxiv:2604.15309]。",
        source_pids=["2604.14920", "2604.15309", "2604.15143"],
        latency_s=1.2,
        used_chunks=3,
        step_count=8,
        retrieval_step_count=2,
    )

    assert row["mode_correct"] is True
    assert row["source_hit"] == 1.0
    assert row["source_recall"] == 0.5
    assert row["source_precision"] == pytest.approx(1 / 3)
    assert row["citation_pids"] == ["2604.14920", "2604.15309"]
    assert row["citation_support_rate"] == 1.0
    assert row["citation_precision"] == 0.5
    assert row["citation_expected_hit"] == 1.0
    assert row["latency_s"] == 1.2
    assert row["used_chunks"] == 3
    assert row["step_count"] == 8
    assert row["retrieval_step_count"] == 2


def test_answer_case_metrics_uses_final_context_for_citation_support() -> None:
    row = answer_case_metrics(
        qid="x002",
        qtype="comparison",
        difficulty="hard",
        expected_paper_ids=["p1", "p2"],
        expected_mode="answer",
        answer="Evidence [arxiv:1111.1111] and [arxiv:2222.2222].",
        source_pids=["p1", "p2", "p3"],
        context_pids=["1111.1111"],
    )

    assert row["source_pids"] == ["p1", "p2", "p3"]
    assert row["context_pids"] == ["1111.1111"]
    assert row["citation_support_rate"] == 0.5


def test_answer_case_metrics_handles_negative_abstention() -> None:
    row = answer_case_metrics(
        qid="n001",
        qtype="negative",
        difficulty="easy",
        expected_paper_ids=[],
        expected_mode="insufficient",
        answer="信息不足，无法回答该问题。",
        source_pids=["Z"],
    )

    assert row["mode_correct"] is True
    assert row["has_expected"] is False
    assert row["source_hit"] is None
    assert row["citation_support_rate"] is None


def test_answer_case_metrics_marks_execution_errors_incorrect() -> None:
    row = answer_case_metrics(
        qid="c001",
        qtype="concept_locate",
        difficulty="easy",
        expected_paper_ids=["2604.14920"],
        expected_mode="answer",
        answer="",
        source_pids=[],
        error="APIConnectionError: Connection error.",
    )

    assert row["mode_correct"] is False


def test_strict_task_success_uses_structured_positive_contract() -> None:
    row = answer_case_metrics(
        qid="p-strict",
        qtype="concept_locate",
        difficulty="easy",
        expected_paper_ids=["1706.03762"],
        expected_mode="answer",
        answer="Supported [arxiv:1706.03762]",
        source_pids=["1706.03762", "1810.04805"],
        context_pids=["1706.03762"],
        final_source_pids=["1706.03762"],
        presentation={"response_mode": "answer", "confidence": "high"},
        degraded_answer=False,
        terminal_failure=False,
    )

    assert row["mode_signal"] == "structured"
    assert row["expected_source_hit"] is True
    assert row["citations_within_final_context"] is True
    assert row["task_success"] is True


def test_strict_task_success_rejects_degraded_and_residual_illegal_citations() -> None:
    degraded = answer_case_metrics(
        qid="p-degraded",
        qtype="concept_locate",
        difficulty="easy",
        expected_paper_ids=["1706.03762"],
        expected_mode="answer",
        answer="Supported [arxiv:1706.03762]",
        source_pids=["1706.03762"],
        context_pids=["1706.03762"],
        final_source_pids=["1706.03762"],
        presentation={"response_mode": "answer"},
        degraded_answer=True,
    )
    illegal = answer_case_metrics(
        qid="p-illegal",
        qtype="concept_locate",
        difficulty="easy",
        expected_paper_ids=["1706.03762"],
        expected_mode="answer",
        answer="Supported [arxiv:1706.03762] plus https://arxiv.org/abs/1810.04805",
        source_pids=["1706.03762", "1810.04805"],
        context_pids=["1706.03762"],
        final_source_pids=["1706.03762"],
        presentation={"response_mode": "answer"},
        removed_citation_pids=["1810.04805"],
    )

    assert degraded["task_success"] is False
    assert illegal["illegal_citations_remaining"] == ["1810.04805"]
    assert illegal["task_success"] is False


def test_strict_negative_success_requires_non_degraded_structured_abstention() -> None:
    common = dict(
        qid="n-strict",
        qtype="negative",
        difficulty="easy",
        expected_paper_ids=["gold-metadata-must-not-make-positive"],
        expected_mode="insufficient",
        answer="No evidence.",
        source_pids=[],
        context_pids=[],
        final_source_pids=[],
        presentation={"response_mode": "insufficient", "confidence": "low"},
    )
    normal = answer_case_metrics(**common, degraded_answer=False)
    degraded = answer_case_metrics(**common, degraded_answer=True)

    assert normal["is_negative"] is True
    assert normal["task_success"] is True
    assert degraded["task_success"] is False

    summary = summarize_answer_cases([normal, degraded])
    assert summary["count_positive"] == 0
    assert summary["count_negative"] == 2
    assert summary["expected_source_hit_rate"] is None


def test_strip_thinking_for_metrics_removes_reasoning_block() -> None:
    answer = "<think>private reasoning [arxiv:2604.00000]</think>\n\n最终答案 [arxiv:2604.14920]"

    assert strip_thinking_for_metrics(answer) == "最终答案 [arxiv:2604.14920]"


def test_detect_abstention_recognizes_grounded_chinese_refusal_variants() -> None:
    assert detect_abstention("参考资料不足以回答该问题。") is True
    assert detect_abstention("检索到的文献均不涉及该主题，无法基于现有语料回答。") is True
    assert detect_abstention("论文未提供具体阈值，但摘要足以说明其方法。") is False


def test_summarize_answer_cases_groups_by_type_and_latency() -> None:
    rows = [
        answer_case_metrics(
            qid="c001",
            qtype="concept_locate",
            difficulty="easy",
            expected_paper_ids=["2604.14920"],
            expected_mode="answer",
            answer="[arxiv:2604.14920]",
            source_pids=["2604.14920"],
            latency_s=0.2,
        ),
        answer_case_metrics(
            qid="n001",
            qtype="negative",
            difficulty="easy",
            expected_paper_ids=[],
            expected_mode="insufficient",
            answer="无法回答。",
            source_pids=[],
            latency_s=0.4,
        ),
    ]

    summary = summarize_answer_cases(rows)

    assert summary["count"] == 2
    assert summary["count_positive"] == 1
    assert summary["count_negative"] == 1
    assert summary["mode_accuracy"] == 1.0
    assert summary["source_recall"] == 1.0
    assert summary["citation_precision"] == 1.0
    assert summary["latency_p90"] == 0.4
    assert summary["latency_p95"] == 0.4
    assert summary["latency_n"] == 2
    assert summary["by_type"]["concept_locate"]["source_recall"] == 1.0
    assert summary["by_type"]["negative"]["count_negative"] == 1


def test_compare_system_summaries_reports_absolute_delta() -> None:
    rows = compare_system_summaries(
        baseline={"mode_accuracy": 0.6, "source_recall": 0.4},
        candidate={"mode_accuracy": 0.75, "source_recall": 0.5},
    )

    assert rows[0] == {
        "metric": "mode_accuracy",
        "baseline": 0.6,
        "candidate": 0.75,
        "absolute_delta": 0.15,
        "relative_delta_pct": 25.0,
    }


def test_select_stratified_questions_keeps_type_balance() -> None:
    questions = [
        {"qid": f"c{i}", "type": "concept_locate"}
        for i in range(5)
    ] + [
        {"qid": f"x{i}", "type": "comparison"}
        for i in range(5)
    ]

    selected = select_stratified_questions(questions, per_type=2)

    assert [q["qid"] for q in selected] == ["c0", "c1", "x0", "x1"]


def test_select_proportional_questions_preserves_type_ratio() -> None:
    questions = (
        [{"qid": f"c{i}", "type": "concept_locate"} for i in range(6)]
        + [{"qid": f"m{i}", "type": "method_detail"} for i in range(4)]
        + [{"qid": f"n{i}", "type": "negative"} for i in range(2)]
    )

    selected = select_proportional_questions(questions, sample_size=6)

    assert [q["qid"] for q in selected] == ["c0", "c1", "c2", "m0", "m1", "n0"]
