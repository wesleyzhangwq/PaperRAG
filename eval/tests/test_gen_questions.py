from __future__ import annotations

from collections import Counter

import pytest

from eval.scripts.gen_questions import (
    _parse_json,
    build_generation_plan,
    build_trend_topic_groups,
    make_negative_questions,
    validate_question_set,
)


def test_build_generation_plan_controls_200_question_distribution() -> None:
    plan = build_generation_plan(
        total=200,
        concept_count=60,
        method_count=40,
        fact_count=30,
        comparison_count=30,
        trend_count=20,
        negative_count=20,
    )

    assert plan == {
        "concept_locate": 60,
        "method_detail": 40,
        "fact_extract": 30,
        "comparison": 30,
        "trend_synthesis": 20,
        "negative": 20,
    }


def test_build_generation_plan_rejects_incorrect_total() -> None:
    with pytest.raises(ValueError, match="sum to 200"):
        build_generation_plan(
            total=200,
            concept_count=60,
            method_count=40,
            fact_count=30,
            comparison_count=30,
            trend_count=20,
            negative_count=19,
        )


def test_parse_json_accepts_python_dict_style_llm_output() -> None:
    assert _parse_json("```json\n{'question': '问题', 'reference_answer': '答案'}\n```") == {
        "question": "问题",
        "reference_answer": "答案",
    }


def test_make_negative_questions_extends_static_bank_with_stable_ids() -> None:
    questions = make_negative_questions(count=20)

    assert len(questions) == 20
    assert [q["qid"] for q in questions] == [f"n{i:03d}" for i in range(1, 21)]
    assert {q["type"] for q in questions} == {"negative"}
    assert {q["expected_mode"] for q in questions} == {"insufficient"}
    assert all(q["expected_paper_ids"] == [] for q in questions)
    assert Counter(q["difficulty"] for q in questions) == {
        "easy": 7,
        "medium": 6,
        "hard": 7,
    }


def test_validate_question_set_checks_count_distribution_and_placeholders() -> None:
    rows = []
    for prefix, qtype, count in [
        ("c", "concept_locate", 2),
        ("m", "method_detail", 1),
        ("f", "fact_extract", 1),
        ("x", "comparison", 1),
        ("t", "trend_synthesis", 1),
        ("n", "negative", 1),
    ]:
        for i in range(1, count + 1):
            rows.append(
                {
                    "qid": f"{prefix}{i:03d}",
                    "query": f"{qtype} question {i}",
                    "expected_paper_ids": [] if qtype == "negative" else ["P1"],
                    "expected_mode": "insufficient" if qtype == "negative" else "answer",
                    "reference_answer": "reference",
                    "difficulty": "easy",
                    "type": qtype,
                    "tags": ["test"],
                }
            )

    summary = validate_question_set(
        rows,
        expected_plan={
            "concept_locate": 2,
            "method_detail": 1,
            "fact_extract": 1,
            "comparison": 1,
            "trend_synthesis": 1,
            "negative": 1,
        },
    )

    assert summary["total"] == 7
    assert summary["by_type"]["concept_locate"] == 2
    assert summary["positive"] == 6
    assert summary["negative"] == 1


def test_validate_question_set_rejects_placeholders() -> None:
    with pytest.raises(ValueError, match="placeholder"):
        validate_question_set(
            [
                {
                    "qid": "c001",
                    "query": "<question text>",
                    "expected_paper_ids": ["P1"],
                    "expected_mode": "answer",
                    "reference_answer": "reference",
                    "difficulty": "easy",
                    "type": "concept_locate",
                    "tags": ["test"],
                }
            ],
            expected_plan={"concept_locate": 1},
        )


def test_build_trend_topic_groups_can_expand_large_categories() -> None:
    papers = [
        {
            "paper_id": f"P{i:03d}",
            "title": f"Large Language Model Agent Retrieval Benchmark {i}",
            "primary_category": "cs.CL" if i < 18 else "cs.LG",
            "corpus_bucket": "language_models" if i < 12 else "retrieval",
        }
        for i in range(30)
    ]

    groups = build_trend_topic_groups(papers, count=6)

    assert len(groups) == 6
    assert all(len(group) >= 2 for _, group in groups)
    assert len({topic for topic, _ in groups}) == 6
