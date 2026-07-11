from __future__ import annotations

from collections import Counter

import pytest

from eval.scripts.gen_questions import (
    _parse_json,
    build_generation_plan,
    build_split_manifest,
    build_trend_topic_groups,
    evidence_chunk_ids_for_papers,
    make_negative_questions,
    paper_evidence_excerpt,
    split_papers_by_category,
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


def test_split_papers_by_category_is_exact_deterministic_and_disjoint() -> None:
    papers = [
        {
            "paper_id": f"P{i:03d}",
            "title": f"Paper {i}",
            "primary_category": "cs.CL" if i < 12 else "cs.AI",
        }
        for i in range(20)
    ]

    dev_a, test_a = split_papers_by_category(papers, dev_size=6, seed=20260711)
    dev_b, test_b = split_papers_by_category(list(reversed(papers)), dev_size=6, seed=20260711)

    assert [paper["paper_id"] for paper in dev_a] == [paper["paper_id"] for paper in dev_b]
    assert [paper["paper_id"] for paper in test_a] == [paper["paper_id"] for paper in test_b]
    assert len(dev_a) == 6
    assert len(test_a) == 14
    assert {paper["paper_id"] for paper in dev_a}.isdisjoint(
        {paper["paper_id"] for paper in test_a}
    )
    assert Counter(paper["primary_category"] for paper in dev_a) == {
        "cs.CL": 4,
        "cs.AI": 2,
    }


def test_evidence_helpers_preserve_labels_and_dedupe_chunk_ids() -> None:
    papers = [
        {
            "paper_id": "P1",
            "evidence_chunks": [
                {"chunk_id": "P1::0", "text": "method evidence"},
                {"chunk_id": "P1::1", "text": "result evidence"},
            ],
            "evidence_text": "[P1::0] method evidence\n\n[P1::1] result evidence",
        },
        {
            "paper_id": "P2",
            "evidence_chunks": [
                {"chunk_id": "P2::0", "text": "comparison evidence"},
                {"chunk_id": "P1::1", "text": "duplicate id"},
            ],
            "evidence_text": "[P2::0] comparison evidence",
        },
    ]

    assert evidence_chunk_ids_for_papers(papers) == ["P1::0", "P1::1", "P2::0"]
    assert paper_evidence_excerpt(papers[0], max_chars=30).startswith("[P1::0]")
    assert len(paper_evidence_excerpt(papers[0], max_chars=30)) <= 30


def test_build_split_manifest_records_exact_disjoint_paper_sets() -> None:
    dev = [{"paper_id": "P1", "chunk_count": 3}]
    test = [{"paper_id": "P2", "chunk_count": 4}, {"paper_id": "P3", "chunk_count": 5}]

    manifest = build_split_manifest(
        corpus_path="corpus.json",
        corpus_sha256="abc123",
        all_papers=dev + test,
        dev_papers=dev,
        test_papers=test,
        seed=7,
        model="test-model",
    )

    assert manifest["paper_count"] == 3
    assert manifest["chunk_count"] == 12
    assert manifest["dev_paper_ids"] == ["P1"]
    assert manifest["test_paper_ids"] == ["P2", "P3"]
    assert set(manifest["dev_paper_ids"]).isdisjoint(manifest["test_paper_ids"])


def test_validate_question_set_strict_mode_checks_evidence_and_generation_status() -> None:
    corpus = {
        "P1": {
            "paper_id": "P1",
            "title": "A Reliable Retrieval System",
            "evidence_chunks": [{"chunk_id": "P1::0", "paper_id": "P1"}],
        }
    }
    row = {
        "qid": "c001",
        "query": "哪项工作通过候选重排提高了证据检索质量？",
        "expected_paper_ids": ["P1"],
        "expected_mode": "answer",
        "reference_answer": "该方法组合语义检索与候选重排。",
        "difficulty": "easy",
        "type": "concept_locate",
        "tags": ["test"],
        "evidence_chunk_ids": ["P1::0"],
        "generation_status": "llm",
    }

    summary = validate_question_set(
        [row],
        expected_plan={"concept_locate": 1},
        corpus_by_id=corpus,
        require_evidence=True,
        reject_fallback=True,
        reject_title_leakage=True,
    )

    assert summary["evidence_chunk_count"] == 1
    assert summary["fallback_count"] == 0


def test_validate_question_set_allows_short_generic_method_title_in_query() -> None:
    corpus = {
        "P1": {
            "paper_id": "P1",
            "title": "Layer Normalization",
            "evidence_chunks": [{"chunk_id": "P1::0", "paper_id": "P1"}],
        }
    }
    row = {
        "qid": "c001",
        "query": "Layer Normalization 如何改善深层网络的训练稳定性？",
        "expected_paper_ids": ["P1"],
        "expected_mode": "answer",
        "reference_answer": "该方法在特征维度执行归一化。",
        "difficulty": "easy",
        "type": "concept_locate",
        "tags": ["test"],
        "evidence_chunk_ids": ["P1::0"],
        "generation_status": "llm",
    }

    validate_question_set(
        [row],
        expected_plan={"concept_locate": 1},
        corpus_by_id=corpus,
        require_evidence=True,
        reject_fallback=True,
        reject_title_leakage=True,
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"evidence_chunk_ids": ["P1::missing"]}, "unknown evidence chunk"),
        ({"generation_status": "fallback"}, "fallback"),
        ({"query": "A Reliable Retrieval System 使用了什么方法？"}, "title leakage"),
        ({"query": "论文 P1 使用了什么方法？"}, "paper id leakage"),
    ],
)
def test_validate_question_set_strict_mode_rejects_untrusted_rows(
    changes: dict,
    message: str,
) -> None:
    corpus = {
        "P1": {
            "paper_id": "P1",
            "title": "A Reliable Retrieval System",
            "evidence_chunks": [{"chunk_id": "P1::0", "paper_id": "P1"}],
        }
    }
    row = {
        "qid": "c001",
        "query": "哪项工作通过候选重排提高了证据检索质量？",
        "expected_paper_ids": ["P1"],
        "expected_mode": "answer",
        "reference_answer": "该方法组合语义检索与候选重排。",
        "difficulty": "easy",
        "type": "concept_locate",
        "tags": ["test"],
        "evidence_chunk_ids": ["P1::0"],
        "generation_status": "llm",
    }
    row.update(changes)

    with pytest.raises(ValueError, match=message):
        validate_question_set(
            [row],
            expected_plan={"concept_locate": 1},
            corpus_by_id=corpus,
            require_evidence=True,
            reject_fallback=True,
            reject_title_leakage=True,
        )
