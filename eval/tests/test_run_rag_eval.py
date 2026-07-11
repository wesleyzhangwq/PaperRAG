from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

from eval.graph_rag_eval import GraphServiceOutcome

from eval.run_rag_eval import (
    build_retrieved_chunks,
    lexical_paper_retrieve,
    load_lexical_corpus,
    postprocess_chunks,
    detect_abstention,
    evaluate_generation_case,
    extract_citation_pids,
    load_questions,
    redact_sensitive_text,
    rows_from_retrieval_detail,
    run_pure_rag_eval,
)


class FakeDocument:
    def __init__(self, page_content: str, metadata: dict) -> None:
        self.page_content = page_content
        self.metadata = metadata


def test_load_questions_skips_placeholders_and_applies_limit(tmp_path) -> None:
    dataset = tmp_path / "questions.jsonl"
    rows = [
        {"qid": "ok1", "query": "real question", "expected_paper_ids": ["A"]},
        {"qid": "skip1", "query": "<question text>", "expected_paper_ids": ["B"]},
        {"qid": "skip2", "query": "...", "expected_paper_ids": ["C"]},
        {"qid": "ok2", "query": "another question", "expected_paper_ids": []},
    ]
    dataset.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )

    loaded = load_questions(dataset, limit=1)

    assert [row["qid"] for row in loaded] == ["ok1"]


def test_build_retrieved_chunks_normalizes_document_metadata_and_snippet() -> None:
    chunks = build_retrieved_chunks(
        [
            (
                FakeDocument(
                    "x" * 500,
                    {
                        "paper_id": "2604.15244",
                        "title": "SpecGuard",
                        "chunk_id": "c1",
                        "page": 3,
                    },
                ),
                0.91,
            )
        ],
        snippet_chars=120,
    )

    assert chunks == [
        {
            "rank": 1,
            "paper_id": "2604.15244",
            "title": "SpecGuard",
            "chunk_id": "c1",
            "page": 3,
            "score": 0.91,
            "snippet": "x" * 120,
        }
    ]


def test_build_retrieved_chunks_redacts_secret_like_snippets() -> None:
    fake_secret = "sk-" + "abcdefghijklmnopqrstuvwxyz1234567890"
    chunks = build_retrieved_chunks(
        [
            (
                FakeDocument(
                    f"token {fake_secret} should not leak",
                    {"paper_id": "A"},
                ),
                0.5,
            )
        ]
    )

    assert chunks[0]["snippet"] == "token [REDACTED_SECRET] should not leak"
    assert redact_sensitive_text(f"x {fake_secret} y") == (
        "x [REDACTED_SECRET] y"
    )


def test_load_lexical_corpus_and_paper_bm25_retrieve(tmp_path) -> None:
    corpus = tmp_path / "papers.json"
    corpus.write_text(
        json.dumps(
            [
                {
                    "paper_id": "A",
                    "title": "Graph Retrieval for Scientific Papers",
                    "abstract": "This paper studies citation-aware graph retrieval.",
                    "primary_category": "cs.IR",
                },
                {
                    "paper_id": "B",
                    "title": "Vision Models for Medical Images",
                    "abstract": "This paper studies segmentation and uncertainty.",
                    "primary_category": "cs.CV",
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    papers = load_lexical_corpus(corpus)
    results = lexical_paper_retrieve("citation graph retrieval", papers, top_k=2)
    chunks = build_retrieved_chunks(results)

    assert chunks[0]["paper_id"] == "A"
    assert chunks[0]["title"] == "Graph Retrieval for Scientific Papers"
    assert chunks[0]["snippet"].startswith("Graph Retrieval")


def test_lexical_retrieve_indexes_and_returns_representative_evidence(tmp_path) -> None:
    corpus = tmp_path / "papers.json"
    corpus.write_text(
        json.dumps(
            [
                {
                    "paper_id": "A",
                    "title": "Generic Research Paper",
                    "abstract": "",
                    "primary_category": "cs.IR",
                    "evidence_chunks": [
                        {
                            "chunk_id": "A::7",
                            "text": "quasarneedle evidence describes citation expansion",
                        }
                    ],
                },
                {
                    "paper_id": "B",
                    "title": "Another Generic Paper",
                    "abstract": "",
                    "primary_category": "cs.IR",
                    "evidence_chunks": [
                        {"chunk_id": "B::2", "text": "unrelated visual segmentation"}
                    ],
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    papers = load_lexical_corpus(corpus)
    results = lexical_paper_retrieve("quasarneedle", papers, top_k=1)
    chunks = build_retrieved_chunks(results)

    assert chunks[0]["paper_id"] == "A"
    assert "quasarneedle evidence" in chunks[0]["snippet"]


def test_postprocess_chunks_can_dedupe_by_paper() -> None:
    chunks = [
        {"rank": 1, "paper_id": "A", "score": 0.9, "snippet": "alpha"},
        {"rank": 2, "paper_id": "A", "score": 0.8, "snippet": "alpha duplicate"},
        {"rank": 3, "paper_id": "B", "score": 0.7, "snippet": "beta"},
    ]

    out = postprocess_chunks(chunks, strategy="paper_dedup", context_k=3)

    assert [chunk["paper_id"] for chunk in out] == ["A", "B"]
    assert [chunk["rank"] for chunk in out] == [1, 2]


def test_postprocess_chunks_mmr_dedup_promotes_diverse_papers() -> None:
    chunks = [
        {"rank": 1, "paper_id": "A", "score": 0.9, "snippet": "alpha alpha alpha graph"},
        {"rank": 2, "paper_id": "A", "score": 0.89, "snippet": "alpha alpha graph duplicate"},
        {"rank": 3, "paper_id": "B", "score": 0.7, "snippet": "beta planning agent"},
        {"rank": 4, "paper_id": "C", "score": 0.69, "snippet": "gamma vision model"},
    ]

    out = postprocess_chunks(chunks, strategy="mmr_dedup", context_k=3, mmr_lambda=0.65)

    assert [chunk["paper_id"] for chunk in out] == ["A", "B", "C"]
    assert [chunk["rank"] for chunk in out] == [1, 2, 3]


def test_generation_case_scores_citations_and_mode() -> None:
    row = evaluate_generation_case(
        answer="SpecGuard uses internal verification signals [arxiv:2604.15244].",
        expected_paper_ids=["2604.15244"],
        expected_mode="answer",
        context_pids=["2604.15244", "2604.15148"],
    )

    assert row["answer_abstained"] is False
    assert row["mode_correct"] is True
    assert row["citation_pids"] == ["2604.15244"]
    assert row["citation_support_rate"] == 1.0
    assert row["citation_precision"] == 1.0
    assert row["citation_expected_hit"] == 1.0


def test_generation_case_recognizes_insufficient_negative_answer() -> None:
    assert detect_abstention("当前语料信息不足，无法确定。") is True
    assert extract_citation_pids("See [arxiv:2604.15244] and 2604.15148.") == [
        "2604.15244",
        "2604.15148",
    ]

    row = evaluate_generation_case(
        answer="当前语料信息不足，无法回答该问题。",
        expected_paper_ids=[],
        expected_mode="insufficient",
        context_pids=["2604.15244"],
    )

    assert row["answer_abstained"] is True
    assert row["mode_correct"] is True
    assert row["citation_support_rate"] is None


def test_rows_from_retrieval_detail_replays_existing_detail_json() -> None:
    rows = rows_from_retrieval_detail(
        {
            "per_question": [
                {
                    "qid": "c001",
                    "query": "single paper",
                    "difficulty": "easy",
                    "type": "concept_locate",
                    "expected_pids": ["A"],
                    "predicted_pids": ["B", "A"],
                    "ndcg": 0.7,
                    "precision": 0.5,
                    "recall": 1.0,
                    "rr": 0.25,
                    "latency": 0.4,
                }
            ]
        },
        k_values=[1, 3],
        context_k=3,
    )

    assert len(rows) == 1
    assert rows[0]["ranked_pids"] == ["B", "A"]
    assert rows[0]["hit_at_1"] == 0.0
    assert rows[0]["ndcg_at_5"] == 0.7
    assert rows[0]["precision_at_5"] == 0.5
    assert rows[0]["recall_at_5"] == 1.0
    assert rows[0]["mrr"] == 0.25
    assert rows[0]["recall_at_3"] == 1.0
    assert rows[0]["retrieved_chunks"][1]["paper_id"] == "A"


def test_pure_rag_runner_uses_service_graph_and_records_expansion_metadata() -> None:
    question = {
        "qid": "g001",
        "query": "compare graph papers",
        "expected_paper_ids": ["B"],
        "expected_mode": "answer",
        "difficulty": "hard",
        "type": "comparison",
        "reference_answer": "",
    }
    graph_doc = FakeDocument(
        "graph evidence",
        {"paper_id": "B", "title": "Graph paper", "chunk_id": "B:1"},
    )
    outcome = GraphServiceOutcome(
        results=[(graph_doc, 0.8)],
        graph_expansion_ms=12.5,
        graph_fallback_reason=None,
        graph_candidate_count=2,
    )

    settings = SimpleNamespace(graph_rag_enabled=True, retrieval_k=12)
    with patch("app.core.config.get_settings", return_value=settings), patch(
        "eval.graph_rag_eval.retrieve_service_graph", return_value=outcome
    ) as retrieve_graph:
        rows = run_pure_rag_eval(
            questions=[question],
            k_values=[1, 5],
            context_k=1,
            retrieval_top_k=4,
            graph_expansion_top_k=6,
            generate=False,
            retriever_name="service_graph",
        )

    retrieve_graph.assert_called_once_with(
        "compare graph papers", seed_top_k=4, expansion_top_k=6
    )
    assert rows[0]["recall_at_5"] == 1.0
    assert rows[0]["graph_expansion_ms"] == 12.5
    assert rows[0]["graph_candidate_count"] == 2
