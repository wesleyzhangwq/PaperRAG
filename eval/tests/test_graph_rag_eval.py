from types import SimpleNamespace

from langchain_core.documents import Document
import pytest

from eval.graph_rag_eval import (
    evaluate_graph_merge_gates,
    require_graph_corpus_coverage,
    retrieve_service_graph,
    summarize_graph_expansion,
)


def _doc(paper_id: str, score: float = 0.0) -> Document:
    return Document(
        page_content=f"evidence for {paper_id}",
        metadata={"paper_id": paper_id, "retrieval_score": score},
    )


def test_service_graph_uses_production_stages_and_combines_second_pass_chunks() -> None:
    seed = Document(page_content="seed", metadata={"paper_id": "seed"})
    expanded = _doc("expanded", 0.7)
    report = SimpleNamespace(
        seed_paper_ids=("seed",),
        candidates=(SimpleNamespace(paper_id="expanded"),),
        fallback_reason=None,
        graph_elapsed_ms=23.5,
    )
    captured = {}

    def wrap(doc: Document, score: float, *, source: str) -> Document:
        return _doc(doc.metadata["paper_id"], score)

    def expand(*, query: str, existing_context: list[Document], top_k: int):
        captured["query"] = query
        captured["seed_ids"] = [doc.metadata["paper_id"] for doc in existing_context]
        captured["top_k"] = top_k
        return [expanded], report

    def process(state: dict) -> dict:
        captured["evidence_ids"] = [doc.metadata["paper_id"] for doc in state["retrieval_context"]]
        return {"retrieval_context": state["retrieval_context"]}

    outcome = retrieve_service_graph(
        "cross-paper question",
        seed_top_k=4,
        expansion_top_k=6,
        retrieve_local=lambda query, top_k: [(seed, 0.9)],
        wrap_local=wrap,
        expand_graph=expand,
        process_evidence=process,
    )

    assert captured == {
        "query": "cross-paper question",
        "seed_ids": ["seed"],
        "top_k": 6,
        "evidence_ids": ["seed", "expanded"],
    }
    assert [doc.metadata["paper_id"] for doc, _ in outcome.results] == ["seed", "expanded"]
    assert outcome.graph_candidate_count == 1
    assert outcome.graph_expansion_ms == 23.5
    assert outcome.graph_fallback_reason is None


def test_service_graph_uses_local_seed_only_when_graph_falls_back() -> None:
    seed = Document(page_content="seed", metadata={"paper_id": "seed"})
    report = SimpleNamespace(
        seed_paper_ids=("seed",),
        candidates=(),
        fallback_reason="neo4j_unavailable",
        graph_elapsed_ms=4.0,
    )

    outcome = retrieve_service_graph(
        "question",
        seed_top_k=4,
        expansion_top_k=6,
        retrieve_local=lambda query, top_k: [(seed, 0.9)],
        wrap_local=lambda doc, score, *, source: _doc(doc.metadata["paper_id"], score),
        expand_graph=lambda **kwargs: (kwargs["existing_context"], report),
        process_evidence=lambda state: {"retrieval_context": state["retrieval_context"]},
    )

    assert [doc.metadata["paper_id"] for doc, _ in outcome.results] == ["seed"]
    assert outcome.graph_fallback_reason == "neo4j_unavailable"


def test_summarize_graph_expansion_and_merge_gates() -> None:
    graph_summary = summarize_graph_expansion(
        [
            {"graph_expansion_ms": 10.0, "graph_fallback_reason": None, "graph_candidate_count": 2},
            {"graph_expansion_ms": 120.0, "graph_fallback_reason": None, "graph_candidate_count": 3},
            {"graph_expansion_ms": 790.0, "graph_fallback_reason": "no_new_local_candidates", "graph_candidate_count": 0},
        ]
    )
    candidate = {
        "ndcg_at_5": 0.825,
        "citation_support_rate": 1.0,
        "by_type": {
            "comparison": {"recall_at_5": 0.70},
            "trend_synthesis": {"recall_at_5": 0.55},
        },
        **graph_summary,
    }
    baseline = {
        "ndcg_at_5": 0.83,
        "by_type": {
            "comparison": {"recall_at_5": 0.65},
            "trend_synthesis": {"recall_at_5": 0.50},
        },
    }

    gates = evaluate_graph_merge_gates(baseline, candidate)

    assert graph_summary["graph_expansion_p95_ms"] == 790.0
    assert graph_summary["graph_fallback_rate"] == pytest.approx(1 / 3, abs=0.0001)
    assert gates["passed"] is True
    assert all(item["passed"] for item in gates["checks"])


def test_merge_gates_report_failed_requirements() -> None:
    baseline = {
        "ndcg_at_5": 0.83,
        "by_type": {
            "comparison": {"recall_at_5": 0.65},
            "trend_synthesis": {"recall_at_5": 0.50},
        },
    }
    candidate = {
        "ndcg_at_5": 0.80,
        "citation_support_rate": 0.99,
        "graph_expansion_p95_ms": 801.0,
        "by_type": {
            "comparison": {"recall_at_5": 0.69},
            "trend_synthesis": {"recall_at_5": 0.54},
        },
    }

    gates = evaluate_graph_merge_gates(baseline, candidate)

    assert gates["passed"] is False
    assert [item["name"] for item in gates["checks"] if not item["passed"]] == [
        "comparison_recall_at_5",
        "trend_synthesis_recall_at_5",
        "overall_ndcg_at_5",
        "fixed_context_citation_support",
        "graph_expansion_p95_ms",
    ]


def test_graph_corpus_preflight_rejects_missing_expected_papers() -> None:
    questions = [
        {"expected_paper_ids": ["A", "B"]},
        {"expected_paper_ids": ["B", "C"]},
        {"expected_paper_ids": []},
    ]

    with pytest.raises(ValueError, match="2/3 expected papers are absent"):
        require_graph_corpus_coverage(questions, ready_paper_ids={"A"})
