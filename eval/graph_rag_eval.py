"""Production-path adapter and gates for Graph RAG evaluation."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from langchain_core.documents import Document


@dataclass(frozen=True)
class GraphServiceOutcome:
    """Pure-RAG-compatible output plus production graph expansion metadata."""

    results: list[tuple[Document, float]]
    graph_expansion_ms: float
    graph_fallback_reason: str | None
    graph_candidate_count: int


def require_graph_corpus_coverage(
    questions: list[dict],
    *,
    ready_paper_ids: set[str] | None = None,
) -> dict:
    """Fail before evaluation when the graph source cannot cover its corpus.

    Neo4j projection starts from MySQL ``Paper`` rows, whereas Pure RAG reads
    Qdrant. A graph candidate run is invalid when the expected papers in the
    evaluation dataset cannot be synchronized from MySQL into Neo4j.
    """
    expected_paper_ids = {
        str(paper_id)
        for question in questions
        for paper_id in question.get("expected_paper_ids") or []
        if str(paper_id).strip()
    }
    if ready_paper_ids is None:
        from app.db.mysql import SessionLocal
        from app.models.paper import Paper

        db = SessionLocal()
        try:
            ready_paper_ids = {
                str(paper_id)
                for (paper_id,) in db.query(Paper.paper_id)
                .filter(
                    Paper.ingest_status == "ok",
                    Paper.num_chunks > 0,
                    Paper.graph_sync_status.in_(("ok", "unresolved")),
                )
                .all()
            }
        finally:
            db.close()

    missing = sorted(expected_paper_ids - ready_paper_ids)
    result = {
        "expected_paper_count": len(expected_paper_ids),
        "graph_source_paper_count": len(ready_paper_ids),
        "covered_expected_paper_count": len(expected_paper_ids) - len(missing),
        "missing_expected_paper_count": len(missing),
        "missing_examples": missing[:10],
    }
    if missing:
        raise ValueError(
            f"Graph RAG evaluation blocked: {len(missing)}/{len(expected_paper_ids)} "
            "expected papers are absent from the MySQL graph-source corpus; "
            f"examples: {', '.join(missing[:5])}"
        )
    return result


def _document_score(doc: Document) -> float:
    raw = (doc.metadata or {}).get("retrieval_score", (doc.metadata or {}).get("score", 0.0))
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def retrieve_service_graph(
    query: str,
    *,
    seed_top_k: int,
    expansion_top_k: int,
    retrieve_local: Callable[..., list[tuple[Document, float]]] | None = None,
    wrap_local: Callable[..., Document] | None = None,
    expand_graph: Callable[..., tuple[list[Document], object]] | None = None,
    process_evidence: Callable[[dict], dict] | None = None,
) -> GraphServiceOutcome:
    """Run the same local -> graph -> local-evidence path as production.

    Neo4j expansion itself stays inside ``retrieve_graph_context``. The eval
    adapter only assembles the preceding local seed retrieval and the following
    deterministic production evidence stage into ranked chunks for metrics.
    """
    if retrieve_local is None or wrap_local is None or expand_graph is None or process_evidence is None:
        from app.agent.nodes.evidence import evidence_node
        from app.services.graph_retriever import retrieve_graph_context, with_retrieval_metadata
        from app.services.retriever import retrieve

        retrieve_local = retrieve_local or retrieve
        wrap_local = wrap_local or with_retrieval_metadata
        expand_graph = expand_graph or retrieve_graph_context
        process_evidence = process_evidence or evidence_node

    seed_results = retrieve_local(query, top_k=max(1, int(seed_top_k)))
    seed_documents = [
        wrap_local(doc, score, source="local")
        for doc, score in seed_results
    ]
    expanded_documents, report = expand_graph(
        query=query,
        existing_context=seed_documents,
        top_k=max(1, int(expansion_top_k)),
    )

    # On production fallback, the graph service returns the original seed
    # context. Do not feed that same list twice into evidence processing.
    fallback_reason = getattr(report, "fallback_reason", None)
    second_pass_documents = [] if fallback_reason is not None else list(expanded_documents)
    evidence_result = process_evidence(
        {"retrieval_context": seed_documents + second_pass_documents, "step_traces": []}
    )
    evidence_documents = list(evidence_result.get("retrieval_context") or [])

    return GraphServiceOutcome(
        results=[(doc, _document_score(doc)) for doc in evidence_documents],
        graph_expansion_ms=float(getattr(report, "graph_elapsed_ms", 0.0)),
        graph_fallback_reason=str(fallback_reason) if fallback_reason is not None else None,
        graph_candidate_count=len(getattr(report, "candidates", ()) or ()),
    )


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(len(ordered) * percentile) - 1)
    return round(ordered[index], 4)


def summarize_graph_expansion(rows: list[dict]) -> dict:
    expansion_ms = [
        float(row["graph_expansion_ms"])
        for row in rows
        if row.get("graph_expansion_ms") is not None
    ]
    fallback_count = sum(1 for row in rows if row.get("graph_fallback_reason"))
    candidate_counts = [
        float(row["graph_candidate_count"])
        for row in rows
        if row.get("graph_candidate_count") is not None
    ]
    count = len(rows)
    return {
        "graph_expansion_count": len(expansion_ms),
        "graph_expansion_p50_ms": _percentile(expansion_ms, 0.50),
        "graph_expansion_p95_ms": _percentile(expansion_ms, 0.95),
        "graph_expansion_mean_ms": round(sum(expansion_ms) / len(expansion_ms), 4)
        if expansion_ms
        else None,
        "graph_fallback_rate": round(fallback_count / count, 4) if count else None,
        "graph_candidate_count_mean": round(sum(candidate_counts) / len(candidate_counts), 4)
        if candidate_counts
        else None,
    }


def _type_metric(summary: dict, qtype: str, metric: str) -> float | None:
    value = ((summary.get("by_type") or {}).get(qtype) or {}).get(metric)
    return float(value) if value is not None else None


def _gate(
    *,
    name: str,
    baseline: float | None,
    candidate: float | None,
    threshold: float,
    direction: str,
) -> dict:
    if baseline is None or candidate is None:
        return {
            "name": name,
            "baseline": baseline,
            "candidate": candidate,
            "threshold": threshold,
            "passed": False,
            "reason": "missing_metric",
        }
    if direction == "increase":
        passed = candidate - baseline >= threshold - 1e-12
    elif direction == "decrease":
        passed = candidate <= threshold + 1e-12
    else:
        raise ValueError(f"Unknown gate direction: {direction}")
    return {
        "name": name,
        "baseline": baseline,
        "candidate": candidate,
        "delta": round(candidate - baseline, 6),
        "threshold": threshold,
        "passed": bool(passed),
    }


def evaluate_graph_merge_gates(baseline: dict, candidate: dict) -> dict:
    """Apply the explicit Graph RAG merge gates to a paired evaluation run."""
    checks = [
        _gate(
            name="comparison_recall_at_5",
            baseline=_type_metric(baseline, "comparison", "recall_at_5"),
            candidate=_type_metric(candidate, "comparison", "recall_at_5"),
            threshold=0.05,
            direction="increase",
        ),
        _gate(
            name="trend_synthesis_recall_at_5",
            baseline=_type_metric(baseline, "trend_synthesis", "recall_at_5"),
            candidate=_type_metric(candidate, "trend_synthesis", "recall_at_5"),
            threshold=0.05,
            direction="increase",
        ),
        _gate(
            name="overall_ndcg_at_5",
            baseline=float(baseline["ndcg_at_5"]) if baseline.get("ndcg_at_5") is not None else None,
            candidate=float(candidate["ndcg_at_5"]) if candidate.get("ndcg_at_5") is not None else None,
            threshold=-0.01,
            direction="increase",
        ),
        _gate(
            name="fixed_context_citation_support",
            baseline=1.0,
            candidate=float(candidate["citation_support_rate"])
            if candidate.get("citation_support_rate") is not None
            else None,
            threshold=0.0,
            direction="increase",
        ),
        _gate(
            name="graph_expansion_p95_ms",
            baseline=0.0,
            candidate=float(candidate["graph_expansion_p95_ms"])
            if candidate.get("graph_expansion_p95_ms") is not None
            else None,
            threshold=800.0,
            direction="decrease",
        ),
    ]
    return {"passed": all(check["passed"] for check in checks), "checks": checks}


__all__ = [
    "GraphServiceOutcome",
    "evaluate_graph_merge_gates",
    "require_graph_corpus_coverage",
    "retrieve_service_graph",
    "summarize_graph_expansion",
]
