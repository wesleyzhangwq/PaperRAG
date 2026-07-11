"""Graph-assisted local evidence retrieval.

Neo4j only selects related *local paper IDs*.  The second pass always retrieves
the final chunks from Qdrant, so graph metadata never becomes answer evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

from langchain_core.documents import Document

from app.core.config import get_settings
from app.db.neo4j import GraphCandidate, GraphUnavailable
from app.schemas.chat import ChatFilter
from app.services.retriever import retrieve
from app.tools.retrieve_graph import retrieve_graph_candidates


@dataclass(frozen=True)
class GraphRetrievalReport:
    seed_paper_ids: tuple[str, ...]
    candidates: tuple[GraphCandidate, ...]
    added_chunks: int
    fallback_reason: str | None
    graph_elapsed_ms: float


def with_retrieval_metadata(
    doc: Document,
    score: float,
    *,
    source: str,
    graph_score: float | None = None,
    graph_paths: list[dict[str, object]] | None = None,
    semantic_score: float | None = None,
) -> Document:
    """Copy a retrieved document while preserving retrieval provenance.

    Retriever results may come from a shared TTL cache, so metadata must never
    be mutated in-place here.
    """
    metadata = {
        **(doc.metadata or {}),
        "retrieval_score": float(score),
        "retrieval_source": source,
    }
    if graph_score is not None:
        metadata["graph_score"] = float(graph_score)
    if semantic_score is not None:
        metadata["semantic_score"] = float(semantic_score)
    if graph_paths:
        metadata["graph_paths"] = graph_paths
    return Document(page_content=doc.page_content, metadata=metadata)


def _document_score(doc: Document) -> float:
    metadata = doc.metadata or {}
    raw = metadata.get("retrieval_score", metadata.get("score", 0.0))
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _seed_scores(existing_context: list[Document], max_seeds: int) -> dict[str, float]:
    """Return the highest-scored local chunk score for each paper."""
    highest_by_paper: dict[str, float] = {}
    for doc in existing_context:
        metadata = doc.metadata or {}
        if metadata.get("retrieval_source") not in {"local", "graph_local"}:
            continue
        paper_id = str(metadata.get("paper_id") or "")
        if not paper_id:
            continue
        highest_by_paper[paper_id] = max(
            highest_by_paper.get(paper_id, float("-inf")), _document_score(doc)
        )
    ranked = sorted(highest_by_paper.items(), key=lambda item: (-item[1], item[0]))
    return dict(ranked[:max(0, max_seeds)])


def _report(
    *,
    seed_scores: dict[str, float],
    candidates: list[GraphCandidate] | tuple[GraphCandidate, ...] = (),
    added_chunks: int = 0,
    fallback_reason: str | None,
    started_at: float,
) -> GraphRetrievalReport:
    return GraphRetrievalReport(
        seed_paper_ids=tuple(seed_scores),
        candidates=tuple(candidates),
        added_chunks=added_chunks,
        fallback_reason=fallback_reason,
        graph_elapsed_ms=round((time.perf_counter() - started_at) * 1000, 2),
    )


def retrieve_graph_context(
    *,
    query: str,
    existing_context: list[Document],
    top_k: int,
) -> tuple[list[Document], GraphRetrievalReport]:
    """Expand local evidence through Neo4j then retrieve local chunks only.

    A graph failure is intentionally indistinguishable from a no-op retrieval:
    the caller retains its existing local context and continues the pipeline.
    """
    started_at = time.perf_counter()
    settings = get_settings()
    if not settings.graph_rag_enabled:
        return list(existing_context), _report(
            seed_scores={}, fallback_reason="graph_disabled", started_at=started_at
        )

    seed_scores = _seed_scores(existing_context, settings.graph_seed_papers)
    if not seed_scores:
        return list(existing_context), _report(
            seed_scores=seed_scores,
            fallback_reason="no_local_seeds",
            started_at=started_at,
        )

    try:
        candidates = retrieve_graph_candidates(
            seed_paper_ids=list(seed_scores),
            seed_scores=seed_scores,
            max_hops=min(2, max(1, settings.graph_max_hops)),
            limit=settings.graph_candidate_limit,
        )
    except GraphUnavailable:
        return list(existing_context), _report(
            seed_scores=seed_scores,
            fallback_reason="neo4j_unavailable",
            started_at=started_at,
        )

    if not candidates:
        return list(existing_context), _report(
            seed_scores=seed_scores,
            candidates=candidates,
            fallback_reason="no_new_local_candidates",
            started_at=started_at,
        )

    candidates_by_paper = {candidate.paper_id: candidate for candidate in candidates}
    docs_scores = retrieve(
        query,
        flt=ChatFilter(paper_ids=list(candidates_by_paper)),
        top_k=max(1, int(top_k)),
    )
    graph_documents: list[Document] = []
    for doc, score in docs_scores:
        paper_id = str((doc.metadata or {}).get("paper_id") or "")
        candidate = candidates_by_paper.get(paper_id)
        if candidate is None:
            # Retrieval filters should make this impossible, but do not attach
            # graph provenance to a chunk we cannot attribute to a candidate.
            continue
        combined_score = float(score) * float(candidate.graph_score)
        graph_documents.append(with_retrieval_metadata(
            doc,
            combined_score,
            source="graph_local",
            graph_score=candidate.graph_score,
            graph_paths=[dict(path) for path in candidate.paths],
            semantic_score=float(score),
        ))

    if not graph_documents:
        return list(existing_context), _report(
            seed_scores=seed_scores,
            candidates=candidates,
            fallback_reason="no_new_local_candidates",
            started_at=started_at,
        )
    return graph_documents, _report(
        seed_scores=seed_scores,
        candidates=candidates,
        added_chunks=len(graph_documents),
        fallback_reason=None,
        started_at=started_at,
    )


__all__ = ["GraphRetrievalReport", "retrieve_graph_context", "with_retrieval_metadata"]
