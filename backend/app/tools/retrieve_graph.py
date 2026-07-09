"""Tool-facing boundary for bounded citation-graph expansion."""
from __future__ import annotations

from app.db.neo4j import GraphCandidate, get_neo4j_repository


def retrieve_graph_candidates(
    *,
    seed_paper_ids: list[str],
    seed_scores: dict[str, float],
    max_hops: int,
    limit: int,
) -> list[GraphCandidate]:
    """Expand local seed papers through the optional Neo4j projection."""
    return get_neo4j_repository().expand_local_papers(
        seed_paper_ids=seed_paper_ids,
        seed_scores=seed_scores,
        max_hops=max_hops,
        limit=limit,
    )


__all__ = ["retrieve_graph_candidates"]
