"""Lazy Neo4j repository for citation-graph synchronization and traversal."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from neo4j import Driver, GraphDatabase
from neo4j.exceptions import Neo4jError

from app.core.config import get_settings

_RELATION_WEIGHTS = {"CITES": 1.0, "AUTHORED_BY": 0.45, "IN_CATEGORY": 0.20}
_RELATIONSHIP_TYPES = tuple(_RELATION_WEIGHTS)


class GraphUnavailable(RuntimeError):
    """Raised when an optional graph operation cannot be completed."""


@dataclass(frozen=True)
class GraphCandidate:
    paper_id: str
    graph_score: float
    paths: tuple[dict[str, object], ...]


class Neo4jGraphRepository:
    def __init__(self, *, driver: Driver, database: str, timeout_ms: int) -> None:
        self._driver = driver
        self._database = database
        self._timeout_seconds = max(0.001, timeout_ms / 1000)

    def ensure_schema(self) -> None:
        statements = (
            "CREATE CONSTRAINT paper_graph_key IF NOT EXISTS FOR (p:Paper) REQUIRE p.graph_key IS UNIQUE",
            "CREATE INDEX paper_local_id IF NOT EXISTS FOR (p:Paper) ON (p.paper_id)",
            "CREATE INDEX paper_s2_id IF NOT EXISTS FOR (p:Paper) ON (p.s2_paper_id)",
            "CREATE CONSTRAINT author_s2_id IF NOT EXISTS FOR (a:Author) REQUIRE a.s2_author_id IS UNIQUE",
            "CREATE CONSTRAINT category_code IF NOT EXISTS FOR (c:Category) REQUIRE c.code IS UNIQUE",
        )
        try:
            with self._driver.session(database=self._database) as session:
                for statement in statements:
                    session.run(statement, timeout=self._timeout_seconds).consume()
        except Neo4jError as exc:
            raise GraphUnavailable(f"schema setup failed: {type(exc).__name__}: {exc}") from exc

    def replace_source_projection(
        self,
        *,
        source_paper_id: str,
        papers: list[dict[str, object]],
        citation_edges: list[dict[str, str]],
        authors: list[dict[str, str]],
        categories: list[str],
    ) -> None:
        delete_edges = (
            "MATCH ()-[relationship]->() "
            "WHERE relationship.source_paper_id = $source_paper_id "
            "DELETE relationship"
        )
        merge_papers = (
            "UNWIND $papers AS item "
            "MERGE (paper:Paper {graph_key: item.graph_key}) "
            "SET paper.paper_id = item.paper_id, paper.s2_paper_id = item.s2_paper_id, "
            "paper.title = item.title, paper.year = item.year, paper.in_corpus = item.in_corpus"
        )
        merge_citations = (
            "UNWIND $edges AS edge "
            "MATCH (source:Paper {graph_key: edge.source_key}) "
            "MATCH (target:Paper {graph_key: edge.target_key}) "
            "MERGE (source)-[relationship:CITES {source_paper_id: $source_paper_id}]->(target) "
            "SET relationship.source = 'semantic_scholar', relationship.retrieved_at = datetime()"
        )
        merge_authors = (
            "UNWIND $authors AS item "
            "MATCH (paper:Paper {graph_key: item.paper_key}) "
            "MERGE (author:Author {s2_author_id: item.author_id}) "
            "SET author.name = item.name "
            "MERGE (paper)-[relationship:AUTHORED_BY {source_paper_id: $source_paper_id}]->(author)"
        )
        merge_categories = (
            "MATCH (paper:Paper {paper_id: $source_paper_id}) "
            "UNWIND $categories AS code "
            "MERGE (category:Category {code: code}) "
            "MERGE (paper)-[relationship:IN_CATEGORY {source_paper_id: $source_paper_id}]->(category)"
        )
        try:
            with self._driver.session(database=self._database) as session:
                session.run(delete_edges, source_paper_id=source_paper_id, timeout=self._timeout_seconds).consume()
                session.run(merge_papers, papers=papers, timeout=self._timeout_seconds).consume()
                session.run(
                    merge_citations,
                    edges=citation_edges,
                    source_paper_id=source_paper_id,
                    timeout=self._timeout_seconds,
                ).consume()
                session.run(
                    merge_authors,
                    authors=authors,
                    source_paper_id=source_paper_id,
                    timeout=self._timeout_seconds,
                ).consume()
                session.run(
                    merge_categories,
                    categories=categories,
                    source_paper_id=source_paper_id,
                    timeout=self._timeout_seconds,
                ).consume()
        except Neo4jError as exc:
            raise GraphUnavailable(f"projection write failed: {type(exc).__name__}: {exc}") from exc

    def expand_local_papers(
        self,
        *,
        seed_paper_ids: list[str],
        seed_scores: dict[str, float],
        max_hops: int,
        limit: int,
    ) -> list[GraphCandidate]:
        if not seed_paper_ids or limit < 1:
            return []
        query = """
        MATCH (seed:Paper {in_corpus: true})
        WHERE seed.paper_id IN $seed_paper_ids
        MATCH path=(seed)-[*1..2]-(candidate:Paper {in_corpus: true})
        WHERE candidate.paper_id <> seed.paper_id
          AND length(path) <= $max_hops
          AND all(rel IN relationships(path) WHERE type(rel) IN $relationship_types)
        RETURN DISTINCT candidate.paper_id AS paper_id,
               seed.paper_id AS seed_paper_id,
               length(path) AS hops,
               [rel IN relationships(path) | type(rel)] AS relations
        LIMIT $raw_limit
        """
        try:
            with self._driver.session(database=self._database) as session:
                rows = list(session.run(
                    query,
                    seed_paper_ids=seed_paper_ids,
                    max_hops=min(2, max(1, max_hops)),
                    relationship_types=list(_RELATIONSHIP_TYPES),
                    raw_limit=max(limit * 6, limit),
                    timeout=self._timeout_seconds,
                ))
        except Neo4jError as exc:
            raise GraphUnavailable(f"graph traversal failed: {type(exc).__name__}: {exc}") from exc

        paths_by_paper: dict[str, list[dict[str, object]]] = {}
        for row in rows:
            values: dict[str, Any] = dict(row)
            paper_id = str(values.get("paper_id") or "")
            seed_paper_id = str(values.get("seed_paper_id") or "")
            relations = [str(item) for item in values.get("relations") or []]
            hops = int(values.get("hops") or 0)
            if not paper_id or not seed_paper_id or hops < 1 or not relations:
                continue
            relation_weight = min(_RELATION_WEIGHTS.get(item, 0.0) for item in relations)
            score = float(seed_scores.get(seed_paper_id, 0.0)) * relation_weight * (0.65 ** (hops - 1))
            if score <= 0:
                continue
            paths_by_paper.setdefault(paper_id, []).append({
                "seed_paper_id": seed_paper_id,
                "hops": hops,
                "relations": relations,
                "score": round(score, 6),
            })

        candidates: list[GraphCandidate] = []
        for paper_id, paths in paths_by_paper.items():
            ranked_paths = sorted(paths, key=lambda item: float(item["score"]), reverse=True)[:3]
            candidates.append(GraphCandidate(
                paper_id=paper_id,
                graph_score=float(ranked_paths[0]["score"]),
                paths=tuple(ranked_paths),
            ))
        return sorted(candidates, key=lambda item: (-item.graph_score, item.paper_id))[:limit]


@lru_cache
def get_neo4j_repository() -> Neo4jGraphRepository:
    settings = get_settings()
    if not settings.neo4j_uri or not settings.neo4j_password:
        raise GraphUnavailable("Neo4j is not configured")
    try:
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            connection_timeout=max(0.001, settings.graph_query_timeout_ms / 1000),
        )
    except Neo4jError as exc:
        raise GraphUnavailable(f"Neo4j driver creation failed: {type(exc).__name__}: {exc}") from exc
    return Neo4jGraphRepository(
        driver=driver,
        database=settings.neo4j_database,
        timeout_ms=settings.graph_query_timeout_ms,
    )
