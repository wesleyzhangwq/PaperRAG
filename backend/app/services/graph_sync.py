"""Idempotent projection of locally ingested papers into Neo4j."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Optional

from sqlalchemy.orm import Session

from app.db.neo4j import get_neo4j_repository
from app.db.mysql import SessionLocal, init_db
from app.models.paper import Paper
from app.services.semantic_scholar import (
    CitationSourceNotFound,
    CitationSnapshot,
    RemotePaper,
    fetch_citation_snapshot,
)


def _mark(paper: Paper, status: str, error: Optional[str] = None) -> str:
    paper.graph_sync_status = status
    paper.graph_sync_error = error
    paper.graph_synced_at = (
        datetime.now(timezone.utc).replace(tzinfo=None) if status == "ok" else None
    )
    return status


def _node_for(remote: RemotePaper, local_papers: dict[str, Paper]) -> dict[str, object]:
    local = local_papers.get(remote.arxiv_id or "")
    graph_key = f"arxiv:{local.paper_id}" if local is not None else f"s2:{remote.s2_paper_id}"
    return {
        "graph_key": graph_key,
        "paper_id": local.paper_id if local is not None else None,
        "s2_paper_id": remote.s2_paper_id,
        "title": remote.title,
        "year": remote.year,
        "in_corpus": local is not None,
    }


def build_projection_payload(
    source_paper: Paper,
    snapshot: CitationSnapshot,
    local_papers: dict[str, Paper],
) -> dict[str, object]:
    """Map a citation snapshot into Neo4j node and edge parameter lists."""
    source_node = _node_for(snapshot.source, local_papers)
    source_node["graph_key"] = f"arxiv:{source_paper.paper_id}"
    source_node["paper_id"] = source_paper.paper_id
    source_node["in_corpus"] = True

    nodes: dict[str, dict[str, object]] = {str(source_node["graph_key"]): source_node}
    citation_edges: list[dict[str, str]] = []
    authors: dict[tuple[str, str], dict[str, str]] = {}

    def add_authors(node: dict[str, object], remote: RemotePaper) -> None:
        if not node.get("in_corpus"):
            return
        for author_id, name in remote.authors:
            authors[(str(node["graph_key"]), author_id)] = {
                "paper_key": str(node["graph_key"]),
                "author_id": author_id,
                "name": name,
            }

    add_authors(source_node, snapshot.source)
    for reference in snapshot.references:
        target_node = _node_for(reference, local_papers)
        nodes[str(target_node["graph_key"])] = target_node
        citation_edges.append({
            "source_key": str(source_node["graph_key"]),
            "target_key": str(target_node["graph_key"]),
        })
        add_authors(target_node, reference)
    for citation in snapshot.citations:
        citing_node = _node_for(citation, local_papers)
        nodes[str(citing_node["graph_key"])] = citing_node
        citation_edges.append({
            "source_key": str(citing_node["graph_key"]),
            "target_key": str(source_node["graph_key"]),
        })
        add_authors(citing_node, citation)

    unique_edges = {
        (edge["source_key"], edge["target_key"]): edge for edge in citation_edges
    }
    return {
        "papers": list(nodes.values()),
        "citation_edges": list(unique_edges.values()),
        "authors": list(authors.values()),
        "categories": [str(category) for category in source_paper.categories or [] if str(category)],
    }


def sync_paper(
    db: Session,
    paper: Paper,
    *,
    local_papers: dict[str, Paper],
) -> str:
    """Synchronize one local paper without allowing graph failures to escape."""
    if paper.ingest_status != "ok" or (paper.num_chunks or 0) < 1:
        return _mark(paper, "pending")
    try:
        snapshot = fetch_citation_snapshot(arxiv_id=paper.paper_id, doi=paper.doi)
        payload = build_projection_payload(paper, snapshot, local_papers)
        get_neo4j_repository().replace_source_projection(
            source_paper_id=paper.paper_id,
            papers=payload["papers"],
            citation_edges=payload["citation_edges"],
            authors=payload["authors"],
            categories=payload["categories"],
        )
    except CitationSourceNotFound:
        try:
            get_neo4j_repository().replace_source_projection(
                source_paper_id=paper.paper_id,
                papers=[
                    {
                        "graph_key": f"arxiv:{paper.paper_id}",
                        "paper_id": paper.paper_id,
                        "s2_paper_id": None,
                        "title": paper.title,
                        "year": paper.year,
                        "in_corpus": True,
                    }
                ],
                citation_edges=[],
                authors=[],
                categories=[
                    str(category)
                    for category in paper.categories or []
                    if str(category)
                ],
            )
        except Exception as exc:
            return _mark(paper, "failed", f"{type(exc).__name__}: {exc}")
        return _mark(paper, "unresolved")
    except Exception as exc:
        return _mark(paper, "failed", f"{type(exc).__name__}: {exc}")
    return _mark(paper, "ok")


def _successful_papers(db: Session) -> list[Paper]:
    return (
        db.query(Paper)
        .filter(Paper.ingest_status == "ok", Paper.num_chunks > 0)
        .order_by(Paper.paper_id.asc())
        .all()
    )


def run_graph_sync(
    paper_ids: Optional[Iterable[str]] = None,
    *,
    force: bool = False,
) -> dict[str, int]:
    """Synchronize all local papers or an explicit set of arXiv IDs."""
    init_db()
    requested = {str(paper_id) for paper_id in paper_ids or [] if str(paper_id)}
    stats = {
        "ok": 0,
        "unresolved": 0,
        "failed": 0,
        "pending": 0,
        "skipped_ok": 0,
        "total": 0,
    }
    db = SessionLocal()
    try:
        local = _successful_papers(db)
        local_papers = {paper.paper_id: paper for paper in local}
        targets = [paper for paper in local if not requested or paper.paper_id in requested]
        for paper in targets:
            stats["total"] += 1
            if paper.graph_sync_status == "ok" and not force:
                stats["skipped_ok"] += 1
                continue
            status = sync_paper(db, paper, local_papers=local_papers)
            stats[status] += 1
            db.commit()
        get_neo4j_repository().prune_orphans()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
    return stats
