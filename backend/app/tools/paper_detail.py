"""Tool: get full paper metadata from MySQL."""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.models.paper import Paper


def get_paper_detail(db: Session, paper_id: str) -> str:
    """Get full metadata for a paper. Returns title, authors, year, categories, abstract."""
    paper = db.query(Paper).filter(Paper.paper_id == paper_id).one_or_none()
    if paper is None:
        return f"Paper {paper_id} not found in database."
    authors = ", ".join(paper.authors or [])
    categories = ", ".join(paper.categories or [])
    return (
        f"paper_id: {paper.paper_id}\n"
        f"title: {paper.title}\n"
        f"authors: {authors}\n"
        f"year: {paper.year}\n"
        f"categories: {categories}\n"
        f"abstract: {paper.abstract or 'N/A'}"
    )
