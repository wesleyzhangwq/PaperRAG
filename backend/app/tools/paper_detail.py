"""Tool: get full paper metadata from MySQL."""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.models.paper import Paper


def load_paper(db: Session, paper_id: str) -> Paper | None:
    """Load one paper by its stable external identifier."""
    return db.query(Paper).filter(Paper.paper_id == paper_id).one_or_none()


def format_paper_detail(paper: Paper | None, paper_id: str) -> str:
    """Format one paper as model-readable evidence."""
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


def get_paper_detail(db: Session, paper_id: str) -> str:
    """Get full metadata for a paper. Returns title, authors, year, categories, abstract."""
    return format_paper_detail(load_paper(db, paper_id), paper_id)
