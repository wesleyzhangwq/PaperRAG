# backend/app/services/tools.py
"""Agent tools for PaperRAG — each wraps an existing service function."""
from __future__ import annotations

from langchain_core.tools import tool
from sqlalchemy.orm import Session

from app.models.paper import Paper
from app.schemas.chat import ChatFilter
from app.services.retriever import retrieve


def _format_chunks(docs_scores: list) -> str:
    parts = []
    for d, score in docs_scores:
        md = d.metadata or {}
        parts.append(
            f"[arxiv:{md.get('paper_id','?')} | score={score:.3f} | "
            f"page={md.get('page_num','?')}]\n{d.page_content[:500]}"
        )
    return "\n\n---\n\n".join(parts) if parts else "No results found."


@tool
def search_papers(query: str, top_k: int = 8) -> str:
    """Search academic papers using hybrid vector + BM25 retrieval.
    Returns ranked chunks with paper IDs, scores, and text snippets.
    Use this when the user asks a question that requires finding relevant papers."""
    docs_scores = retrieve(query, top_k=top_k)
    return _format_chunks(docs_scores)


@tool
def filter_papers(query: str, category: str = "", year_min: int = 0, year_max: int = 0) -> str:
    """Search papers with metadata filters applied.
    Use this when the user wants results from a specific category (e.g. 'cs.CL')
    or year range. Pass category as arXiv category string, year_min/year_max as integers (0 means no limit)."""
    flt = ChatFilter(
        category=category or None,
        year_min=year_min if year_min > 0 else None,
        year_max=year_max if year_max > 0 else None,
    )
    docs_scores = retrieve(query, flt=flt, top_k=8)
    return _format_chunks(docs_scores)


def get_paper_detail(db: Session, paper_id: str) -> str:
    """Get full metadata for a specific paper by its arxiv paper_id.
    Returns title, authors, year, categories, abstract."""
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


def compare_papers(db: Session, paper_ids: list[str]) -> str:
    """Compare multiple papers side by side.
    Returns a formatted comparison of titles, authors, years, and abstracts."""
    parts = []
    for pid in paper_ids[:5]:
        paper = db.query(Paper).filter(Paper.paper_id == pid).one_or_none()
        if paper is None:
            parts.append(f"## {pid}\nNot found in database.")
            continue
        authors = ", ".join((paper.authors or [])[:5])
        parts.append(
            f"## {paper.paper_id}\n"
            f"title: {paper.title}\n"
            f"authors: {authors}\n"
            f"year: {paper.year}\n"
            f"category: {paper.primary_category}\n"
            f"abstract: {(paper.abstract or 'N/A')[:600]}"
        )
    return "\n\n---\n\n".join(parts) if parts else "No papers found."
