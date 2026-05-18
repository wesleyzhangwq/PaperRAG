"""Tool: real-time arXiv API search."""
from __future__ import annotations

import arxiv
from langchain_core.tools import tool

from app.core.config import get_settings


@tool
def retrieve_arxiv_tool(query: str, max_results: int = 5) -> str:
    """Search arXiv for recent papers matching the query.
    Returns paper titles, abstracts, IDs, and categories.
    Use when local database may not have the latest papers or when you need broader coverage."""
    settings = get_settings()
    max_results = min(max_results, settings.arxiv_max_results)

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    client = arxiv.Client()
    results = list(client.results(search))

    if not results:
        return "No papers found on arXiv for this query."

    parts = []
    for r in results:
        paper_id = r.entry_id.split("/abs/")[-1] if "/abs/" in r.entry_id else r.entry_id
        authors = ", ".join(str(a) for a in (r.authors or [])[:3])
        parts.append(
            f"[arxiv:{paper_id} | {r.primary_category} | {r.published.year}]\n"
            f"title: {r.title}\n"
            f"authors: {authors}\n"
            f"abstract: {r.summary[:400]}"
        )
    return "\n\n---\n\n".join(parts)
