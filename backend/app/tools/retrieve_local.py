"""Tool: local vector + BM25 hybrid retrieval."""
from __future__ import annotations

from langchain_core.tools import tool

from app.schemas.chat import ChatFilter
from app.services.retriever import retrieve


def _format_chunks(docs_scores: list) -> str:
    if not docs_scores:
        return "No results found."
    parts = []
    for d, score in docs_scores:
        md = d.metadata or {}
        parts.append(
            f"[arxiv:{md.get('paper_id', '?')} | {md.get('title', '')[:100]} | "
            f"score={score:.3f} | page={md.get('page_num', '?')}]\n"
            f"{d.page_content[:500]}"
        )
    return "\n\n---\n\n".join(parts)


@tool
def retrieve_local_tool(
    query: str,
    top_k: int = 8,
    category: str = "",
    year_min: int = 0,
    year_max: int = 0,
) -> str:
    """Search local paper database using hybrid vector + BM25 retrieval.
    Returns ranked chunks with paper IDs, relevance scores, and text snippets.
    Use when you need to find relevant academic paper content."""
    flt = None
    if category or year_min or year_max:
        flt = ChatFilter(
            category=category or None,
            year_min=year_min if year_min > 0 else None,
            year_max=year_max if year_max > 0 else None,
        )
    docs_scores = retrieve(query, flt=flt, top_k=top_k)
    return _format_chunks(docs_scores)
