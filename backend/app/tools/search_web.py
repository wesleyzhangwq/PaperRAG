"""Tool: web search via Tavily API."""
from __future__ import annotations

from langchain_core.tools import tool

from app.core.config import get_settings


@tool
def search_web_tool(query: str, max_results: int = 3) -> str:
    """Search the web for background knowledge, explanations, or recent news.
    Use when the question requires context beyond academic papers."""
    settings = get_settings()
    if not settings.tavily_api_key:
        return "Web search is not configured (TAVILY_API_KEY missing)."

    from tavily import TavilyClient

    client = TavilyClient(api_key=settings.tavily_api_key)
    response = client.search(query=query, max_results=max_results)

    results = response.results if hasattr(response, "results") else response.get("results", [])
    if not results:
        return "No web results found."

    parts = []
    for r in results:
        title = r.get("title", "") if isinstance(r, dict) else r.title
        url = r.get("url", "") if isinstance(r, dict) else r.url
        content = r.get("content", "") if isinstance(r, dict) else r.content
        parts.append(f"[{title}]({url})\n{content[:300]}")
    return "\n\n---\n\n".join(parts)
