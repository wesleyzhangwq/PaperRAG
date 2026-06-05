"""Tool: web search via Tavily API."""
from __future__ import annotations

from langchain_core.tools import tool

from app.core.config import get_settings


def _format_unavailable_error(exc: Exception) -> str:
    message = str(exc).strip().replace("\n", " ")
    if len(message) > 240:
        message = message[:237] + "..."
    return f"Web search unavailable: {type(exc).__name__}: {message}"


@tool
def search_web_tool(query: str, max_results: int = 3) -> str:
    """Search the web for background knowledge, explanations, or recent news.
    Use when the question requires context beyond academic papers."""
    settings = get_settings()
    if not settings.tavily_api_key:
        return "Web search is not configured (TAVILY_API_KEY missing)."

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=settings.tavily_api_key)
        response = client.search(query=query, max_results=max_results)
    except Exception as exc:
        return _format_unavailable_error(exc)

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
