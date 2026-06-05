"""Test search_web tool."""
from unittest.mock import patch, MagicMock

from requests.exceptions import SSLError

from app.tools.search_web import search_web_tool


def test_search_web_returns_formatted():
    mock_response = {"results": [
        {"title": "What is Attention?", "url": "https://example.com/attention", "content": "Attention mechanisms allow models to focus..."},
    ]}
    with patch("app.tools.search_web.get_settings") as mock_settings, \
         patch("tavily.TavilyClient") as MockClient:
        mock_settings.return_value.tavily_api_key = "test-key"
        MockClient.return_value.search.return_value = mock_response
        result = search_web_tool.invoke({"query": "attention mechanism explained", "max_results": 3})

    assert "What is Attention?" in result
    assert "example.com" in result


def test_search_web_no_api_key():
    with patch("app.tools.search_web.get_settings") as mock_settings:
        mock_settings.return_value.tavily_api_key = None
        result = search_web_tool.invoke({"query": "test"})

    assert "not configured" in result.lower()


def test_search_web_handles_tavily_ssl_error():
    with patch("app.tools.search_web.get_settings") as mock_settings, \
         patch("tavily.TavilyClient") as MockClient:
        mock_settings.return_value.tavily_api_key = "test-key"
        MockClient.return_value.search.side_effect = SSLError("unexpected eof")

        result = search_web_tool.invoke({"query": "latest transformer history", "max_results": 3})

    assert "web search unavailable" in result.lower()
    assert "unexpected eof" in result.lower()
