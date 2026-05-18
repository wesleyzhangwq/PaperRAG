"""Test retrieve_arxiv tool."""
from unittest.mock import patch, MagicMock

from app.tools.retrieve_arxiv import retrieve_arxiv_tool


def _mock_arxiv_result(title, summary, entry_id, published_year):
    r = MagicMock()
    r.title = title
    r.summary = summary
    r.entry_id = entry_id
    r.published = MagicMock()
    r.published.year = published_year
    r.primary_category = "cs.CL"
    r.authors = [MagicMock(__str__=lambda self: "Author A")]
    return r


def test_retrieve_arxiv_returns_formatted():
    mock_results = [
        _mock_arxiv_result("Paper A", "Abstract A about transformers", "http://arxiv.org/abs/2401.00001", 2024),
    ]
    with patch("app.tools.retrieve_arxiv.arxiv") as mock_arxiv:
        mock_client = MagicMock()
        mock_arxiv.Client.return_value = mock_client
        mock_client.results.return_value = iter(mock_results)
        mock_arxiv.Search.return_value = MagicMock()
        mock_arxiv.SortCriterion.Relevance = "relevance"

        result = retrieve_arxiv_tool.invoke({"query": "transformers", "max_results": 5})

    assert "Paper A" in result
    assert "2401.00001" in result
    assert "Abstract A" in result


def test_retrieve_arxiv_empty():
    with patch("app.tools.retrieve_arxiv.arxiv") as mock_arxiv:
        mock_client = MagicMock()
        mock_arxiv.Client.return_value = mock_client
        mock_client.results.return_value = iter([])
        mock_arxiv.Search.return_value = MagicMock()
        mock_arxiv.SortCriterion.Relevance = "relevance"

        result = retrieve_arxiv_tool.invoke({"query": "nonexistent", "max_results": 3})

    assert "No papers" in result
