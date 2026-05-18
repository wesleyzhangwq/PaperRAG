"""Test retrieve_local tool."""
from unittest.mock import patch
from langchain_core.documents import Document

from app.tools.retrieve_local import retrieve_local_tool


def test_retrieve_local_returns_formatted_chunks():
    mock_docs = [
        (Document(page_content="Attention is all you need", metadata={"paper_id": "1706.03762", "title": "Attention", "page_num": 1}), 0.92),
        (Document(page_content="BERT uses masked LM", metadata={"paper_id": "1810.04805", "title": "BERT", "page_num": 3}), 0.85),
    ]
    with patch("app.tools.retrieve_local.retrieve", return_value=mock_docs):
        result = retrieve_local_tool.invoke({"query": "attention mechanism", "top_k": 8})

    assert "1706.03762" in result
    assert "0.920" in result
    assert "Attention is all you need" in result


def test_retrieve_local_with_filter():
    mock_docs = [
        (Document(page_content="Some NLP content", metadata={"paper_id": "2301.00001", "title": "NLP Paper", "page_num": 2}), 0.88),
    ]
    with patch("app.tools.retrieve_local.retrieve", return_value=mock_docs) as mock_retrieve:
        result = retrieve_local_tool.invoke({"query": "NLP", "top_k": 5, "category": "cs.CL", "year_min": 2023})

    call_args = mock_retrieve.call_args
    assert call_args[1]["flt"] is not None
    assert call_args[1]["flt"].category == "cs.CL"


def test_retrieve_local_empty_results():
    with patch("app.tools.retrieve_local.retrieve", return_value=[]):
        result = retrieve_local_tool.invoke({"query": "nonexistent topic"})

    assert "No results" in result
