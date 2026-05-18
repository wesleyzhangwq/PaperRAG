"""Test reasoning tools (query_rewrite, evaluate_docs)."""
from unittest.mock import patch, MagicMock

from app.tools.query_rewrite import rewrite_query
from app.tools.evaluate_docs import evaluate_docs


def test_rewrite_query_returns_list():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='["BERT pretraining strategy", "GPT pretraining approach"]'
    )
    with patch("app.tools.query_rewrite._get_llm", return_value=mock_llm):
        result = rewrite_query("compare BERT and GPT pretraining", {"type": "comparison", "entities": ["BERT", "GPT"]})
    assert isinstance(result, list)
    assert len(result) == 2
    assert "BERT" in result[0]


def test_rewrite_query_single():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content='["attention mechanism in transformers"]')
    with patch("app.tools.query_rewrite._get_llm", return_value=mock_llm):
        result = rewrite_query("what is attention", {"type": "simple"})
    assert isinstance(result, list)
    assert len(result) == 1


def test_evaluate_docs_sufficient():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='{"sufficient": true, "reason": "Context covers the topic well", "missing_aspects": []}'
    )
    with patch("app.tools.evaluate_docs._get_llm", return_value=mock_llm):
        result = evaluate_docs("what is attention", ["chunk about attention mechanisms"])
    assert result["sufficient"] is True
    assert result["missing_aspects"] == []


def test_evaluate_docs_insufficient():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='{"sufficient": false, "reason": "Missing information about multi-head attention", "missing_aspects": ["multi-head attention details"]}'
    )
    with patch("app.tools.evaluate_docs._get_llm", return_value=mock_llm):
        result = evaluate_docs("explain multi-head attention", ["basic attention info"])
    assert result["sufficient"] is False
    assert len(result["missing_aspects"]) > 0
