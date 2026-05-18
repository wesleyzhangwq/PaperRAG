"""Test knowledge tools (paper_detail, paper_chunks)."""
from unittest.mock import MagicMock

from app.tools.paper_detail import get_paper_detail
from app.tools.paper_chunks import get_paper_chunks


def _mock_paper():
    p = MagicMock()
    p.paper_id = "2301.00001"
    p.title = "Test Paper"
    p.authors = ["Author A", "Author B"]
    p.year = 2023
    p.primary_category = "cs.CL"
    p.categories = ["cs.CL", "cs.AI"]
    p.abstract = "This paper presents a novel approach."
    p.doi = None
    return p


def test_get_paper_detail_found():
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.one_or_none.return_value = _mock_paper()
    result = get_paper_detail(mock_db, "2301.00001")
    assert "2301.00001" in result
    assert "Test Paper" in result
    assert "Author A" in result


def test_get_paper_detail_not_found():
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.one_or_none.return_value = None
    result = get_paper_detail(mock_db, "9999.99999")
    assert "not found" in result.lower()


def _mock_chunk(text, page_num):
    c = MagicMock()
    c.chunk_text = text
    c.page_num = page_num
    c.chunk_index = 0
    return c


def test_get_paper_chunks_found():
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = [
        _mock_chunk("First chunk content", 1),
        _mock_chunk("Second chunk content", 2),
    ]
    result = get_paper_chunks(mock_db, "2301.00001", max_chunks=10)
    assert "First chunk" in result
    assert "Second chunk" in result


def test_get_paper_chunks_empty():
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []
    result = get_paper_chunks(mock_db, "9999.99999")
    assert "no chunks" in result.lower() or "not found" in result.lower()
