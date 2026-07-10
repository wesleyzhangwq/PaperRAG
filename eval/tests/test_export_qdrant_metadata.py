from __future__ import annotations

from eval.scripts.export_qdrant_metadata import (
    collect_papers_from_payloads,
    normalize_text,
    paper_from_chunks,
)


def test_normalize_text_collapses_whitespace() -> None:
    assert normalize_text(" a\n\n b\t c ") == "a b c"


def test_normalize_text_redacts_secret_like_tokens() -> None:
    secret_like = "sk-" + "abc123456789XYZ"
    assert normalize_text(f"token {secret_like} value") == "token [REDACTED_SECRET] value"


def test_paper_from_chunks_builds_metadata_from_sorted_unique_chunks() -> None:
    chunks = [
        {
            "metadata": {
                "paper_id": "2604.00001",
                "title": "Example Paper",
                "primary_category": "cs.CL",
                "year": 2026,
                "doi": "10.123/example",
                "page_num": 2,
                "chunk_index": 2,
            },
            "text": "second chunk",
        },
        {
            "metadata": {
                "paper_id": "2604.00001",
                "title": "Example Paper",
                "primary_category": "cs.CL",
                "year": 2026,
                "doi": "10.123/example",
                "page_num": 1,
                "chunk_index": 1,
            },
            "text": " first\nchunk ",
        },
        {
            "metadata": {
                "paper_id": "2604.00001",
                "title": "Example Paper",
                "primary_category": "cs.CL",
                "year": 2026,
                "page_num": 1,
                "chunk_index": 3,
            },
            "text": "first chunk",
        },
    ]

    paper = paper_from_chunks("2604.00001", chunks)

    assert paper["paper_id"] == "2604.00001"
    assert paper["title"] == "Example Paper"
    assert paper["primary_category"] == "cs.CL"
    assert paper["categories"] == ["cs.CL"]
    assert paper["published"] == "2026-01-01"
    assert paper["pdf_url"] == "https://arxiv.org/pdf/2604.00001"
    assert paper["abstract"] == "first chunk second chunk"
    assert paper["chunk_count"] == 3


def test_collect_papers_from_payloads_skips_payloads_without_paper_id() -> None:
    payloads = [
        {"metadata": {}, "text": "ignored"},
        {
            "metadata": {
                "paper_id": "2604.00002",
                "title": "Another Paper",
                "primary_category": "cs.LG",
            },
            "text": "content",
        },
    ]

    papers = collect_papers_from_payloads(payloads)

    assert len(papers) == 1
    assert papers[0]["paper_id"] == "2604.00002"
