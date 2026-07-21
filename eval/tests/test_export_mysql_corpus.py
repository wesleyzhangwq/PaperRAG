from __future__ import annotations

from types import SimpleNamespace

import pytest

from eval.scripts.export_mysql_corpus import (
    build_paper_record,
    normalize_text,
    select_representative_chunks,
)


def _chunk(index: int, text: str, *, paper_id: str = "2501.00001") -> dict:
    return {
        "chunk_id": f"{paper_id}::{index}",
        "paper_id": paper_id,
        "chunk_index": index,
        "page_num": index + 1,
        "text": text,
    }


def test_normalize_text_collapses_whitespace_and_redacts_secrets() -> None:
    secret = "sk-" + "abc123456789XYZ"

    assert normalize_text(f"alpha\n\n beta\t{secret}") == "alpha beta [REDACTED_SECRET]"


def test_select_representative_chunks_is_deterministic_and_section_aware() -> None:
    chunks = [
        _chunk(5, "Appendix implementation details and acknowledgements."),
        _chunk(0, "Paper title and abstract. We introduce a retrieval framework."),
        _chunk(4, "Conclusion. The proposed approach improves robust retrieval."),
        _chunk(2, "Method. Our architecture combines dense retrieval and reranking."),
        _chunk(3, "Experiments and Results. Evaluation on three benchmarks shows gains."),
        _chunk(1, "Introduction and related work overview."),
    ]

    first = select_representative_chunks(chunks, limit=4, text_chars=500)
    second = select_representative_chunks(list(reversed(chunks)), limit=4, text_chars=500)

    assert [item["chunk_id"] for item in first] == [item["chunk_id"] for item in second]
    assert "2501.00001::0" in {item["chunk_id"] for item in first}
    assert "2501.00001::2" in {item["chunk_id"] for item in first}
    assert "2501.00001::3" in {item["chunk_id"] for item in first}


def test_build_paper_record_bounds_evidence_and_preserves_chunk_ownership() -> None:
    paper = SimpleNamespace(
        paper_id="2501.00001",
        title="A Test Paper",
        authors=["Ada"],
        year=2025,
        primary_category="cs.CL",
        categories=["cs.CL", "cs.IR"],
        doi=None,
        abstract=None,
    )
    chunks = [
        _chunk(i, f"Method section {i}. " + ("useful evidence " * 80))
        for i in range(10)
    ]

    record = build_paper_record(
        paper,
        chunks,
        evidence_limit=5,
        text_chars=240,
        max_evidence_chars=900,
    )

    assert record["paper_id"] == "2501.00001"
    assert record["chunk_count"] == 10
    assert 1 <= len(record["evidence_chunks"]) <= 5
    assert len(record["evidence_text"]) <= 900
    assert all(item["paper_id"] == record["paper_id"] for item in record["evidence_chunks"])
    assert all(item["chunk_id"].startswith("2501.00001::") for item in record["evidence_chunks"])


def test_build_paper_record_rejects_foreign_chunk() -> None:
    paper = SimpleNamespace(
        paper_id="2501.00001",
        title="A Test Paper",
        authors=[],
        year=2025,
        primary_category="cs.CL",
        categories=["cs.CL"],
        doi=None,
        abstract=None,
    )

    with pytest.raises(ValueError, match="does not belong"):
        build_paper_record(paper, [_chunk(0, "foreign", paper_id="2501.99999")])
