from app.utils.citations import extract_citation_ids, strip_disallowed_citations


def test_citation_parser_supports_arxiv_and_uploaded_sources():
    text = (
        "Published evidence [arxiv:1706.03762] and local evidence "
        "[source:local-a1b2c3]."
    )

    assert extract_citation_ids(text) == ["1706.03762", "local-a1b2c3"]


def test_citation_gate_strips_disallowed_local_source_marker():
    cleaned, removed = strip_disallowed_citations(
        "Keep [source:local-ok] remove [source:local-unknown].",
        {"local-ok"},
    )

    assert "[source:local-ok]" in cleaned
    assert "local-unknown" not in cleaned
    assert removed == ["local-unknown"]
