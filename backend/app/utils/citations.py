"""Canonical citation parsing for arXiv and uploaded corpus documents."""
from __future__ import annotations

import re
from collections.abc import Iterator


CITATION_RE = re.compile(
    r"(?:"
    r"(?:\[\s*)?arxiv\s*:\s*([0-9]{4}\.[0-9]{4,6})(?:v\d+)?(?:\s*\])?"
    r"|"
    r"\[\s*source\s*:\s*"
    r"([A-Za-z0-9](?:[A-Za-z0-9._:-]{0,62}[A-Za-z0-9])?)\s*\]"
    r"|"
    r"https?://arxiv\.org/abs/([0-9]{4}\.[0-9]{4,6})(?:v\d+)?"
    r")",
    re.IGNORECASE,
)
ARXIV_CITATION_RE = CITATION_RE


def _paper_id(match: re.Match[str]) -> str:
    return str(match.group(1) or match.group(2) or match.group(3))


def iter_citations(text: str) -> Iterator[tuple[re.Match[str], str]]:
    for match in CITATION_RE.finditer(text or ""):
        yield match, _paper_id(match)


def iter_arxiv_citations(text: str) -> Iterator[tuple[re.Match[str], str]]:
    """Backward-compatible alias; now yields all supported corpus citations."""
    yield from iter_citations(text)


def extract_citation_ids(text: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for _, paper_id in iter_citations(text):
        if paper_id not in seen:
            found.append(paper_id)
            seen.add(paper_id)
    return found


def extract_arxiv_ids(text: str) -> list[str]:
    """Backward-compatible name used throughout the agent safety chain."""
    return extract_citation_ids(text)


def strip_disallowed_citations(text: str, allowed_ids: set[str]) -> tuple[str, list[str]]:
    removed: list[str] = []

    def replace(match: re.Match[str]) -> str:
        paper_id = _paper_id(match)
        if paper_id in allowed_ids:
            return match.group(0)
        if paper_id not in removed:
            removed.append(paper_id)
        return ""

    cleaned = CITATION_RE.sub(replace, text or "")
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return cleaned, removed


__all__ = [
    "ARXIV_CITATION_RE",
    "CITATION_RE",
    "extract_citation_ids",
    "extract_arxiv_ids",
    "iter_citations",
    "iter_arxiv_citations",
    "strip_disallowed_citations",
]
