"""Canonical arXiv citation parsing for runtime gates and offline metrics."""
from __future__ import annotations

import re
from collections.abc import Iterator


ARXIV_CITATION_RE = re.compile(
    r"(?:"
    r"(?:\[\s*)?arxiv\s*:\s*([0-9]{4}\.[0-9]{4,6})(?:v\d+)?(?:\s*\])?"
    r"|"
    r"https?://arxiv\.org/abs/([0-9]{4}\.[0-9]{4,6})(?:v\d+)?"
    r")",
    re.IGNORECASE,
)


def iter_arxiv_citations(text: str) -> Iterator[tuple[re.Match[str], str]]:
    for match in ARXIV_CITATION_RE.finditer(text or ""):
        yield match, str(match.group(1) or match.group(2))


def extract_arxiv_ids(text: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for _, paper_id in iter_arxiv_citations(text):
        if paper_id not in seen:
            found.append(paper_id)
            seen.add(paper_id)
    return found


def strip_disallowed_citations(text: str, allowed_ids: set[str]) -> tuple[str, list[str]]:
    removed: list[str] = []

    def replace(match: re.Match[str]) -> str:
        paper_id = str(match.group(1) or match.group(2))
        if paper_id in allowed_ids:
            return match.group(0)
        if paper_id not in removed:
            removed.append(paper_id)
        return ""

    cleaned = ARXIV_CITATION_RE.sub(replace, text or "")
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return cleaned, removed


__all__ = [
    "ARXIV_CITATION_RE",
    "extract_arxiv_ids",
    "iter_arxiv_citations",
    "strip_disallowed_citations",
]
