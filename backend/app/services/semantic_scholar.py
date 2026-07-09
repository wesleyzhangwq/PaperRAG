"""Semantic Scholar citation metadata adapter."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import requests

from app.core.config import get_settings

_API_ROOT = "https://api.semanticscholar.org/graph/v1"
_TIMEOUT_SECONDS = 20
_PAPER_FIELDS = "paperId,externalIds,title,year,authors.authorId,authors.name"


class CitationSourceUnavailable(RuntimeError):
    """Raised for retryable citation-source failures."""


class CitationSourceNotFound(RuntimeError):
    """Raised when no external identity resolves for a local paper."""


@dataclass(frozen=True)
class RemotePaper:
    s2_paper_id: str
    arxiv_id: Optional[str]
    doi: Optional[str]
    title: str
    year: Optional[int]
    authors: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class CitationSnapshot:
    source: RemotePaper
    references: tuple[RemotePaper, ...]
    citations: tuple[RemotePaper, ...]


def _headers() -> dict[str, str]:
    api_key = get_settings().semantic_scholar_api_key
    return {"x-api-key": api_key} if api_key else {}


def _request_json(path: str, *, params: Optional[dict] = None) -> dict:
    try:
        response = requests.get(
            f"{_API_ROOT}{path}",
            headers=_headers(),
            params=params,
            timeout=_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise CitationSourceUnavailable(
            f"request failed: {type(exc).__name__}: {exc}"
        ) from exc
    if response.status_code == 404:
        raise CitationSourceNotFound(path)
    if response.status_code >= 400:
        raise CitationSourceUnavailable(
            f"response status {response.status_code}: {response.text[:200]}"
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise CitationSourceUnavailable("response was not JSON") from exc
    if not isinstance(payload, dict):
        raise CitationSourceUnavailable("response JSON was not an object")
    return payload


def _remote_paper(raw: dict) -> RemotePaper:
    external_ids = raw.get("externalIds") or {}
    paper_id = str(raw.get("paperId") or "").strip()
    if not paper_id:
        raise CitationSourceUnavailable("citation response omitted paperId")
    authors = tuple(
        (str(author.get("authorId") or ""), str(author.get("name") or ""))
        for author in raw.get("authors") or []
        if author.get("authorId") and author.get("name")
    )
    year = raw.get("year")
    return RemotePaper(
        s2_paper_id=paper_id,
        arxiv_id=str(external_ids.get("ArXiv") or "").strip() or None,
        doi=str(external_ids.get("DOI") or "").strip() or None,
        title=str(raw.get("title") or "").strip(),
        year=int(year) if year is not None else None,
        authors=authors,
    )


def _fetch_paged_papers(paper_ref: str, direction: str, item_key: str) -> tuple[RemotePaper, ...]:
    offset = 0
    papers: list[RemotePaper] = []
    while True:
        payload = _request_json(
            f"/paper/{paper_ref}/{direction}",
            params={"fields": _PAPER_FIELDS, "limit": 1000, "offset": offset},
        )
        for item in payload.get("data") or []:
            raw = item.get(item_key) if isinstance(item, dict) else None
            if isinstance(raw, dict):
                papers.append(_remote_paper(raw))
        next_offset = payload.get("next")
        if not isinstance(next_offset, int) or next_offset <= offset:
            break
        offset = next_offset
    return tuple(papers)


def _fetch_source(paper_ref: str) -> RemotePaper:
    return _remote_paper(_request_json(
        f"/paper/{paper_ref}", params={"fields": _PAPER_FIELDS}
    ))


def fetch_citation_snapshot(*, arxiv_id: str, doi: Optional[str]) -> CitationSnapshot:
    """Fetch one source paper and its explicit reference/citation neighbors."""
    identifiers = [f"ARXIV:{arxiv_id}"]
    if doi:
        identifiers.append(f"DOI:{doi}")

    source: Optional[RemotePaper] = None
    source_ref = ""
    for identifier in identifiers:
        try:
            source = _fetch_source(identifier)
            source_ref = identifier
            break
        except CitationSourceNotFound:
            continue
    if source is None:
        raise CitationSourceNotFound(arxiv_id)

    return CitationSnapshot(
        source=source,
        references=_fetch_paged_papers(source_ref, "references", "citedPaper"),
        citations=_fetch_paged_papers(source_ref, "citations", "citingPaper"),
    )
