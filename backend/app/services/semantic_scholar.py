"""Semantic Scholar citation metadata adapter."""
from __future__ import annotations

from dataclasses import dataclass
from email.utils import parsedate_to_datetime
import threading
import time
from typing import Optional

import requests

from app.core.config import get_settings

_API_ROOT = "https://api.semanticscholar.org/graph/v1"
_TIMEOUT_SECONDS = 20
_PAPER_FIELDS = "paperId,externalIds,title,year,authors"
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_request_lock = threading.Lock()
_last_request_started_at = 0.0


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


def _wait_for_rate_limit() -> None:
    """Serialize request starts and keep them below the approved S2 quota."""
    global _last_request_started_at
    interval = max(1.01, float(get_settings().semantic_scholar_min_interval_sec))
    with _request_lock:
        now = time.monotonic()
        wait_seconds = interval - (now - _last_request_started_at)
        if wait_seconds > 0:
            time.sleep(wait_seconds)
            now += wait_seconds
        _last_request_started_at = now


def _retry_after_seconds(response: requests.Response, fallback: float) -> float:
    raw = str(response.headers.get("Retry-After") or "").strip()
    if not raw:
        return fallback
    try:
        return max(fallback, float(raw))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(raw)
            return max(fallback, retry_at.timestamp() - time.time())
        except (TypeError, ValueError, OverflowError):
            return fallback


def _request_json(path: str, *, params: Optional[dict] = None) -> dict:
    settings = get_settings()
    attempts = max(1, int(settings.semantic_scholar_max_retries))
    response: requests.Response | None = None
    for attempt in range(attempts):
        _wait_for_rate_limit()
        try:
            response = requests.get(
                f"{_API_ROOT}{path}",
                headers=_headers(),
                params=params,
                timeout=_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            if attempt >= attempts - 1:
                raise CitationSourceUnavailable(
                    f"request failed: {type(exc).__name__}: {exc}"
                ) from exc
            time.sleep(
                max(0.0, float(settings.semantic_scholar_retry_backoff_sec))
                * (2**attempt)
            )
            continue
        if response.status_code not in _RETRYABLE_STATUS:
            break
        if attempt >= attempts - 1:
            break
        fallback = max(
            0.0,
            float(settings.semantic_scholar_retry_backoff_sec) * (2**attempt),
        )
        time.sleep(_retry_after_seconds(response, fallback))

    if response is None:
        raise CitationSourceUnavailable("request produced no response")
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


def _fetch_paged_papers(
    paper_ref: str,
    direction: str,
    item_key: str,
    *,
    max_papers: int,
) -> tuple[RemotePaper, ...]:
    offset = 0
    papers: list[RemotePaper] = []
    limit = max(0, int(max_papers))
    while len(papers) < limit:
        page_limit = min(1000, limit - len(papers))
        payload = _request_json(
            f"/paper/{paper_ref}/{direction}",
            params={"fields": _PAPER_FIELDS, "limit": page_limit, "offset": offset},
        )
        for item in payload.get("data") or []:
            raw = item.get(item_key) if isinstance(item, dict) else None
            if isinstance(raw, dict) and raw.get("paperId"):
                papers.append(_remote_paper(raw))
                if len(papers) >= limit:
                    break
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

    neighbor_limit = max(0, int(get_settings().semantic_scholar_neighbor_limit))
    return CitationSnapshot(
        source=source,
        references=_fetch_paged_papers(
            source_ref,
            "references",
            "citedPaper",
            max_papers=neighbor_limit,
        ),
        citations=_fetch_paged_papers(
            source_ref,
            "citations",
            "citingPaper",
            max_papers=neighbor_limit,
        ),
    )
