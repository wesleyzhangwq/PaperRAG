"""Weekly paper radar for a focused PaperRAG research direction.

The job is intentionally arXiv-first because PaperRAG's source resolver and
citation UI currently use arXiv IDs as the stable paper identity.
"""
from __future__ import annotations

import json
import math
import re
import time
from html import unescape
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import arxiv
import requests

from app.core.config import PROJECT_ROOT, get_settings
from app.services.ingest import run_ingest

DEFAULT_TOPIC_TERMS = (
    "retrieval augmented generation",
    "rag",
    "dense retrieval",
    "rerank",
    "reranking",
    "question answering",
    "scientific question answering",
    "scientific ai",
    "llm agent",
    "llm agents",
    "agentic",
    "tool use",
    "reasoning",
    "reflection",
    "self-reflection",
    "evaluation",
    "hallucination",
    "factuality",
    "citation",
    "attribution",
)

NOVELTY_TERMS = (
    "benchmark",
    "dataset",
    "framework",
    "agent",
    "tool",
    "evaluation",
    "hallucination",
    "factuality",
    "retrieval",
    "rerank",
    "citation",
    "scientific",
)

TERM_WEIGHTS = {
    "retrieval augmented generation": 4.5,
    "rag": 4.0,
    "dense retrieval": 3.5,
    "rerank": 3.0,
    "reranking": 3.0,
    "scientific question answering": 3.5,
    "scientific ai": 3.0,
    "hallucination": 3.5,
    "factuality": 3.5,
    "citation": 3.0,
    "attribution": 3.0,
    "agentic": 2.4,
    "llm agent": 2.4,
    "llm agents": 2.4,
    "reasoning": 1.2,
    "evaluation": 2.0,
    "reflection": 2.2,
    "self-reflection": 2.2,
    "tool use": 2.0,
}

TARGET_CATEGORY_WEIGHTS = {
    "cs.CL": 15.0,
    "cs.AI": 15.0,
    "cs.IR": 15.0,
    "cs.LG": 10.0,
}

MAX_PDF_BYTES = 80 * 1024 * 1024


@dataclass
class RadarConfig:
    topic_name: str = "agentic_rag_scientific_ai"
    top_k: int = 10
    window_days: int = 7
    categories: tuple[str, ...] = ("cs.CL", "cs.AI", "cs.IR", "cs.LG")
    topic_terms: tuple[str, ...] = DEFAULT_TOPIC_TERMS
    output_dir: Path = PROJECT_ROOT / "data" / "weekly_paper_radar"
    pdf_dir: Path = PROJECT_ROOT / "data" / "pdfs" / "weekly_radar"
    delete_pdfs_after_ingest: bool = True
    max_candidates: int = 240
    enrichment_pool_size: int = 50
    http_timeout_sec: int = 45
    user_agent: str = "PaperRAG/1.0 (weekly paper radar; mailto:dev@paperrag.local)"
    openalex_mailto: str | None = None

    @classmethod
    def from_settings(cls) -> "RadarConfig":
        settings = get_settings()
        categories = tuple(
            part.strip()
            for part in settings.weekly_radar_categories.split(",")
            if part.strip()
        )
        return cls(
            topic_name=settings.weekly_radar_topic,
            top_k=settings.weekly_radar_top_k,
            window_days=settings.weekly_radar_window_days,
            categories=categories or cls.categories,
            output_dir=_project_path(settings.weekly_radar_output_dir),
            pdf_dir=_project_path(settings.pdf_dir) / "weekly_radar",
            delete_pdfs_after_ingest=settings.weekly_radar_delete_pdfs,
            max_candidates=settings.weekly_radar_max_candidates,
            openalex_mailto=settings.openalex_mailto,
        )


@dataclass
class PaperCandidate:
    paper_id: str
    title: str
    authors: list[str]
    year: int
    published: datetime | None
    updated: datetime | None
    primary_category: str
    categories: list[str]
    abstract: str = ""
    pdf_url: str = ""
    entry_id: str = ""
    doi: str | None = None
    source: str = "arxiv"
    citation_count: int = 0
    influential_citation_count: int = 0
    impact_source: str | None = None


@dataclass
class ScoredPaper:
    candidate: PaperCandidate
    total_score: float
    score_breakdown: dict[str, float]
    reasons: list[str] = field(default_factory=list)


@dataclass
class WeeklyRadarResult:
    generated_at: datetime
    window_start: datetime
    window_end: datetime
    topic_name: str
    candidate_count: int
    selected: list[ScoredPaper]
    ingest_stats: dict[str, int]
    downloaded_pdfs: list[Path]
    skipped_downloads: list[dict[str, str]]
    report_json: Path | None = None
    report_markdown: Path | None = None
    ingest_metadata_json: Path | None = None


def _project_path(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _ensure_aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def normalize_paper_id(paper_id: str) -> str:
    paper_id = paper_id.rsplit("/", 1)[-1].strip()
    return re.sub(r"v\d+$", "", paper_id)


def normalize_title(title: str) -> str:
    title = re.sub(r"[^a-z0-9]+", " ", title.lower())
    return re.sub(r"\s+", " ", title).strip()


def date_window(now: datetime | None = None, days: int = 7) -> tuple[datetime, datetime]:
    end = _ensure_aware(now) or datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    return start, end


def _text_for_scoring(candidate: PaperCandidate) -> str:
    return " ".join([
        candidate.title,
        candidate.abstract,
        " ".join(candidate.categories),
        candidate.primary_category,
    ]).lower()


def _term_weight(term: str) -> float:
    configured = TERM_WEIGHTS.get(term.lower())
    if configured is not None:
        return configured
    if " " in term or "-" in term:
        return 3.0
    return 1.5


def _topic_score(candidate: PaperCandidate, config: RadarConfig) -> tuple[float, list[str]]:
    text = _text_for_scoring(candidate)
    hits = _matched_terms(text, config.topic_terms)
    raw = sum(_term_weight(term) for term in hits)
    score = min(40.0, raw * 3.2)
    return score, hits[:8]


def _contains_term(text: str, term: str) -> bool:
    return bool(_term_matches(text, term))


def _matched_terms(text: str, terms: Iterable[str]) -> list[str]:
    indexed_terms = list(enumerate(terms))
    indexed_terms.sort(key=lambda item: (-len(_term_parts(item[1])), item[0]))
    occupied: list[tuple[int, int]] = []
    hits: list[str] = []

    for _, term in indexed_terms:
        matches = _term_matches(text, term)
        if not matches:
            continue
        if all(_span_overlaps(match, occupied) for match in matches):
            continue
        hits.append(term)
        occupied.extend(matches)

    original_order = {term: index for index, term in enumerate(terms)}
    hits.sort(key=lambda term: original_order.get(term, 0))
    return hits


def _term_parts(term: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", term.lower())


def _term_matches(text: str, term: str) -> list[tuple[int, int]]:
    parts = re.findall(r"[a-z0-9]+", term.lower())
    if not parts:
        return []
    patterns: list[str] = []
    for index, part in enumerate(parts):
        escaped = re.escape(part)
        if index == len(parts) - 1 and len(part) > 3 and not part.endswith("s"):
            escaped = f"{escaped}s?"
        patterns.append(escaped)
    pattern = r"(?<![a-z0-9])" + r"[\s\-]+".join(patterns) + r"(?![a-z0-9])"
    return [match.span() for match in re.finditer(pattern, text)]


def _span_overlaps(span: tuple[int, int], occupied: list[tuple[int, int]]) -> bool:
    return any(span[0] < other[1] and other[0] < span[1] for other in occupied)


def _category_score(candidate: PaperCandidate, config: RadarConfig) -> float:
    categories = [candidate.primary_category, *candidate.categories]
    allowed = set(config.categories)
    best = 0.0
    for category in categories:
        if category in allowed:
            best = max(best, TARGET_CATEGORY_WEIGHTS.get(category, 8.0))
    return best


def _recency_score(candidate: PaperCandidate, now: datetime) -> float:
    published = _ensure_aware(candidate.published)
    if not published:
        return 6.0
    age_days = max((now - published).total_seconds() / 86400, 0.0)
    if age_days <= 2:
        return 10.0
    if age_days <= 7:
        return max(7.0, 10.0 - (age_days - 2) * 0.6)
    return max(0.0, 7.0 - (age_days - 7) * 0.5)


def _novelty_score(candidate: PaperCandidate) -> tuple[float, list[str]]:
    text = _text_for_scoring(candidate)
    hits = [term for term in NOVELTY_TERMS if _contains_term(text, term)]
    return min(10.0, len(hits) * 2.0), hits[:6]


def _impact_scores(candidates: list[PaperCandidate]) -> dict[str, float]:
    max_log = max((math.log1p(max(candidate.citation_count, 0)) for candidate in candidates), default=0.0)
    if max_log <= 0:
        return {candidate.paper_id: 0.0 for candidate in candidates}
    return {
        candidate.paper_id: round(25.0 * math.log1p(max(candidate.citation_count, 0)) / max_log, 3)
        for candidate in candidates
    }


def dedupe_candidates(candidates: Iterable[PaperCandidate]) -> list[PaperCandidate]:
    by_key: dict[str, PaperCandidate] = {}
    title_to_key: dict[str, str] = {}

    for candidate in candidates:
        candidate.paper_id = normalize_paper_id(candidate.paper_id)
        candidate.entry_id = candidate.entry_id or f"https://arxiv.org/abs/{candidate.paper_id}"
        candidate.pdf_url = candidate.pdf_url or f"https://arxiv.org/pdf/{candidate.paper_id}.pdf"
        title_key = normalize_title(candidate.title)
        key = candidate.paper_id or title_key
        if title_key in title_to_key:
            key = title_to_key[title_key]
        else:
            title_to_key[title_key] = key

        existing = by_key.get(key)
        if existing is None or _candidate_richness(candidate) > _candidate_richness(existing):
            by_key[key] = candidate

    return list(by_key.values())


def _candidate_richness(candidate: PaperCandidate) -> tuple[int, int, int, int]:
    return (
        max(candidate.citation_count, 0),
        len(candidate.abstract or ""),
        len(candidate.authors or []),
        len(candidate.categories or []),
    )


def rank_candidates(
    candidates: Iterable[PaperCandidate],
    config: RadarConfig,
    *,
    now: datetime | None = None,
) -> list[ScoredPaper]:
    now = _ensure_aware(now) or datetime.now(timezone.utc)
    candidate_list = dedupe_candidates(candidates)
    impact_by_id = _impact_scores(candidate_list)
    scored: list[ScoredPaper] = []

    for candidate in candidate_list:
        topic_score, topic_hits = _topic_score(candidate, config)
        category_score = _category_score(candidate, config)
        recency_score = _recency_score(candidate, now)
        novelty_score, novelty_hits = _novelty_score(candidate)
        impact_score = impact_by_id.get(candidate.paper_id, 0.0)
        total = round(topic_score + impact_score + category_score + recency_score + novelty_score, 3)

        reasons = []
        if topic_hits:
            reasons.append("topic: " + ", ".join(topic_hits))
        if candidate.citation_count:
            reasons.append(f"impact: {candidate.citation_count} citations via {candidate.impact_source or 'metadata'}")
        if novelty_hits:
            reasons.append("novelty: " + ", ".join(novelty_hits))

        scored.append(ScoredPaper(
            candidate=candidate,
            total_score=total,
            score_breakdown={
                "topic_relevance": round(topic_score, 3),
                "impact_signal": round(impact_score, 3),
                "category_match": round(category_score, 3),
                "recency": round(recency_score, 3),
                "novelty_signal": round(novelty_score, 3),
            },
            reasons=reasons,
        ))

    return sorted(scored, key=lambda item: (-item.total_score, item.candidate.published or datetime.min.replace(tzinfo=timezone.utc), item.candidate.title))


def _build_arxiv_query(categories: tuple[str, ...], start: datetime, end: datetime) -> str:
    category_clause = " OR ".join(f"cat:{category}" for category in categories)
    start_text = start.strftime("%Y%m%d%H%M")
    end_text = end.strftime("%Y%m%d%H%M")
    return f"({category_clause}) AND submittedDate:[{start_text} TO {end_text}]"


def fetch_recent_arxiv_candidates(
    config: RadarConfig,
    window_start: datetime,
    window_end: datetime,
) -> list[PaperCandidate]:
    """Fetch recent arXiv candidates for the configured vertical."""
    try:
        candidates = _fetch_arxiv_recent_html_candidates(config)
        if candidates:
            return candidates
    except Exception:
        pass
    try:
        return _fetch_arxiv_api_candidates(config, window_start, window_end)
    except Exception:
        return []


def _fetch_arxiv_api_candidates(
    config: RadarConfig,
    window_start: datetime,
    window_end: datetime,
) -> list[PaperCandidate]:
    search = arxiv.Search(
        query=_build_arxiv_query(config.categories, window_start, window_end),
        max_results=config.max_candidates,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=0)
    candidates: list[PaperCandidate] = []

    for result in client.results(search):
        paper_id = normalize_paper_id(result.entry_id.split("/abs/")[-1])
        categories = list(result.categories or [])
        candidates.append(PaperCandidate(
            paper_id=paper_id,
            title=result.title,
            authors=[str(author) for author in (result.authors or [])],
            year=result.published.year if result.published else 0,
            published=_ensure_aware(result.published),
            updated=_ensure_aware(result.updated),
            primary_category=result.primary_category or (categories[0] if categories else ""),
            categories=categories,
            abstract=(result.summary or "").strip(),
            pdf_url=result.pdf_url or f"https://arxiv.org/pdf/{paper_id}.pdf",
            entry_id=result.entry_id or f"https://arxiv.org/abs/{paper_id}",
            doi=result.doi,
        ))

    return dedupe_candidates(candidates)


def _fetch_arxiv_recent_html_candidates(config: RadarConfig) -> list[PaperCandidate]:
    candidates: list[PaperCandidate] = []
    for category in config.categories:
        try:
            html_text = _fetch_recent_html_page(category, config)
            candidates.extend(_parse_arxiv_recent_html(html_text, category))
        except Exception:
            continue
    return dedupe_candidates(candidates)[: config.max_candidates]


def _fetch_recent_html_page(category: str, config: RadarConfig) -> str:
    show = _arxiv_recent_show_value(config.max_candidates)
    errors: list[Exception] = []
    for host in ("https://export.arxiv.org", "https://arxiv.org"):
        url = f"{host}/list/{category}/recent?show={show}"
        try:
            response = requests.get(
                url,
                headers={"User-Agent": config.user_agent},
                timeout=config.http_timeout_sec,
            )
            response.raise_for_status()
            return response.text
        except Exception as exc:
            errors.append(exc)
    raise RuntimeError(f"failed to fetch arXiv recent page for {category}: {errors[-1]}")


def _arxiv_recent_show_value(max_candidates: int) -> int:
    if max_candidates <= 25:
        return 25
    if max_candidates <= 400:
        return 100
    return 1000


def _parse_arxiv_recent_html(html_text: str, fallback_category: str) -> list[PaperCandidate]:
    pairs = re.findall(r"(<dt\b.*?</dt>)\s*<dd\b.*?>(.*?)</dd>", html_text, flags=re.S | re.I)
    candidates: list[PaperCandidate] = []
    for dt_html, dd_html in pairs:
        id_match = re.search(r"/abs/([^\"'>\s]+)", dt_html)
        if not id_match:
            continue
        paper_id = normalize_paper_id(id_match.group(1))
        title = _extract_recent_field(dd_html, "list-title")
        title = re.sub(r"^Title:\s*", "", title, flags=re.I).strip()
        if not title:
            continue
        authors_text = _extract_recent_field(dd_html, "list-authors")
        authors_text = re.sub(r"^Authors?:\s*", "", authors_text, flags=re.I).strip()
        authors = [part.strip() for part in authors_text.split(",") if part.strip()]
        subjects = _extract_recent_field(dd_html, "list-subjects")
        categories = re.findall(r"\(([a-z-]+\.[A-Z]{2})\)", subjects)
        if not categories:
            categories = [fallback_category]
        primary_category = categories[0]
        candidates.append(PaperCandidate(
            paper_id=paper_id,
            title=title,
            authors=authors,
            year=_year_from_arxiv_id(paper_id),
            published=None,
            updated=None,
            primary_category=primary_category,
            categories=categories,
            abstract="",
            pdf_url=f"https://arxiv.org/pdf/{paper_id}.pdf",
            entry_id=f"https://arxiv.org/abs/{paper_id}",
        ))
    return candidates


def _extract_recent_field(block: str, class_name: str) -> str:
    match = re.search(
        rf"<div[^>]+class=['\"][^'\"]*{re.escape(class_name)}[^'\"]*['\"][^>]*>(.*?)</div>",
        block,
        flags=re.S | re.I,
    )
    if not match:
        return ""
    text = re.sub(r"<[^>]+>", " ", match.group(1))
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _year_from_arxiv_id(paper_id: str) -> int:
    match = re.match(r"(\d{2})(\d{2})\.", paper_id)
    if not match:
        return 0
    year = int(match.group(1))
    return 2000 + year if year < 90 else 1900 + year


def enrich_openalex_citations(candidates: Iterable[PaperCandidate], config: RadarConfig) -> None:
    """Best-effort citation enrichment. Failure never blocks the weekly job."""
    if not config.openalex_mailto:
        return
    session = requests.Session()
    headers = {"User-Agent": config.user_agent}

    for candidate in candidates:
        try:
            params: dict[str, str | int] = {"search": candidate.title, "per-page": 5}
            if config.openalex_mailto:
                params["mailto"] = config.openalex_mailto
            response = session.get(
                "https://api.openalex.org/works",
                params=params,
                headers=headers,
                timeout=10,
            )
            if response.status_code != 200:
                continue
            works = response.json().get("results", [])
            best = _best_openalex_match(candidate.title, works)
            if not best:
                continue
            cited_by = int(best.get("cited_by_count") or 0)
            if cited_by > candidate.citation_count:
                candidate.citation_count = cited_by
                candidate.impact_source = "OpenAlex"
        except Exception:
            continue
        time.sleep(0.1)


def _best_openalex_match(title: str, works: list[dict[str, Any]]) -> dict[str, Any] | None:
    title_key = normalize_title(title)
    best_work: dict[str, Any] | None = None
    best_ratio = 0.0
    for work in works:
        work_title = work.get("display_name") or work.get("title") or ""
        ratio = SequenceMatcher(None, title_key, normalize_title(work_title)).ratio()
        if ratio > best_ratio:
            best_work = work
            best_ratio = ratio
    return best_work if best_ratio >= 0.88 else None


def download_pdf(candidate: PaperCandidate, pdf_dir: Path, config: RadarConfig) -> Path:
    pdf_dir.mkdir(parents=True, exist_ok=True)
    paper_id = normalize_paper_id(candidate.paper_id)
    pdf_url = candidate.pdf_url or f"https://arxiv.org/pdf/{paper_id}.pdf"
    target = pdf_dir / f"{paper_id}.pdf"
    tmp_target = target.with_suffix(".pdf.part")

    headers = {"User-Agent": config.user_agent}
    with requests.get(pdf_url, headers=headers, timeout=config.http_timeout_sec, stream=True) as response:
        response.raise_for_status()
        downloaded = 0
        with tmp_target.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 512):
                if not chunk:
                    continue
                downloaded += len(chunk)
                if downloaded > MAX_PDF_BYTES:
                    raise RuntimeError(f"pdf too large: {paper_id}")
                fh.write(chunk)

    if tmp_target.stat().st_size < 1024:
        tmp_target.unlink(missing_ok=True)
        raise RuntimeError(f"pdf too small or empty: {paper_id}")
    tmp_target.replace(target)
    return target


def to_ingest_record(candidate: PaperCandidate, pdf_path: Path) -> dict[str, Any]:
    paper_id = normalize_paper_id(candidate.paper_id)
    categories = list(dict.fromkeys([*candidate.categories, "weekly_agentic_rag_scientific_ai"]))
    published = _ensure_aware(candidate.published)
    updated = _ensure_aware(candidate.updated)
    return {
        "paper_id": paper_id,
        "title": candidate.title,
        "authors": candidate.authors,
        "year": candidate.year or (published.year if published else 0),
        "published": published.isoformat() if published else None,
        "updated": updated.isoformat() if updated else None,
        "primary_category": (candidate.primary_category or "")[:32],
        "categories": categories,
        "doi": candidate.doi,
        "abstract": candidate.abstract,
        "pdf_url": f"https://arxiv.org/pdf/{paper_id}.pdf",
        "entry_id": f"https://arxiv.org/abs/{paper_id}",
        "pdf_path": str(pdf_path),
        "corpus_bucket": "weekly_agentic_rag_scientific_ai",
        "importance_reason": "Weekly radar selection for Agentic RAG for Scientific AI.",
        "corpus_score": None,
        "corpus_source": "weekly_radar_arxiv",
    }


def _result_dict(result: WeeklyRadarResult) -> dict[str, Any]:
    return {
        "generated_at": result.generated_at.isoformat(),
        "window_start": result.window_start.isoformat(),
        "window_end": result.window_end.isoformat(),
        "topic_name": result.topic_name,
        "candidate_count": result.candidate_count,
        "ingest_stats": result.ingest_stats,
        "downloaded_pdfs": [str(path) for path in result.downloaded_pdfs],
        "skipped_downloads": result.skipped_downloads,
        "ingest_metadata_json": str(result.ingest_metadata_json) if result.ingest_metadata_json else None,
        "selected": [_scored_dict(scored) for scored in result.selected],
    }


def _scored_dict(scored: ScoredPaper) -> dict[str, Any]:
    candidate = scored.candidate
    return {
        "paper_id": normalize_paper_id(candidate.paper_id),
        "title": candidate.title,
        "authors": candidate.authors,
        "year": candidate.year,
        "published": _ensure_aware(candidate.published).isoformat() if candidate.published else None,
        "primary_category": candidate.primary_category,
        "categories": candidate.categories,
        "abstract": candidate.abstract,
        "pdf_url": candidate.pdf_url,
        "entry_id": candidate.entry_id,
        "citation_count": candidate.citation_count,
        "impact_source": candidate.impact_source,
        "total_score": scored.total_score,
        "score_breakdown": scored.score_breakdown,
        "reasons": scored.reasons,
    }


def _week_slug(result: WeeklyRadarResult) -> str:
    topic = re.sub(r"[^a-z0-9]+", "-", result.topic_name.lower()).strip("-")
    return f"{result.window_end.date().isoformat()}-{topic}-top{len(result.selected)}"


def write_reports(result: WeeklyRadarResult, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = _week_slug(result)
    json_path = output_dir / f"{slug}.json"
    md_path = output_dir / f"{slug}.md"

    json_path.write_text(json.dumps(_result_dict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown_report(result), encoding="utf-8")
    result.report_json = json_path
    result.report_markdown = md_path
    return json_path, md_path


def _markdown_report(result: WeeklyRadarResult) -> str:
    lines = [
        f"# Weekly Paper Radar: {result.topic_name}",
        "",
        f"- Window: {result.window_start.date().isoformat()} to {result.window_end.date().isoformat()}",
        f"- Candidates: {result.candidate_count}",
        f"- Selected: {len(result.selected)}",
        f"- Ingest stats: `{json.dumps(result.ingest_stats, ensure_ascii=False)}`",
        "",
    ]
    for index, scored in enumerate(result.selected, start=1):
        candidate = scored.candidate
        lines.extend([
            f"## {index}. {candidate.title}",
            "",
            f"- arXiv: [{normalize_paper_id(candidate.paper_id)}]({candidate.entry_id or f'https://arxiv.org/abs/{normalize_paper_id(candidate.paper_id)}'})",
            f"- Category: {candidate.primary_category}",
            f"- Score: {scored.total_score}",
            f"- Breakdown: `{json.dumps(scored.score_breakdown, ensure_ascii=False)}`",
            f"- Authors: {', '.join(candidate.authors[:8])}",
        ])
        if candidate.citation_count:
            lines.append(f"- Impact: {candidate.citation_count} citations via {candidate.impact_source or 'metadata'}")
        if scored.reasons:
            lines.append(f"- Reasons: {'; '.join(scored.reasons)}")
        if candidate.abstract:
            lines.extend(["", candidate.abstract.strip()])
        lines.append("")
    if result.skipped_downloads:
        lines.extend(["## Skipped Downloads", ""])
        for item in result.skipped_downloads:
            lines.append(f"- {item.get('paper_id')}: {item.get('error')}")
    return "\n".join(lines).strip() + "\n"


def _write_ingest_metadata(records: list[dict[str, Any]], result: WeeklyRadarResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{_week_slug(result)}.metadata.json"
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run_weekly_radar(
    *,
    config: RadarConfig | None = None,
    dry_run: bool = False,
    no_ingest: bool = False,
    now: datetime | None = None,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
) -> WeeklyRadarResult:
    config = config or RadarConfig.from_settings()
    generated_at = _ensure_aware(now) or datetime.now(timezone.utc)
    if window_start is None or window_end is None:
        window_start, window_end = date_window(generated_at, config.window_days)
    else:
        window_start = _ensure_aware(window_start) or window_start
        window_end = _ensure_aware(window_end) or window_end

    candidates = fetch_recent_arxiv_candidates(config, window_start, window_end)
    initial_ranked = rank_candidates(candidates, config, now=generated_at)
    enrichment_limit = min(config.enrichment_pool_size, max(config.top_k * 3, config.top_k))
    enrichment_pool = [item.candidate for item in initial_ranked[:enrichment_limit]]
    enrich_openalex_citations(enrichment_pool, config)
    ranked = rank_candidates(candidates, config, now=generated_at)

    selected: list[ScoredPaper] = []
    ingest_records: list[dict[str, Any]] = []
    downloaded_pdfs: list[Path] = []
    skipped_downloads: list[dict[str, str]] = []
    ingest_stats = {"ok": 0, "skipped": 0, "failed": 0, "total": 0}

    if dry_run or no_ingest:
        selected = ranked[: config.top_k]
    else:
        for scored in ranked:
            if len(selected) >= config.top_k:
                break
            try:
                pdf_path = download_pdf(scored.candidate, config.pdf_dir, config)
                record = to_ingest_record(scored.candidate, pdf_path)
                record["corpus_score"] = scored.total_score
                ingest_records.append(record)
                downloaded_pdfs.append(pdf_path)
                selected.append(scored)
            except Exception as exc:
                skipped_downloads.append({
                    "paper_id": normalize_paper_id(scored.candidate.paper_id),
                    "title": scored.candidate.title,
                    "error": f"{type(exc).__name__}: {exc}",
                })

        ingest_stats = {"ok": 0, "skipped": 0, "failed": 0, "total": len(ingest_records)}

    result = WeeklyRadarResult(
        generated_at=generated_at,
        window_start=window_start,
        window_end=window_end,
        topic_name=config.topic_name,
        candidate_count=len(dedupe_candidates(candidates)),
        selected=selected,
        ingest_stats=ingest_stats,
        downloaded_pdfs=downloaded_pdfs,
        skipped_downloads=skipped_downloads,
    )

    if ingest_records and not dry_run and not no_ingest:
        metadata_path = _write_ingest_metadata(ingest_records, result, config.output_dir)
        result.ingest_metadata_json = metadata_path
        result.ingest_stats = run_ingest(metadata_json=str(metadata_path), force=False)

    write_reports(result, config.output_dir)

    if config.delete_pdfs_after_ingest and not dry_run and not no_ingest:
        for path in downloaded_pdfs:
            path.unlink(missing_ok=True)

    return result
