"""arXiv ID import helpers for Cite Scope uploads."""
from __future__ import annotations

from datetime import datetime
from html import unescape
import re
from pathlib import Path
from typing import Any

import arxiv
import requests

from app.core.config import get_settings

settings = get_settings()

_ARXIV_ID_RE = re.compile(r"(?P<id>\d{4}\.\d{4,6})(?:v\d+)?", re.IGNORECASE)
_USER_AGENT = "CiteScope/0.1 (https://github.com/; mailto:dev@citescope.local)"
_HTTP_TIMEOUT = 60
_METADATA_TIMEOUT = 20

_TOPIC_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "agents_reasoning",
        (
            "agent",
            "agents",
            "tool use",
            "tool-use",
            "planning",
            "reasoning",
            "self-evolving",
            "self improvement",
            "multi-agent",
            "workflow",
        ),
    ),
    (
        "rag_ir_memory",
        (
            "retrieval",
            "rag",
            "rerank",
            "reranking",
            "dense retrieval",
            "vector search",
            "information retrieval",
            "memory",
            "bm25",
        ),
    ),
    (
        "llm_transformer",
        (
            "transformer",
            "large language model",
            "llm",
            "pretraining",
            "pre-train",
            "scaling",
            "instruction tuning",
            "language model",
        ),
    ),
    (
        "evaluation_factuality",
        (
            "benchmark",
            "evaluation",
            "evaluate",
            "factuality",
            "hallucination",
            "faithfulness",
            "truthfulness",
            "attribution",
        ),
    ),
    (
        "alignment_safety_eval",
        (
            "alignment",
            "safety",
            "rlhf",
            "preference",
            "harmless",
            "red team",
            "constitutional",
            "reward model",
        ),
    ),
    (
        "multimodal_generative",
        (
            "multimodal",
            "vision-language",
            "vision language",
            "diffusion",
            "image generation",
            "text-to-image",
            "video generation",
            "vlm",
        ),
    ),
    (
        "deep_learning",
        (
            "deep learning",
            "neural network",
            "optimization",
            "gradient",
            "representation learning",
            "cnn",
            "rnn",
            "sequence model",
        ),
    ),
)


class ArxivImportError(RuntimeError):
    """Raised when arXiv metadata or PDF import cannot complete."""


def normalize_arxiv_id(raw: str) -> str:
    """Extract a modern numeric arXiv ID from a raw ID or arXiv URL."""
    value = (raw or "").strip()
    match = _ARXIV_ID_RE.search(value)
    if not match:
        raise ValueError(f"invalid arXiv ID: {raw}")
    return match.group("id")


def classify_topic_bucket(title: str, abstract: str | None, categories: list[str]) -> str | None:
    """Map paper metadata to one corpus overview bucket when possible."""
    category_text = " ".join(categories).lower()
    haystack = f"{title} {abstract or ''} {category_text}".lower()

    scores: dict[str, int] = {}
    if "cs.ir" in category_text:
        scores["rag_ir_memory"] = scores.get("rag_ir_memory", 0) + 2
    if "cs.cv" in category_text:
        scores["multimodal_generative"] = scores.get("multimodal_generative", 0) + 1

    for key, keywords in _TOPIC_KEYWORDS:
        for keyword in keywords:
            if keyword in haystack:
                scores[key] = scores.get(key, 0) + 1

    if not scores:
        return None
    return max(scores.items(), key=lambda item: item[1])[0]


def _clean_text(value: str | None) -> str:
    return " ".join((value or "").split())


def build_arxiv_record(result: Any) -> dict:
    """Convert an arxiv.Result-like object to the metadata shape used by ingest."""
    paper_id = normalize_arxiv_id(result.get_short_id())
    categories = list(getattr(result, "categories", []) or [])
    title = _clean_text(getattr(result, "title", ""))
    abstract = _clean_text(getattr(result, "summary", ""))
    bucket = classify_topic_bucket(title, abstract, categories)
    if bucket and bucket not in categories:
        categories = [*categories, bucket]

    published = getattr(result, "published", None)
    updated = getattr(result, "updated", None)

    return {
        "paper_id": paper_id,
        "title": title,
        "authors": [author.name for author in getattr(result, "authors", [])],
        "year": published.year if published else 0,
        "published": published.isoformat() if published else None,
        "updated": updated.isoformat() if updated else None,
        "primary_category": getattr(result, "primary_category", "") or "",
        "categories": categories,
        "doi": getattr(result, "doi", None),
        "abstract": abstract,
        "pdf_url": getattr(result, "pdf_url", None) or f"https://arxiv.org/pdf/{paper_id}",
        "pdf_path": None,
        "entry_id": getattr(result, "entry_id", None),
    }


def _extract_meta_values(html: str, name: str) -> list[str]:
    pattern = re.compile(
        rf'<meta\s+[^>]*name=["\']{re.escape(name)}["\'][^>]*content=["\'](?P<content>.*?)["\'][^>]*>',
        re.IGNORECASE | re.DOTALL,
    )
    return [_clean_text(unescape(match.group("content"))) for match in pattern.finditer(html)]


def _extract_subject_categories(html: str) -> list[str]:
    match = re.search(
        r'<td[^>]+class=["\'][^"\']*subjects[^"\']*["\'][^>]*>(?P<subjects>.*?)</td>',
        html,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return []
    subjects = re.sub(r"<[^>]+>", " ", match.group("subjects"))
    return re.findall(r"\(([a-z-]+\.[A-Z]{2})\)", unescape(subjects))


def _build_record_from_abs_page(arxiv_id: str, html: str) -> dict:
    id_values = _extract_meta_values(html, "citation_arxiv_id")
    paper_id = normalize_arxiv_id(id_values[0] if id_values else arxiv_id)
    if paper_id != arxiv_id:
        raise ArxivImportError(f"arXiv ID mismatch: requested={arxiv_id}, page={paper_id}")

    title_values = _extract_meta_values(html, "citation_title")
    abstract_values = _extract_meta_values(html, "citation_abstract")
    date_values = _extract_meta_values(html, "citation_date")
    pdf_values = _extract_meta_values(html, "citation_pdf_url")
    doi_values = _extract_meta_values(html, "citation_doi")
    authors = _extract_meta_values(html, "citation_author")
    categories = _extract_subject_categories(html)
    title = title_values[0] if title_values else f"arXiv:{paper_id}"
    abstract = abstract_values[0] if abstract_values else ""

    published = None
    year = 0
    if date_values:
        try:
            parsed_date = datetime.strptime(date_values[0], "%Y/%m/%d")
            published = parsed_date.date().isoformat()
            year = parsed_date.year
        except ValueError:
            pass

    bucket = classify_topic_bucket(title, abstract, categories)
    if bucket and bucket not in categories:
        categories = [*categories, bucket]

    return {
        "paper_id": paper_id,
        "title": title,
        "authors": authors,
        "year": year,
        "published": published,
        "updated": None,
        "primary_category": categories[0] if categories else "",
        "categories": categories,
        "doi": doi_values[0] if doi_values else None,
        "abstract": abstract,
        "pdf_url": pdf_values[0] if pdf_values else f"https://arxiv.org/pdf/{paper_id}",
        "pdf_path": None,
        "entry_id": f"https://arxiv.org/abs/{paper_id}",
    }


def fetch_arxiv_record(arxiv_id: str) -> dict:
    """Fetch metadata for one arXiv ID from the official abs page."""
    normalized = normalize_arxiv_id(arxiv_id)
    url = f"https://arxiv.org/abs/{normalized}"
    try:
        response = requests.get(
            url,
            headers={"User-Agent": _USER_AGENT},
            timeout=_METADATA_TIMEOUT,
        )
        response.raise_for_status()
        return _build_record_from_abs_page(normalized, response.text)
    except Exception as exc:
        raise ArxivImportError(
            f"arXiv metadata fetch failed for {normalized}: {type(exc).__name__}: {exc}"
        ) from exc


def download_arxiv_pdf(record: dict) -> dict:
    """Download the official arXiv PDF and return a record with pdf_path set."""
    paper_id = record["paper_id"]
    pdf_url = record.get("pdf_url") or f"https://arxiv.org/pdf/{paper_id}"
    pdf_dir = Path(settings.pdf_dir)
    project_root = Path(settings.data_dir).parent
    pdf_dir.mkdir(parents=True, exist_ok=True)
    out_path = pdf_dir / f"{paper_id}.pdf"

    if not (out_path.exists() and out_path.stat().st_size > 10_000):
        tmp = out_path.with_suffix(".pdf.part")
        try:
            with requests.get(
                pdf_url,
                headers={"User-Agent": _USER_AGENT},
                timeout=_HTTP_TIMEOUT,
                stream=True,
            ) as response:
                response.raise_for_status()
                with tmp.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=32768):
                        if chunk:
                            handle.write(chunk)
            if tmp.stat().st_size <= 10_000:
                raise ArxivImportError(f"downloaded PDF is too small: {paper_id}")
            tmp.rename(out_path)
        finally:
            if tmp.exists():
                tmp.unlink()

    return {
        **record,
        "pdf_path": str(out_path.relative_to(project_root)),
    }
