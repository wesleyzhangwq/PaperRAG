"""Export paper-level metadata from the current Qdrant collection.

The question generator expects paper metadata, but retrieval evaluation must
use the same corpus that Qdrant can actually retrieve. This script aggregates
chunk payloads from Qdrant into a compact metadata JSON file.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import get_settings
from qdrant_client import QdrantClient

SPACE_RE = re.compile(r"\s+")
SECRET_RE = re.compile(r"sk-[A-Za-z0-9_-]{10,}")


def normalize_text(text: str) -> str:
    clean = SECRET_RE.sub("[REDACTED_SECRET]", str(text or ""))
    return SPACE_RE.sub(" ", clean.strip())


def _number_or_default(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _chunk_sort_key(chunk: dict) -> tuple[int, int, str]:
    metadata = chunk.get("metadata") or {}
    return (
        _number_or_default(metadata.get("page_num"), 999_999),
        _number_or_default(metadata.get("chunk_index"), 999_999),
        str(metadata.get("source_chunk_id") or ""),
    )


def paper_from_chunks(
    paper_id: str,
    chunks: list[dict],
    *,
    abstract_chunk_limit: int = 8,
    abstract_char_limit: int = 4000,
) -> dict:
    if not chunks:
        raise ValueError(f"{paper_id} has no chunks")

    chunks = sorted(chunks, key=_chunk_sort_key)
    first_metadata = chunks[0].get("metadata") or {}
    title = normalize_text(first_metadata.get("title") or paper_id)
    primary_category = str(first_metadata.get("primary_category") or "unknown")
    year = first_metadata.get("year")
    doi = first_metadata.get("doi")

    abstract_parts: list[str] = []
    seen_text: set[str] = set()
    for chunk in chunks:
        text = normalize_text(chunk.get("text") or "")
        if not text or text in seen_text:
            continue
        abstract_parts.append(text)
        seen_text.add(text)
        if len(abstract_parts) >= abstract_chunk_limit:
            break
    abstract = normalize_text(" ".join(abstract_parts))[:abstract_char_limit]

    return {
        "paper_id": paper_id,
        "title": title,
        "authors": [],
        "year": year,
        "published": f"{year}-01-01" if year else None,
        "updated": None,
        "primary_category": primary_category,
        "categories": [primary_category] if primary_category != "unknown" else [],
        "doi": doi,
        "abstract": abstract,
        "pdf_url": f"https://arxiv.org/pdf/{paper_id}",
        "entry_id": f"https://arxiv.org/abs/{paper_id}",
        "pdf_path": None,
        "chunk_count": len(chunks),
        "abstract_source": "qdrant_payload_first_chunks",
    }


def collect_papers_from_payloads(payloads: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for payload in payloads:
        metadata = payload.get("metadata") or {}
        paper_id = str(metadata.get("paper_id") or "").strip()
        if not paper_id:
            continue
        grouped[paper_id].append(
            {
                "metadata": metadata,
                "text": payload.get("text") or "",
            }
        )

    papers = [
        paper_from_chunks(paper_id, chunks)
        for paper_id, chunks in sorted(grouped.items())
    ]
    return [paper for paper in papers if paper.get("abstract")]


def scroll_payloads(
    client: QdrantClient,
    *,
    collection_name: str,
    page_size: int,
) -> list[dict]:
    payloads: list[dict] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=page_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        payloads.extend((point.payload or {}) for point in points)
        if offset is None:
            break
    return payloads


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Qdrant corpus metadata")
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "eval/datasets/qdrant_papers.json"),
    )
    parser.add_argument("--collection", type=str, default=None)
    parser.add_argument("--page-size", type=int, default=512)
    args = parser.parse_args()

    settings = get_settings()
    collection = args.collection or settings.qdrant_collection
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        check_compatibility=False,
    )
    payloads = scroll_payloads(
        client,
        collection_name=collection,
        page_size=max(1, args.page_size),
    )
    papers = collect_papers_from_payloads(payloads)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(papers, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"Exported {len(papers)} papers from {len(payloads)} chunks in {collection} -> {output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
