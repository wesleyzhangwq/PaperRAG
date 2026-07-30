"""Unified ingestion: parse → normalize → chunk → index → persist.

Both arXiv PDFs and user-uploaded heterogeneous documents enter this service.
Qdrant is written before MySQL chunk replacement, so a failed embedding/index
operation cannot erase the last committed relational corpus snapshot.
"""
from __future__ import annotations

import hashlib
import json
import traceback
from pathlib import Path
from typing import Callable

from sqlalchemy.orm import Session
from tqdm import tqdm

from app.core.config import get_settings
from app.db.mysql import SessionLocal, init_db
from app.db.qdrant import get_qdrant_vector_store
from app.models.paper import Chunk, Paper
from app.services.document_parser import media_type_for_filename, parse_document
from app.utils.chunker import chunk_document_blocks

settings = get_settings()
ProgressCallback = Callable[[str, int, str], None]


def content_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _progress(
    callback: ProgressCallback | None,
    stage: str,
    percent: int,
    message: str,
) -> None:
    if callback is not None:
        callback(stage, percent, message)


def _source_path(record: dict) -> Path:
    raw_path = record.get("source_path") or record.get("pdf_path")
    if not raw_path:
        raise ValueError("record has no source_path or pdf_path")
    resolved = Path(str(raw_path))
    if not resolved.is_absolute():
        resolved = Path(settings.data_dir).parent / resolved
    if not resolved.exists():
        raise FileNotFoundError(f"source file not found: {resolved}")
    return resolved


def _upsert_paper(db: Session, record: dict) -> Paper:
    paper = db.query(Paper).filter(Paper.paper_id == record["paper_id"]).one_or_none()
    if paper is None:
        paper = Paper(paper_id=record["paper_id"])
        db.add(paper)
    paper.title = record.get("title") or ""
    paper.authors = record.get("authors")
    paper.year = record.get("year") or 0
    paper.primary_category = record.get("primary_category") or ""
    paper.categories = record.get("categories")
    paper.doi = record.get("doi")
    paper.abstract = record.get("abstract")
    paper.pdf_url = record.get("pdf_url")
    paper.pdf_path = record.get("pdf_path") or record.get("source_path")
    paper.entry_id = record.get("entry_id")
    paper.published = record.get("published")
    paper.updated = record.get("updated")
    paper.source_kind = record.get("source_kind") or "arxiv"
    paper.media_type = record.get("media_type")
    paper.content_hash = record.get("content_hash")
    paper.original_filename = record.get("original_filename")
    paper.ingest_metadata = record.get("ingest_metadata")
    db.flush()
    return paper


def _ingest_one(
    db: Session,
    record: dict,
    force: bool = False,
    *,
    progress: ProgressCallback | None = None,
) -> tuple[str, str]:
    """Ingest one source and return ``(paper_id, ok|skipped)``.

    Failures raise and leave the last committed MySQL chunks untouched. The
    caller owns commit/rollback and records job-level failure diagnostics.
    """
    paper_id = str(record["paper_id"])
    existing = db.query(Paper).filter(Paper.paper_id == paper_id).one_or_none()
    if (
        not force
        and existing is not None
        and existing.ingest_status == "ok"
        and existing.num_chunks > 0
    ):
        return paper_id, "skipped"

    source_path = _source_path(record)
    _progress(progress, "parsing", 25, "parsing_document")
    parsed = parse_document(source_path)

    _progress(progress, "normalizing", 45, "normalizing_modalities")
    chunks = chunk_document_blocks(parsed.blocks)
    if not chunks:
        raise RuntimeError("no chunks produced after normalization")

    effective_record = {
        **record,
        "title": record.get("title") or parsed.title or source_path.stem,
        "media_type": record.get("media_type") or media_type_for_filename(source_path.name),
        "content_hash": record.get("content_hash") or content_sha256(source_path),
        "original_filename": record.get("original_filename") or source_path.name,
        "ingest_metadata": {
            **(record.get("ingest_metadata") or {}),
            **parsed.metadata,
            "warnings": parsed.warnings,
        },
    }

    old_ids = [
        chunk.chunk_id
        for chunk in db.query(Chunk).filter(Chunk.paper_id == paper_id).all()
    ]

    _progress(progress, "chunking", 60, f"prepared_{len(chunks)}_chunks")
    chunk_ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []
    db_chunks: list[Chunk] = []

    for ch in chunks:
        chunk_id = f"{paper_id}::{ch.chunk_index}"
        chunk_ids.append(chunk_id)
        texts.append(ch.text)
        metadatas.append({
            "paper_id": paper_id,
            "chunk_index": ch.chunk_index,
            "page_num": ch.page_num or 0,
            "title": str(effective_record.get("title") or "")[:500],
            "year": int(effective_record.get("year") or 0),
            "primary_category": effective_record.get("primary_category") or "",
            "doi": effective_record.get("doi") or "",
            "source_kind": effective_record.get("source_kind") or "arxiv",
            "media_type": effective_record.get("media_type") or "",
            "content_hash": effective_record.get("content_hash") or "",
            "modality": ch.modality,
            "source_locator": ch.source_locator or {},
            "section": ch.section or "",
        })
        db_chunks.append(Chunk(
            chunk_id=chunk_id,
            paper_id=paper_id,
            chunk_index=ch.chunk_index,
            chunk_text=ch.text,
            page_num=ch.page_num,
            n_tokens=len(ch.text) // 4,
            modality=ch.modality,
            section=ch.section,
            source_locator=ch.source_locator or {},
        ))

    _progress(progress, "indexing", 75, "embedding_and_upserting_qdrant")
    vs = get_qdrant_vector_store()
    vs.add_texts(texts=texts, metadatas=metadatas, ids=chunk_ids)

    # Deterministic ids overwrite matching chunks. Delete only obsolete tail
    # ids after the replacement upsert; deletion failures are not swallowed.
    new_id_set = set(chunk_ids)
    stale_ids = [chunk_id for chunk_id in old_ids if chunk_id not in new_id_set]
    if stale_ids:
        vs.delete(ids=stale_ids)

    _progress(progress, "persisting", 90, "persisting_mysql_metadata")
    paper = _upsert_paper(db, effective_record)
    db.query(Chunk).filter(Chunk.paper_id == paper_id).delete(
        synchronize_session=False
    )
    db.add_all(db_chunks)
    paper.num_chunks = len(chunks)
    paper.ingest_status = "ok"
    paper.ingest_error = None
    db.flush()
    return paper_id, "ok"


def run_ingest(metadata_json: str | None = None, force: bool = False) -> dict:
    init_db()
    metadata_path = Path(metadata_json or settings.metadata_json)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}. Run scripts/download_arxiv.py first.")

    records = json.loads(metadata_path.read_text())
    stats = {"ok": 0, "skipped": 0, "failed": 0, "total": len(records)}

    for rec in tqdm(records, desc="Ingesting", unit="paper"):
        db = SessionLocal()
        try:
            try:
                _, status = _ingest_one(db, rec, force=force)
                stats[status] += 1
                db.commit()
            except Exception:
                db.rollback()
                traceback.print_exc()
                stats["failed"] += 1
        finally:
            db.close()

    print(f"[ingest] stats={stats}")
    return stats


__all__ = ["_ingest_one", "content_sha256", "run_ingest"]
