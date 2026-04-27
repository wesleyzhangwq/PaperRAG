"""Ingest service: parse PDFs → chunk → embed → persist (MySQL + Qdrant).

Idempotent per paper_id. Safe to rerun; already-ingested papers are skipped.
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path

from sqlalchemy.orm import Session
from tqdm import tqdm

from app.core.config import get_settings
from app.db.mysql import SessionLocal, init_db
from app.db.vector import get_vector_store
from app.models.paper import Chunk, Paper
from app.utils.chunker import chunk_pages
from app.utils.pdf import extract_pages

settings = get_settings()


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
    paper.pdf_path = record.get("pdf_path")
    paper.entry_id = record.get("entry_id")
    paper.published = record.get("published")
    paper.updated = record.get("updated")
    db.flush()
    return paper


def _ingest_one(db: Session, record: dict, force: bool = False) -> tuple[str, str]:
    """Returns (paper_id, status). status in {'ok','skipped','failed'}."""
    paper = _upsert_paper(db, record)

    if not force and paper.ingest_status == "ok" and paper.num_chunks > 0:
        return paper.paper_id, "skipped"

    pdf_path = record.get("pdf_path")
    if not pdf_path:
        paper.ingest_status = "failed"
        paper.ingest_error = "no pdf_path"
        return paper.paper_id, "failed"

    abs_pdf = Path(pdf_path)
    if not abs_pdf.is_absolute():
        abs_pdf = Path(settings.data_dir).parent / pdf_path
    if not abs_pdf.exists():
        paper.ingest_status = "failed"
        paper.ingest_error = f"pdf not found: {abs_pdf}"
        return paper.paper_id, "failed"

    try:
        pages = extract_pages(abs_pdf)
        if not pages:
            raise RuntimeError("no text extracted")
        chunks = chunk_pages(pages)
        if not chunks:
            raise RuntimeError("no chunks produced")
    except Exception as e:
        paper.ingest_status = "failed"
        paper.ingest_error = f"{type(e).__name__}: {e}"
        return paper.paper_id, "failed"

    # Idempotent rerun: refresh Qdrant first, only mutate MySQL chunks after vectors persist.
    # Otherwise a failed embed still gets committed and leaves MySQL without matching vectors.
    vs = get_vector_store()
    old_ids = [c.chunk_id for c in db.query(Chunk).filter(Chunk.paper_id == paper.paper_id).all()]
    if old_ids:
        try:
            vs.delete(ids=old_ids)
        except Exception:
            pass

    # Build chunks (in memory only until Qdrant upsert succeeds)
    chunk_ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []
    db_chunks: list[Chunk] = []

    for ch in chunks:
        chunk_id = f"{paper.paper_id}::{ch.chunk_index}"
        chunk_ids.append(chunk_id)
        texts.append(ch.text)
        metadatas.append({
            "paper_id": paper.paper_id,
            "chunk_index": ch.chunk_index,
            "page_num": ch.page_num or 0,
            "title": paper.title[:500],
            "year": paper.year,
            "primary_category": paper.primary_category,
            "doi": paper.doi or "",
        })
        db_chunks.append(Chunk(
            chunk_id=chunk_id,
            paper_id=paper.paper_id,
            chunk_index=ch.chunk_index,
            chunk_text=ch.text,
            page_num=ch.page_num,
            n_tokens=len(ch.text) // 4,
        ))

    try:
        vs.add_texts(texts=texts, metadatas=metadatas, ids=chunk_ids)
    except Exception as e:
        paper.ingest_status = "failed"
        paper.ingest_error = f"embed: {type(e).__name__}: {e}"
        return paper.paper_id, "failed"

    db.query(Chunk).filter(Chunk.paper_id == paper.paper_id).delete(synchronize_session=False)
    db.add_all(db_chunks)

    paper.num_chunks = len(chunks)
    paper.ingest_status = "ok"
    paper.ingest_error = None
    return paper.paper_id, "ok"


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
