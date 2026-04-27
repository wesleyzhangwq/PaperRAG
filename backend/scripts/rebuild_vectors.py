"""Rebuild Qdrant vector index from existing chunks in MySQL.

Usage:
    python scripts/rebuild_vectors.py
    python scripts/rebuild_vectors.py --paper-id 2401.01234
    python scripts/rebuild_vectors.py --batch-size 64
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.vector import get_vector_store  # noqa: E402
from app.models.paper import Chunk, Paper  # noqa: E402


def _build_metadata(db, paper_id: str) -> dict:
    paper = db.query(Paper).filter(Paper.paper_id == paper_id).one_or_none()
    if paper is None:
        return {
            "paper_id": paper_id,
            "title": "",
            "year": 0,
            "primary_category": "",
            "doi": "",
        }
    return {
        "paper_id": paper.paper_id,
        "title": (paper.title or "")[:500],
        "year": paper.year,
        "primary_category": paper.primary_category,
        "doi": paper.doi or "",
    }


def rebuild_vectors(paper_id: str | None = None, batch_size: int = 64) -> dict:
    from sqlalchemy.orm import sessionmaker

    from app.db.mysql import engine, init_db

    init_db()
    vs = get_vector_store()
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    try:
        query = db.query(Chunk).order_by(Chunk.paper_id, Chunk.chunk_index)
        if paper_id:
            query = query.filter(Chunk.paper_id == paper_id)
        chunks = query.all()
        if not chunks:
            return {"total_chunks": 0, "reindexed_chunks": 0, "failed_batches": 0}

        all_ids = [c.chunk_id for c in chunks]
        try:
            vs.delete(ids=all_ids)
        except Exception:
            pass

        reindexed = 0
        failed = 0
        metadata_cache: dict[str, dict] = {}

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts: list[str] = []
            ids: list[str] = []
            metadatas: list[dict] = []

            for c in batch:
                if c.paper_id not in metadata_cache:
                    metadata_cache[c.paper_id] = _build_metadata(db, c.paper_id)
                meta = dict(metadata_cache[c.paper_id])
                meta.update(
                    {
                        "chunk_index": c.chunk_index,
                        "page_num": c.page_num or 0,
                    }
                )
                texts.append(c.chunk_text)
                ids.append(c.chunk_id)
                metadatas.append(meta)

            try:
                vs.add_texts(texts=texts, metadatas=metadatas, ids=ids)
                reindexed += len(batch)
            except Exception:
                failed += 1

        return {
            "total_chunks": len(chunks),
            "reindexed_chunks": reindexed,
            "failed_batches": failed,
        }
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-id", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    stats = rebuild_vectors(paper_id=args.paper_id, batch_size=args.batch_size)
    print(f"[rebuild_vectors] stats={stats}")
    return 0 if stats["failed_batches"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
