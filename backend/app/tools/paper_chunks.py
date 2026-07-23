"""Tool: get chunks for a specific paper."""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.models.paper import Chunk


def load_paper_chunks(db: Session, paper_id: str, max_chunks: int = 10) -> list[Chunk]:
    """Load chunks for a specific paper in document order."""
    return (
        db.query(Chunk)
        .filter(Chunk.paper_id == paper_id)
        .order_by(Chunk.chunk_index)
        .limit(max_chunks)
        .all()
    )


def format_paper_chunks(chunks: list[Chunk], paper_id: str) -> str:
    """Format loaded chunks as a compact model-readable transcript."""
    if not chunks:
        return f"No chunks found for paper {paper_id}."
    parts = []
    for c in chunks:
        parts.append(f"[page={c.page_num} | chunk={c.chunk_index}]\n{c.chunk_text[:500]}")
    return "\n\n---\n\n".join(parts)


def get_paper_chunks(db: Session, paper_id: str, max_chunks: int = 10) -> str:
    """Get text chunks for a specific paper, ordered by chunk_index."""
    chunks = load_paper_chunks(db, paper_id, max_chunks)
    return format_paper_chunks(chunks, paper_id)
