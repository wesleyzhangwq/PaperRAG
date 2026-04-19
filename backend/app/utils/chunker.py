"""Chunk extracted pages into overlapping windows with page attribution."""
from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings


@dataclass
class PaperChunk:
    chunk_index: int
    text: str
    page_num: int | None


def chunk_pages(pages: list[tuple[int, str]]) -> list[PaperChunk]:
    """Split each page independently then globally re-index; keeps page_num per chunk."""
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", "。", "！", "？", " ", ""],
        length_function=len,
    )

    results: list[PaperChunk] = []
    for page_num, text in pages:
        for piece in splitter.split_text(text):
            piece = piece.strip()
            if len(piece) < 40:
                continue
            results.append(PaperChunk(
                chunk_index=len(results),
                text=piece,
                page_num=page_num,
            ))
    return results
