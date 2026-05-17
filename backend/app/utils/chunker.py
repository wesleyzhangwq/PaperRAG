"""Chunk extracted pages into overlapping windows with page attribution."""
from __future__ import annotations

from dataclasses import dataclass
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings


@dataclass
class PaperChunk:
    chunk_index: int
    text: str
    page_num: int | None


_SENTENCE_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "。", "！", "？", " ", ""]
_HEADER_PATTERNS = [
    r"^\s*abstract\s*$",
    r"^\s*introduction\s*$",
    r"^\s*related work\s*$",
    r"^\s*method(s)?\s*$",
    r"^\s*experiment(s)?\s*$",
    r"^\s*conclusion(s)?\s*$",
    r"^\s*references\s*$",
]


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_noisy(text: str, min_chars: int, symbol_ratio_limit: float) -> bool:
    if len(text) < min_chars:
        return True
    letters_or_digits = sum(1 for c in text if c.isalnum())
    symbols = sum(1 for c in text if not c.isalnum() and not c.isspace())
    total = max(1, letters_or_digits + symbols)
    return (symbols / total) > symbol_ratio_limit


def _is_references_header(line: str) -> bool:
    for p in _HEADER_PATTERNS:
        if re.match(p, line, flags=re.IGNORECASE):
            return True
    return False


def _build_fulltext_with_offsets(pages: list[tuple[int, str]]) -> tuple[str, list[tuple[int, int]]]:
    parts: list[str] = []
    page_offsets: list[tuple[int, int]] = []
    cursor = 0
    for page_num, text in pages:
        cleaned = _clean_text(text)
        if not cleaned:
            continue
        page_offsets.append((cursor, page_num))
        parts.append(cleaned)
        cursor += len(cleaned) + 2
    return "\n\n".join(parts), page_offsets


def _page_from_offset(char_offset: int, page_offsets: list[tuple[int, int]]) -> int | None:
    if not page_offsets:
        return None
    page_num = page_offsets[0][1]
    for start, p in page_offsets:
        if start > char_offset:
            break
        page_num = p
    return page_num


def chunk_pages(pages: list[tuple[int, str]]) -> list[PaperChunk]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=_SENTENCE_SEPARATORS,
        length_function=len,
    )

    full_text, page_offsets = _build_fulltext_with_offsets(pages)
    if not full_text:
        return []

    pieces = splitter.split_text(full_text)
    results: list[PaperChunk] = []
    cursor = 0
    in_references = False

    for piece in pieces:
        cleaned = _clean_text(piece)
        if not cleaned:
            continue

        first_line = cleaned.splitlines()[0].strip().lower() if cleaned.splitlines() else ""
        if _is_references_header(first_line):
            in_references = True
        if settings.chunk_drop_references and in_references:
            continue
        if _is_noisy(cleaned, settings.chunk_min_chars, settings.chunk_noise_symbol_ratio):
            continue

        loc = full_text.find(cleaned, max(0, cursor - settings.chunk_overlap))
        if loc == -1:
            loc = cursor
        page_num = _page_from_offset(loc, page_offsets)
        cursor = loc + len(cleaned)

        results.append(PaperChunk(
            chunk_index=len(results),
            text=cleaned,
            page_num=page_num,
        ))
    return results
