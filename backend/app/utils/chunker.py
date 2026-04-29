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

_TITLE_RE = re.compile(r"^(\d+(\.\d+)*\s+)?[A-Z][A-Za-z0-9 ,:/()\\-]{3,}$")
_LIST_RE = re.compile(r"^(\s*[-*•]|\s*\d+\.)\s+")
_FORMULA_RE = re.compile(r"(\\begin\{|\\[a-zA-Z]+|=|\\sum|\\int|\\mathbb|\\mathbf)")


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
    """
    Return full text plus page start offsets.
    page_offsets: [(start_char_offset, page_num), ...]
    """
    parts: list[str] = []
    page_offsets: list[tuple[int, int]] = []
    cursor = 0
    for page_num, text in pages:
        cleaned = _clean_text(text)
        if not cleaned:
            continue
        page_offsets.append((cursor, page_num))
        parts.append(cleaned)
        cursor += len(cleaned) + 2  # account for '\n\n'
    return "\n\n".join(parts), page_offsets


def _split_blocks_with_offsets(full_text: str) -> list[tuple[int, str]]:
    """Split full text into paragraph-like blocks with char offsets."""
    blocks: list[tuple[int, str]] = []
    cursor = 0
    for raw in re.split(r"\n\s*\n+", full_text):
        block = _clean_text(raw)
        if not block:
            continue
        loc = full_text.find(raw, cursor)
        if loc < 0:
            loc = cursor
        blocks.append((loc, block))
        cursor = loc + len(raw)
    return blocks


def _page_from_offset(char_offset: int, page_offsets: list[tuple[int, int]]) -> int | None:
    if not page_offsets:
        return None
    page_num = page_offsets[0][1]
    for start, p in page_offsets:
        if start > char_offset:
            break
        page_num = p
    return page_num


def _classify_block(block: str) -> str:
    first_line = block.splitlines()[0].strip() if block.splitlines() else ""
    lowered = first_line.lower()
    if _is_references_header(lowered):
        return "references"
    if _TITLE_RE.match(first_line) and len(block) < 220:
        return "title"
    if _LIST_RE.match(block):
        return "list"
    if _FORMULA_RE.search(block):
        return "formula"
    return "regular"


def _split_by_type(text: str, block_type: str, base_size: int, base_overlap: int) -> list[str]:
    size = base_size
    overlap = base_overlap
    if block_type == "title":
        size = min(base_size, 520)
        overlap = min(base_overlap, 80)
    elif block_type == "list":
        size = min(base_size, 680)
        overlap = min(base_overlap, 90)
    elif block_type == "formula":
        size = min(base_size, 620)
        overlap = min(base_overlap, 100)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max(300, size),
        chunk_overlap=max(40, overlap),
        separators=_SENTENCE_SEPARATORS,
        length_function=len,
    )
    return splitter.split_text(text)


def _needs_bridge(prev_piece: str, next_block: str) -> bool:
    prev = prev_piece.rstrip()
    if not prev:
        return False
    if prev.endswith(("。", ".", "!", "?", "！", "？")):
        return False
    return len(next_block) > 40


def chunk_pages_v1(pages: list[tuple[int, str]]) -> list[PaperChunk]:
    """Legacy strategy: split each page independently."""
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=_SENTENCE_SEPARATORS,
        length_function=len,
    )
    results: list[PaperChunk] = []
    for page_num, text in pages:
        for piece in splitter.split_text(text):
            piece = _clean_text(piece)
            if _is_noisy(piece, settings.chunk_min_chars, settings.chunk_noise_symbol_ratio):
                continue
            results.append(PaperChunk(
                chunk_index=len(results),
                text=piece,
                page_num=page_num,
            ))
    return results


def chunk_pages_v3(pages: list[tuple[int, str]]) -> list[PaperChunk]:
    """
    V3 strategy:
    - document-level split with offset -> page mapping
    - structure-aware block classification (title/list/formula/regular/references)
    - dynamic chunk size by block type
    - lightweight bridge chunks across hard boundaries
    """
    settings = get_settings()
    full_text, page_offsets = _build_fulltext_with_offsets(pages)
    if not full_text:
        return []

    blocks = _split_blocks_with_offsets(full_text)
    results: list[PaperChunk] = []
    in_references = False
    prev_piece = ""

    for idx, (block_offset, block) in enumerate(blocks):
        block_type = _classify_block(block)
        if block_type == "references":
            in_references = True
        if settings.chunk_drop_references and in_references:
            continue

        # If we hit a short heading-like block, attach it to next block.
        if block_type == "title" and idx + 1 < len(blocks):
            _, nxt = blocks[idx + 1]
            block = f"{block}\n\n{nxt}"
            block_type = "regular"

        pieces = _split_by_type(
            text=block,
            block_type=block_type,
            base_size=settings.chunk_size,
            base_overlap=settings.chunk_overlap,
        )
        for piece in pieces:
            cleaned = _clean_text(piece)
            if not cleaned:
                continue
            if _is_noisy(cleaned, settings.chunk_min_chars, settings.chunk_noise_symbol_ratio):
                continue

            loc = full_text.find(cleaned, max(0, block_offset))
            if loc == -1:
                loc = block_offset
            page_num = _page_from_offset(loc, page_offsets)
            results.append(PaperChunk(
                chunk_index=len(results),
                text=cleaned,
                page_num=page_num,
            ))
            prev_piece = cleaned

        # Create a tiny bridge chunk for boundary continuity when needed.
        if idx + 1 < len(blocks):
            _, next_block = blocks[idx + 1]
            if _needs_bridge(prev_piece, next_block):
                tail = prev_piece[-180:].strip()
                head = _clean_text(next_block)[:180].strip()
                bridge = _clean_text(f"{tail}\n{head}")
                if (
                    len(bridge) >= settings.chunk_min_chars
                    and not _is_noisy(bridge, settings.chunk_min_chars, settings.chunk_noise_symbol_ratio)
                ):
                    bridge_loc = full_text.find(tail, max(0, block_offset))
                    if bridge_loc == -1:
                        bridge_loc = block_offset
                    results.append(PaperChunk(
                        chunk_index=len(results),
                        text=bridge,
                        page_num=_page_from_offset(bridge_loc, page_offsets),
                    ))

    return results


def chunk_pages_v2(pages: list[tuple[int, str]]) -> list[PaperChunk]:
    """
    Improved strategy:
    - clean and concatenate pages at document level (keeps cross-page continuity)
    - split with overlap
    - map chunk offsets back to page numbers
    - optionally drop references section noise
    """
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


def chunk_pages(pages: list[tuple[int, str]]) -> list[PaperChunk]:
    settings = get_settings()
    strategy = (settings.chunk_strategy or "v2").lower()
    if strategy == "v1":
        return chunk_pages_v1(pages)
    if strategy == "v3":
        return chunk_pages_v3(pages)
    return chunk_pages_v2(pages)
