"""PDF text extraction.

Primary parser: pdfplumber (pure-python, robust).
Fallback:      PyMuPDF (fitz) — much faster, less precise on tables.

Returns a list of (page_num, page_text) tuples with blank pages filtered.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pdfplumber

try:
    import fitz  # type: ignore   # PyMuPDF
except Exception:
    fitz = None  # type: ignore


def _extract_with_pdfplumber(pdf_path: Path) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            text = text.strip()
            if text:
                out.append((i, text))
    return out


def _extract_with_fitz(pdf_path: Path) -> list[tuple[int, str]]:
    if fitz is None:
        raise RuntimeError("PyMuPDF not installed")
    out: list[tuple[int, str]] = []
    doc = fitz.open(pdf_path)
    try:
        for i, page in enumerate(doc, start=1):
            text = (page.get_text("text") or "").strip()
            if text:
                out.append((i, text))
    finally:
        doc.close()
    return out


def extract_pages(pdf_path: str | Path) -> list[tuple[int, str]]:
    """Try pdfplumber first, fall back to PyMuPDF."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    try:
        pages = _extract_with_pdfplumber(pdf_path)
        if pages:
            return pages
    except Exception:
        pass

    return _extract_with_fitz(pdf_path)


def join_pages(pages: Iterable[tuple[int, str]]) -> str:
    return "\n\n".join(text for _, text in pages)
