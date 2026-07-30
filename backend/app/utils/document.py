"""Shared document-ingestion value objects."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DocumentBlock:
    """A normalized, provenance-preserving unit emitted by a file parser."""

    text: str
    modality: str = "text"
    source_locator: dict = field(default_factory=dict)
    section: str | None = None


@dataclass(frozen=True)
class ParsedDocument:
    blocks: list[DocumentBlock]
    title: str | None = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

