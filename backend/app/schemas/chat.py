"""API I/O schemas."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ChatFilter(BaseModel):
    category: Optional[str] = None          # exact match, e.g. "cs.CL"
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    paper_ids: Optional[list[str]] = None   # restrict to specific papers


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    session_id: str = "default"
    filter: Optional[ChatFilter] = None
    top_k: Optional[int] = None
    final_k: Optional[int] = None


class Source(BaseModel):
    paper_id: str
    title: str
    authors: list[str] = []
    year: Optional[int] = None
    primary_category: Optional[str] = None
    doi: Optional[str] = None
    arxiv_url: Optional[str] = None
    score: Optional[float] = None
    page_num: Optional[int] = None
    snippet: Optional[str] = None
    chunk_index: Optional[int] = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source] = []
    used_chunks: int = 0


class PaperSummary(BaseModel):
    paper_id: str
    title: str
    authors: list[str] = []
    year: int
    primary_category: str
    categories: list[str] = []
    doi: Optional[str] = None
    abstract: Optional[str] = None
    arxiv_url: Optional[str] = None
    ingest_status: str
    num_chunks: int


class PaperListResponse(BaseModel):
    total: int
    items: list[PaperSummary]


class UploadResponse(BaseModel):
    paper_id: str
    status: str
    num_chunks: int
    message: Optional[str] = None


class IngestResponse(BaseModel):
    stats: dict
