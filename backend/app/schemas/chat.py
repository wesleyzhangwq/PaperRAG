"""API I/O schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ChatFilter(BaseModel):
    category: Optional[str] = None          # exact match, e.g. "cs.CL"
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    paper_ids: Optional[list[str]] = None   # restrict to specific papers


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    session_id: str = "default"
    conversation_id: Optional[str] = None  # preferred; falls back to session_id
    filter: Optional[ChatFilter] = None
    top_k: Optional[int] = Field(None, ge=1, le=50)
    final_k: Optional[int] = Field(None, ge=1, le=20)


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
    step_traces: Optional[list[dict]] = None
    reflection_result: Optional[dict] = None


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


class CorpusRepresentativePaper(BaseModel):
    paper_id: str
    title: str
    year: Optional[int] = None
    primary_category: str
    arxiv_url: str


class CorpusTopicBucket(BaseModel):
    key: str
    label: str
    description: str
    paper_count: int
    chunk_count: int
    representative_papers: list[CorpusRepresentativePaper] = []


class CorpusOverviewResponse(BaseModel):
    total_papers: int
    total_chunks: int
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    topic_buckets: list[CorpusTopicBucket] = []
    suggested_questions: list[str] = []
    generated_at: datetime


class UploadResponse(BaseModel):
    job_id: Optional[str] = None
    paper_id: str
    status: str
    num_chunks: int
    message: Optional[str] = None


class UploadJobResponse(BaseModel):
    job_id: str
    paper_id: str
    filename: str
    title: str
    status: str
    num_chunks: int
    message: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class UploadJobListResponse(BaseModel):
    total: int
    items: list[UploadJobResponse]


class IngestResponse(BaseModel):
    stats: dict


class AnswerFeedbackRequest(BaseModel):
    conversation_id: str = Field(..., min_length=1, max_length=64)
    message_id: Optional[int] = Field(None, ge=1)
    vote: str = Field(..., pattern="^(up|down)$")
    reason: Optional[str] = Field(None, max_length=128)
    comment: Optional[str] = Field(None, max_length=2000)


class AnswerFeedbackResponse(BaseModel):
    status: str


class AnswerFeedbackItem(BaseModel):
    id: int
    conversation_id: str
    message_id: Optional[int] = None
    vote: str
    reason: Optional[str] = None
    comment: Optional[str] = None
    created_at: datetime


class AnswerFeedbackListResponse(BaseModel):
    total: int
    items: list[AnswerFeedbackItem]
