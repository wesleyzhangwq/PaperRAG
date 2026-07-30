from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.mysql import get_db
from app.models.paper import Paper
from app.schemas.chat import (
    CorpusOverviewResponse,
    CorpusRepresentativePaper,
    CorpusTopicBucket,
    PaperListResponse,
    PaperSummary,
)

router = APIRouter(prefix="/papers", tags=["papers"])


@dataclass(frozen=True)
class TopicDefinition:
    key: str
    label: str
    description: str
    question: str


TOPIC_DEFINITIONS: tuple[TopicDefinition, ...] = (
    TopicDefinition(
        key="rag_ir_memory",
        label="RAG / Retrieval",
        description="Retrieval-augmented generation, dense retrieval, reranking, memory, and evidence search.",
        question="RAG / Retrieval 的技术路线是如何演进的？",
    ),
    TopicDefinition(
        key="agents_reasoning",
        label="Agents / Reasoning",
        description="Tool use, planning, self-reflection, multi-step reasoning, and agent workflows.",
        question="Agents / Reasoning 相关论文主要解决了哪些问题？",
    ),
    TopicDefinition(
        key="llm_transformer",
        label="LLM / Transformer",
        description="Transformer models, pretraining, scaling, instruction tuning, and model families.",
        question="Transformer 到现代 LLM 的关键转折点是什么？",
    ),
    TopicDefinition(
        key="evaluation_factuality",
        label="Evaluation / Factuality",
        description="Benchmarks, hallucination, truthfulness, attribution, and answer faithfulness.",
        question="这批论文里关于 hallucination 和 factuality 的主要评估方法有哪些？",
    ),
    TopicDefinition(
        key="alignment_safety_eval",
        label="Alignment / Safety",
        description="RLHF, preference learning, safety evaluation, harmlessness, and red teaming.",
        question="Alignment / Safety 论文里有哪些代表性训练和评估方法？",
    ),
    TopicDefinition(
        key="multimodal_generative",
        label="Multimodal / Generative",
        description="Vision-language models, diffusion, image generation, and multimodal instruction tuning.",
        question="Multimodal / Generative 方向的关键模型路线是什么？",
    ),
    TopicDefinition(
        key="deep_learning",
        label="Deep Learning Foundations",
        description="Core optimization, representation learning, sequence models, and neural architecture foundations.",
        question="Deep Learning Foundations 里哪些论文奠定了现代大模型基础？",
    ),
)

TOPIC_BY_KEY = {topic.key: topic for topic in TOPIC_DEFINITIONS}
TOPIC_PRIORITY = {topic.key: index for index, topic in enumerate(TOPIC_DEFINITIONS)}
OTHER_TOPIC = TopicDefinition(
    key="other",
    label="Other AI Papers",
    description="Additional AI papers grouped by arXiv metadata when no curated corpus bucket is available.",
    question="这批其他 AI 论文主要覆盖哪些方向？",
)


def _paper_source_kind(paper: Paper) -> str:
    value = getattr(paper, "source_kind", None)
    return value if isinstance(value, str) and value else "arxiv"


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _to_summary(p: Paper) -> PaperSummary:
    topic_key = _bucket_key_for_paper(p)
    topic = TOPIC_BY_KEY.get(topic_key, OTHER_TOPIC)
    source_kind = _paper_source_kind(p)
    return PaperSummary(
        paper_id=p.paper_id,
        title=p.title or "",
        authors=p.authors or [],
        year=p.year,
        primary_category=p.primary_category or "",
        categories=p.categories or [],
        topic_bucket_key=topic.key,
        topic_bucket_label=topic.label,
        doi=p.doi,
        abstract=p.abstract,
        arxiv_url=(
            f"https://arxiv.org/abs/{p.paper_id}"
            if source_kind == "arxiv"
            else None
        ),
        source_kind=source_kind,
        media_type=_optional_text(getattr(p, "media_type", None)),
        original_filename=_optional_text(getattr(p, "original_filename", None)),
        ingest_status=p.ingest_status or "pending",
        num_chunks=p.num_chunks or 0,
    )


@router.get("", response_model=PaperListResponse)
def list_papers(
    category: Optional[str] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    q: Optional[str] = Query(None, description="search title/abstract"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> PaperListResponse:
    qs = db.query(Paper).filter(Paper.ingest_status == "ok")
    if category:
        qs = qs.filter(Paper.primary_category == category)
    if year_min is not None:
        qs = qs.filter(Paper.year >= year_min)
    if year_max is not None:
        qs = qs.filter(Paper.year <= year_max)
    if q:
        like = f"%{q}%"
        qs = qs.filter((Paper.title.ilike(like)) | (Paper.abstract.ilike(like)))

    total = qs.with_entities(func.count(Paper.id)).scalar() or 0
    items = qs.order_by(Paper.year.desc(), Paper.id.desc()).offset(offset).limit(limit).all()
    return PaperListResponse(total=total, items=[_to_summary(p) for p in items])


@router.get("/overview", response_model=CorpusOverviewResponse)
def papers_overview(db: Session = Depends(get_db)) -> CorpusOverviewResponse:
    papers = db.query(Paper).filter(Paper.ingest_status == "ok").all()
    total_papers = len(papers)
    total_chunks = sum(max(int(getattr(p, "num_chunks", 0) or 0), 0) for p in papers)
    years = [int(p.year) for p in papers if getattr(p, "year", None)]

    buckets: dict[str, list[Paper]] = {}
    for paper in papers:
        bucket_key = _bucket_key_for_paper(paper)
        buckets.setdefault(bucket_key, []).append(paper)

    topic_buckets = [
        _build_topic_bucket(bucket_key, bucket_papers)
        for bucket_key, bucket_papers in buckets.items()
    ]
    topic_buckets.sort(
        key=lambda bucket: (
            -bucket.paper_count,
            TOPIC_PRIORITY.get(bucket.key, len(TOPIC_PRIORITY)),
            bucket.label,
        )
    )
    visible_buckets = topic_buckets[:7]

    return CorpusOverviewResponse(
        total_papers=total_papers,
        total_chunks=total_chunks,
        year_min=min(years) if years else None,
        year_max=max(years) if years else None,
        topic_buckets=visible_buckets,
        suggested_questions=_suggested_questions(visible_buckets),
        generated_at=datetime.now(timezone.utc),
    )


def _bucket_key_for_paper(paper: Paper) -> str:
    categories = getattr(paper, "categories", None) or []
    for topic in TOPIC_DEFINITIONS:
        if topic.key in categories:
            return topic.key
    return OTHER_TOPIC.key


def _build_topic_bucket(bucket_key: str, papers: list[Paper]) -> CorpusTopicBucket:
    topic = TOPIC_BY_KEY.get(bucket_key, OTHER_TOPIC)
    representatives = [
        _to_representative(paper)
        for paper in sorted(papers, key=_representative_sort_key)[:3]
    ]
    return CorpusTopicBucket(
        key=topic.key,
        label=topic.label,
        description=topic.description,
        paper_count=len(papers),
        chunk_count=sum(max(int(getattr(p, "num_chunks", 0) or 0), 0) for p in papers),
        representative_papers=representatives,
    )


def _representative_sort_key(paper: Paper) -> tuple[int, int, str]:
    year = int(getattr(paper, "year", 9999) or 9999)
    num_chunks = int(getattr(paper, "num_chunks", 0) or 0)
    return (year, -num_chunks, (paper.title or "").lower())


def _to_representative(paper: Paper) -> CorpusRepresentativePaper:
    source_kind = _paper_source_kind(paper)
    return CorpusRepresentativePaper(
        paper_id=paper.paper_id,
        title=paper.title or "",
        year=paper.year,
        primary_category=paper.primary_category or "",
        arxiv_url=(
            f"https://arxiv.org/abs/{paper.paper_id}"
            if source_kind == "arxiv"
            else None
        ),
    )


def _suggested_questions(topic_buckets: list[CorpusTopicBucket]) -> list[str]:
    if not topic_buckets:
        return ["如何上传第一篇论文？"]
    questions = []
    for bucket in topic_buckets:
        topic = TOPIC_BY_KEY.get(bucket.key, OTHER_TOPIC)
        questions.append(topic.question)
        if len(questions) >= 6:
            break
    if any(bucket.key == "rag_ir_memory" for bucket in topic_buckets) and len(questions) < 6:
        questions.append("对比这批论文里的 dense retrieval、rerank 和 RAG 方法。")
    return questions[:6]


@router.get("/{paper_id}", response_model=PaperSummary)
def get_paper(paper_id: str, db: Session = Depends(get_db)) -> PaperSummary:
    p = db.query(Paper).filter(Paper.paper_id == paper_id).one_or_none()
    if not p:
        raise HTTPException(404, f"paper {paper_id} not found")
    return _to_summary(p)
