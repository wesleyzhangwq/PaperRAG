from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.mysql import get_db
from app.models.paper import Paper
from app.schemas.chat import PaperListResponse, PaperSummary

router = APIRouter(prefix="/papers", tags=["papers"])


def _to_summary(p: Paper) -> PaperSummary:
    return PaperSummary(
        paper_id=p.paper_id,
        title=p.title or "",
        authors=p.authors or [],
        year=p.year,
        primary_category=p.primary_category or "",
        categories=p.categories or [],
        doi=p.doi,
        abstract=p.abstract,
        arxiv_url=f"https://arxiv.org/abs/{p.paper_id}",
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


@router.get("/{paper_id}", response_model=PaperSummary)
def get_paper(paper_id: str, db: Session = Depends(get_db)) -> PaperSummary:
    p = db.query(Paper).filter(Paper.paper_id == paper_id).one_or_none()
    if not p:
        raise HTTPException(404, f"paper {paper_id} not found")
    return _to_summary(p)
