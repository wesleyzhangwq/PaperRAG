"""User feedback endpoints for answer quality signals."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.mysql import get_db
from app.models.feedback import AnswerFeedback
from app.schemas.chat import (
    AnswerFeedbackItem,
    AnswerFeedbackListResponse,
    AnswerFeedbackRequest,
    AnswerFeedbackResponse,
)

router = APIRouter(prefix="/feedback", tags=["feedback"])


def _to_feedback_item(row: AnswerFeedback) -> AnswerFeedbackItem:
    return AnswerFeedbackItem(
        id=row.id,
        conversation_id=row.conversation_id,
        message_id=row.message_id,
        vote=row.vote,
        reason=row.reason,
        comment=row.comment,
        created_at=row.created_at,
    )


@router.post("", response_model=AnswerFeedbackResponse)
def submit_answer_feedback(
    req: AnswerFeedbackRequest,
    db: Session = Depends(get_db),
) -> AnswerFeedbackResponse:
    db.add(AnswerFeedback(
        conversation_id=req.conversation_id,
        message_id=req.message_id,
        vote=req.vote,
        reason=req.reason,
        comment=req.comment,
    ))
    db.commit()
    return AnswerFeedbackResponse(status="recorded")


@router.get("", response_model=AnswerFeedbackListResponse)
def list_answer_feedback(
    vote: str | None = Query(None, pattern="^(up|down)$"),
    conversation_id: str | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> AnswerFeedbackListResponse:
    qs = db.query(AnswerFeedback)
    if vote:
        qs = qs.filter(AnswerFeedback.vote == vote)
    if conversation_id:
        qs = qs.filter(AnswerFeedback.conversation_id == conversation_id)
    total = qs.with_entities(func.count(AnswerFeedback.id)).scalar() or 0
    rows = qs.order_by(AnswerFeedback.id.desc()).offset(offset).limit(limit).all()
    return AnswerFeedbackListResponse(total=total, items=[_to_feedback_item(row) for row in rows])
