"""Conversations CRUD: list/create/pin/delete + message history."""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.db.mysql import get_db
from app.models.chat_history import ChatHistory
from app.models.conversation import Conversation
from app.utils.content_safety import strip_hidden_reasoning

router = APIRouter(prefix="/conversations", tags=["conversations"])


class ConversationOut(BaseModel):
    id: str
    title: str
    pinned: bool
    created_at: str
    updated_at: str


class ConversationCreate(BaseModel):
    id: Optional[str] = None  # client may supply UUID
    title: Optional[str] = "新对话"


class ConversationUpdate(BaseModel):
    title: Optional[str] = None
    pinned: Optional[bool] = None


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    sources: list[dict] = []
    thinking: list[dict] = []
    presentation: Optional[dict] = None
    created_at: str


def _to_out(c: Conversation) -> ConversationOut:
    return ConversationOut(
        id=c.id,
        title=c.title,
        pinned=bool(c.pinned),
        created_at=c.created_at.isoformat(),
        updated_at=c.updated_at.isoformat(),
    )


@router.get("", response_model=list[ConversationOut])
def list_conversations(db: Session = Depends(get_db)) -> list[ConversationOut]:
    rows = (
        db.query(Conversation)
        .order_by(desc(Conversation.pinned), desc(Conversation.updated_at))
        .all()
    )
    return [_to_out(c) for c in rows]


@router.post("", response_model=ConversationOut)
def create_conversation(
    payload: ConversationCreate, db: Session = Depends(get_db)
) -> ConversationOut:
    cid = payload.id or str(uuid.uuid4())
    existing = db.query(Conversation).filter(Conversation.id == cid).one_or_none()
    if existing:
        return _to_out(existing)
    conv = Conversation(
        id=cid,
        title=(payload.title or "新对话")[:255],
        pinned=False,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return _to_out(conv)


@router.patch("/{cid}", response_model=ConversationOut)
def update_conversation(
    cid: str, payload: ConversationUpdate, db: Session = Depends(get_db)
) -> ConversationOut:
    conv = db.query(Conversation).filter(Conversation.id == cid).one_or_none()
    if not conv:
        raise HTTPException(404, "conversation not found")
    if payload.title is not None:
        conv.title = payload.title[:255]
    if payload.pinned is not None:
        conv.pinned = bool(payload.pinned)
    conv.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(conv)
    return _to_out(conv)


@router.delete("/{cid}")
def delete_conversation(cid: str, db: Session = Depends(get_db)) -> dict:
    conv = db.query(Conversation).filter(Conversation.id == cid).one_or_none()
    if not conv:
        return {"deleted": False}
    db.query(ChatHistory).filter(ChatHistory.conversation_id == cid).delete()
    db.delete(conv)
    db.commit()
    return {"deleted": True}


@router.get("/{cid}/messages", response_model=list[MessageOut])
def list_messages(cid: str, db: Session = Depends(get_db)) -> list[MessageOut]:
    rows = (
        db.query(ChatHistory)
        .filter(ChatHistory.conversation_id == cid)
        .order_by(ChatHistory.id.asc())
        .all()
    )
    out = []
    for r in rows:
        sources = []
        thinking: list[dict] = []
        presentation: Optional[dict] = None
        if r.sources_json:
            try:
                sources = json.loads(r.sources_json)
            except Exception:
                sources = []
        if r.thinking_json:
            try:
                payload = json.loads(r.thinking_json)
                # New format: {"traces": [...], "presentation": {...}}
                # Legacy format: a bare list of traces.
                if isinstance(payload, dict):
                    thinking = list(payload.get("traces") or [])
                    presentation = payload.get("presentation")
                elif isinstance(payload, list):
                    thinking = payload
            except Exception:
                thinking = []
        out.append(MessageOut(
            id=r.id,
            role=r.role,
            content=strip_hidden_reasoning(r.content) if r.role == "assistant" else r.content,
            sources=sources,
            thinking=thinking,
            presentation=presentation,
            created_at=r.created_at.isoformat() if r.created_at else "",
        ))
    return out
