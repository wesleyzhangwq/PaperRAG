"""Chat router: placeholder until Task 12 rewrites with agent + SSE."""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/chat", tags=["chat"])


@router.get("/health")
def health():
    return {"status": "ok"}
