"""Ingest trigger: re-process PDFs into chunks and vectors.

This is a resource-intensive admin operation, so it requires the
ADMIN_API_KEY header (if configured). If ADMIN_API_KEY is empty/unset,
the endpoint is open (dev mode).
"""
from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException
from typing import Optional

from app.core.config import get_settings
from app.schemas.chat import IngestResponse
from app.services.ingest import run_ingest

router = APIRouter(prefix="/ingest", tags=["ingest"])


def _check_admin_key(x_admin_key: Optional[str] = Header(None)) -> None:
    expected = get_settings().admin_api_key
    if expected and x_admin_key != expected:
        raise HTTPException(403, "Invalid or missing X-Admin-Key header")


@router.post("", response_model=IngestResponse, dependencies=[])
def trigger_ingest(
    force: bool = False,
    x_admin_key: Optional[str] = Header(None),
) -> IngestResponse:
    _check_admin_key(x_admin_key)
    stats = run_ingest(force=force)
    return IngestResponse(stats=stats)
