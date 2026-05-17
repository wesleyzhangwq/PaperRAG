from __future__ import annotations

from fastapi import APIRouter

from app.schemas.chat import IngestResponse
from app.services.ingest import run_ingest

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("", response_model=IngestResponse)
def trigger_ingest(force: bool = False) -> IngestResponse:
    stats = run_ingest(force=force)
    return IngestResponse(stats=stats)
