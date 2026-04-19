from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks

from app.schemas.chat import IngestResponse
from app.services.ingest import run_ingest

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("", response_model=IngestResponse)
def trigger_ingest(background: BackgroundTasks, force: bool = False) -> IngestResponse:
    # Inline run — for 50-paper pilot ~ few minutes. Scale later with background.
    stats = run_ingest(force=force)
    return IngestResponse(stats=stats)
