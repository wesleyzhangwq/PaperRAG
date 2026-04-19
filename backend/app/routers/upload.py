"""Allow users to upload their own PDFs and auto-ingest."""
from __future__ import annotations

import re
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.mysql import get_db
from app.schemas.chat import UploadResponse
from app.services.ingest import _ingest_one

router = APIRouter(prefix="/upload", tags=["upload"])

settings = get_settings()

_SAFE_ID = re.compile(r"[^a-zA-Z0-9_\-\.]")


def _safe_paper_id(name: str) -> str:
    base = Path(name).stem or "upload"
    cleaned = _SAFE_ID.sub("_", base)[:40] or "upload"
    return f"user_{cleaned}_{uuid.uuid4().hex[:8]}"


@router.post("", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    title: str | None = Form(None),
    db: Session = Depends(get_db),
) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only .pdf files are accepted")

    paper_id = _safe_paper_id(file.filename)

    pdf_dir = Path(settings.pdf_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    dest = pdf_dir / f"{paper_id}.pdf"

    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    record = {
        "paper_id": paper_id,
        "title": title or Path(file.filename).stem,
        "authors": [],
        "year": 0,
        "primary_category": "user.upload",
        "categories": ["user.upload"],
        "doi": None,
        "abstract": None,
        "pdf_url": None,
        "pdf_path": str(dest.relative_to(Path(settings.data_dir).parent)),
        "entry_id": None,
        "published": None,
        "updated": None,
    }

    pid, status = _ingest_one(db, record, force=True)
    db.commit()

    paper = db.query(__import__("app.models.paper", fromlist=["Paper"]).Paper).filter_by(paper_id=pid).one()
    return UploadResponse(
        paper_id=pid,
        status=status,
        num_chunks=paper.num_chunks,
        message=paper.ingest_error if status == "failed" else "ingested",
    )
