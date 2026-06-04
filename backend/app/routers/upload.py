"""Allow users to upload their own PDFs and auto-ingest."""
from __future__ import annotations

import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.mysql import SessionLocal, get_db
from app.models.paper import Paper
from app.models.upload_job import UploadJob
from app.schemas.chat import UploadJobListResponse, UploadJobResponse, UploadResponse
from app.services.ingest import _ingest_one

router = APIRouter(prefix="/upload", tags=["upload"])

settings = get_settings()

_SAFE_ID = re.compile(r"[^a-zA-Z0-9_\-\.]")
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB


def _upload_error(code: str, user_message: str, action_hint: str, retryable: bool) -> dict:
    return {
        "code": code,
        "user_message": user_message,
        "action_hint": action_hint,
        "retryable": retryable,
    }


def _safe_paper_id(name: str) -> str:
    base = Path(name).stem or "upload"
    cleaned = _SAFE_ID.sub("_", base)[:40] or "upload"
    return f"user_{cleaned}_{uuid.uuid4().hex[:8]}"


def _to_job_response(job: UploadJob) -> UploadJobResponse:
    return UploadJobResponse(
        job_id=job.job_id,
        paper_id=job.paper_id,
        filename=job.filename,
        title=job.title,
        status=job.status,
        num_chunks=job.num_chunks or 0,
        message=job.message,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


def _run_upload_ingest_job(job_id: str, record: dict) -> None:
    db: Session = SessionLocal()
    try:
        job = db.query(UploadJob).filter(UploadJob.job_id == job_id).one_or_none()
        if job is None:
            return
        job.status = "running"
        job.updated_at = datetime.utcnow()
        db.commit()

        pid, ingest_status = _ingest_one(db, record, force=True)
        paper = db.query(Paper).filter_by(paper_id=pid).one_or_none()
        job = db.query(UploadJob).filter(UploadJob.job_id == job_id).one()
        job.status = "succeeded" if ingest_status in {"ok", "skipped"} else "failed"
        job.num_chunks = paper.num_chunks if paper is not None else 0
        job.message = (
            paper.ingest_error
            if paper is not None and ingest_status == "failed"
            else ingest_status
        )
        job.updated_at = datetime.utcnow()
        db.commit()
    except Exception as exc:
        db.rollback()
        job = db.query(UploadJob).filter(UploadJob.job_id == job_id).one_or_none()
        if job is not None:
            job.status = "failed"
            job.message = f"{type(exc).__name__}: {exc}"
            job.updated_at = datetime.utcnow()
            db.commit()
    finally:
        db.close()


@router.post("", response_model=UploadResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str | None = Form(None),
    db: Session = Depends(get_db),
) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail=_upload_error(
                "invalid_file_type",
                "只支持上传 PDF 文件。",
                "请选择 .pdf 文件后重新上传。",
                False,
            ),
        )

    # Check file size (read size header or first-pass read)
    if file.size and file.size > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=_upload_error(
                "file_too_large",
                f"文件过大，当前最大支持 {_MAX_UPLOAD_BYTES // (1024*1024)} MB。",
                "请压缩 PDF 或拆分后重新上传。",
                False,
            ),
        )

    filename = file.filename
    paper_id = _safe_paper_id(filename)
    job_id = uuid.uuid4().hex

    pdf_dir = Path(settings.pdf_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    dest = pdf_dir / f"{paper_id}.pdf"

    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    record = {
        "paper_id": paper_id,
        "title": title or Path(filename).stem,
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

    db.add(UploadJob(
        job_id=job_id,
        paper_id=paper_id,
        filename=filename,
        title=record["title"],
        status="queued",
        num_chunks=0,
        message="queued",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    ))
    db.commit()
    background_tasks.add_task(_run_upload_ingest_job, job_id, record)

    return UploadResponse(
        job_id=job_id,
        paper_id=paper_id,
        status="queued",
        num_chunks=0,
        message="queued",
    )


@router.get("/jobs", response_model=UploadJobListResponse)
def list_upload_jobs(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> UploadJobListResponse:
    qs = db.query(UploadJob)
    total = qs.with_entities(func.count(UploadJob.id)).scalar() or 0
    items = (
        qs.order_by(UploadJob.updated_at.desc(), UploadJob.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return UploadJobListResponse(total=total, items=[_to_job_response(job) for job in items])


@router.get("/jobs/{job_id}", response_model=UploadJobResponse)
def get_upload_job(job_id: str, db: Session = Depends(get_db)) -> UploadJobResponse:
    job = db.query(UploadJob).filter(UploadJob.job_id == job_id).one_or_none()
    if job is None:
        raise HTTPException(404, "upload job not found")
    return _to_job_response(job)
