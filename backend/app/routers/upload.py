"""Import papers into the local PaperRAG corpus."""
from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.mysql import SessionLocal, get_db
from app.models.paper import Paper
from app.models.upload_job import UploadJob
from app.schemas.chat import (
    ArxivImportBatchResponse,
    ArxivImportRequest,
    UploadJobListResponse,
    UploadJobResponse,
    UploadResponse,
)
from app.services.arxiv_import import (
    download_arxiv_pdf,
    fetch_arxiv_record,
    normalize_arxiv_id,
)
from app.services.ingest import _ingest_one

router = APIRouter(prefix="/upload", tags=["upload"])


def _upload_error(code: str, user_message: str, action_hint: str, retryable: bool) -> dict:
    return {
        "code": code,
        "user_message": user_message,
        "action_hint": action_hint,
        "retryable": retryable,
    }


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


def _dedupe_arxiv_ids(raw_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    arxiv_ids: list[str] = []
    for raw_id in raw_ids:
        arxiv_id = normalize_arxiv_id(raw_id)
        if arxiv_id not in seen:
            seen.add(arxiv_id)
            arxiv_ids.append(arxiv_id)
    return arxiv_ids


def _paper_is_ingested(paper: Paper | None) -> bool:
    return paper is not None and paper.ingest_status == "ok" and (paper.num_chunks or 0) > 0


def _new_upload_job(
    *,
    job_id: str,
    paper_id: str,
    filename: str,
    title: str,
    status: str,
    num_chunks: int,
    message: str,
) -> UploadJob:
    now = datetime.utcnow()
    return UploadJob(
        job_id=job_id,
        paper_id=paper_id,
        filename=filename,
        title=title,
        status=status,
        num_chunks=num_chunks,
        message=message,
        created_at=now,
        updated_at=now,
    )


def _run_arxiv_import_job(job_id: str, arxiv_id: str) -> None:
    db: Session = SessionLocal()
    try:
        job = db.query(UploadJob).filter(UploadJob.job_id == job_id).one_or_none()
        if job is None:
            return
        job.status = "running"
        job.message = "fetching_metadata"
        job.updated_at = datetime.utcnow()
        db.commit()

        record = fetch_arxiv_record(arxiv_id)
        job = db.query(UploadJob).filter(UploadJob.job_id == job_id).one()
        job.paper_id = record["paper_id"]
        job.filename = f"{record['paper_id']}.pdf"
        job.title = record.get("title") or record["paper_id"]
        job.message = "downloading_pdf"
        job.updated_at = datetime.utcnow()
        db.commit()

        record = download_arxiv_pdf(record)
        job = db.query(UploadJob).filter(UploadJob.job_id == job_id).one()
        job.message = "ingesting"
        job.updated_at = datetime.utcnow()
        db.commit()

        pid, ingest_status = _ingest_one(db, record, force=False)
        paper = db.query(Paper).filter_by(paper_id=pid).one_or_none()
        job = db.query(UploadJob).filter(UploadJob.job_id == job_id).one()
        if ingest_status == "ok":
            job.status = "succeeded"
            job.message = "ok"
        elif ingest_status == "skipped":
            job.status = "skipped"
            job.message = "already_exists"
        else:
            job.status = "failed"
            job.message = paper.ingest_error if paper is not None else "failed"
        job.num_chunks = paper.num_chunks if paper is not None else 0
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


@router.post("/arxiv", response_model=ArxivImportBatchResponse)
def import_arxiv_papers(
    request: ArxivImportRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> ArxivImportBatchResponse:
    try:
        arxiv_ids = _dedupe_arxiv_ids(request.arxiv_ids)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=_upload_error(
                "invalid_arxiv_id",
                "请输入有效的 arXiv ID 或 arXiv URL。",
                "例如 2511.16043、arXiv:2511.16043v2 或 https://arxiv.org/abs/2511.16043。",
                False,
            ),
        ) from exc

    items: list[UploadResponse] = []
    for arxiv_id in arxiv_ids:
        existing = db.query(Paper).filter(Paper.paper_id == arxiv_id).one_or_none()
        job_id = uuid.uuid4().hex
        if _paper_is_ingested(existing):
            db.add(_new_upload_job(
                job_id=job_id,
                paper_id=arxiv_id,
                filename=f"{arxiv_id}.pdf",
                title=existing.title or arxiv_id,
                status="skipped",
                num_chunks=existing.num_chunks or 0,
                message="already_exists",
            ))
            items.append(UploadResponse(
                job_id=job_id,
                paper_id=arxiv_id,
                status="skipped",
                num_chunks=existing.num_chunks or 0,
                message="already_exists",
            ))
            continue

        db.add(_new_upload_job(
            job_id=job_id,
            paper_id=arxiv_id,
            filename=f"{arxiv_id}.pdf",
            title=arxiv_id,
            status="queued",
            num_chunks=0,
            message="queued",
        ))
        items.append(UploadResponse(
            job_id=job_id,
            paper_id=arxiv_id,
            status="queued",
            num_chunks=0,
            message="queued",
        ))
        background_tasks.add_task(_run_arxiv_import_job, job_id, arxiv_id)

    db.commit()
    return ArxivImportBatchResponse(total=len(items), items=items)


@router.post("")
async def upload_pdf_disabled() -> JSONResponse:
    return JSONResponse(
        status_code=410,
        content=_upload_error(
            "pdf_upload_disabled",
            "本地 PDF 上传已关闭。",
            "请输入 arXiv ID 或 arXiv URL，系统会自动拉取官方 metadata 和 PDF。",
            False,
        ),
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
