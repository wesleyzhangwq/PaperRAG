"""Import papers into the local Cite Scope corpus."""
from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.mysql import SessionLocal, get_db
from app.models.paper import Paper
from app.models.upload_job import UploadJob
from app.schemas.chat import (
    ArxivImportBatchResponse,
    ArxivImportRequest,
    FileUploadBatchResponse,
    UploadJobListResponse,
    UploadJobResponse,
    UploadResponse,
)
from app.services.arxiv_import import (
    download_arxiv_pdf,
    fetch_arxiv_record,
    normalize_arxiv_id,
)
from app.services.document_parser import (
    UnsupportedDocumentError,
    media_type_for_filename,
    safe_filename,
)
from app.services.ingest import _ingest_one
from app.utils.time import utc_now

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
        stage=job.stage or job.status,
        progress=job.progress or 0,
        num_chunks=job.num_chunks or 0,
        message=job.message,
        source_kind=job.source_kind or "arxiv",
        media_type=job.media_type,
        content_hash=job.content_hash,
        error_code=job.error_code,
        warnings=list(job.warnings or []),
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
    source_kind: str = "arxiv",
    media_type: str | None = None,
    content_hash: str | None = None,
    stage: str | None = None,
    progress: int = 0,
    error_code: str | None = None,
    warnings: list[str] | None = None,
) -> UploadJob:
    now = utc_now()
    return UploadJob(
        job_id=job_id,
        paper_id=paper_id,
        filename=filename,
        title=title,
        status=status,
        stage=stage or status,
        progress=progress,
        num_chunks=num_chunks,
        message=message,
        source_kind=source_kind,
        media_type=media_type,
        content_hash=content_hash,
        error_code=error_code,
        warnings=warnings or [],
        created_at=now,
        updated_at=now,
    )


def _update_job(
    job_id: str,
    *,
    status: str | None = None,
    stage: str | None = None,
    progress: int | None = None,
    message: str | None = None,
    error_code: str | None = None,
) -> None:
    """Persist progress through a short independent transaction."""
    progress_db: Session = SessionLocal()
    try:
        job = (
            progress_db.query(UploadJob)
            .filter(UploadJob.job_id == job_id)
            .one_or_none()
        )
        if job is None:
            return
        if status is not None:
            job.status = status
        if stage is not None:
            job.stage = stage
        if progress is not None:
            job.progress = max(0, min(100, progress))
        if message is not None:
            job.message = message
        if error_code is not None:
            job.error_code = error_code
        job.updated_at = utc_now()
        progress_db.commit()
    finally:
        progress_db.close()


def _error_code(exc: Exception) -> str:
    if isinstance(exc, UnsupportedDocumentError):
        return "unsupported_file_type"
    if isinstance(exc, FileNotFoundError):
        return "source_file_missing"
    if "no usable document blocks" in str(exc):
        return "no_extractable_content"
    if "Embedding" in str(exc) or "embedding" in str(exc):
        return "embedding_failed"
    return "ingest_failed"


def _run_arxiv_import_job(job_id: str, arxiv_id: str) -> None:
    db: Session = SessionLocal()
    try:
        job = db.query(UploadJob).filter(UploadJob.job_id == job_id).one_or_none()
        if job is None:
            return
        job.status = "running"
        job.stage = "fetching_metadata"
        job.progress = 5
        job.message = "fetching_metadata"
        job.updated_at = utc_now()
        db.commit()

        record = fetch_arxiv_record(arxiv_id)
        job = db.query(UploadJob).filter(UploadJob.job_id == job_id).one()
        job.paper_id = record["paper_id"]
        job.filename = f"{record['paper_id']}.pdf"
        job.title = record.get("title") or record["paper_id"]
        job.message = "downloading_pdf"
        job.stage = "downloading"
        job.progress = 15
        job.updated_at = utc_now()
        db.commit()

        record = download_arxiv_pdf(record)
        job = db.query(UploadJob).filter(UploadJob.job_id == job_id).one()
        job.message = "parsing_document"
        job.stage = "parsing"
        job.progress = 25
        job.updated_at = utc_now()
        db.commit()

        pid, ingest_status = _ingest_one(
            db,
            {
                **record,
                "source_kind": "arxiv",
                "media_type": "application/pdf",
                "original_filename": f"{record['paper_id']}.pdf",
            },
            force=False,
            progress=lambda stage, percent, message: _update_job(
                job_id,
                status="running",
                stage=stage,
                progress=percent,
                message=message,
            ),
        )
        paper = db.query(Paper).filter_by(paper_id=pid).one_or_none()
        job = db.query(UploadJob).filter(UploadJob.job_id == job_id).one()
        if ingest_status == "ok":
            job.status = "succeeded"
            job.stage = "completed"
            job.progress = 100
            job.message = "ok"
        elif ingest_status == "skipped":
            job.status = "skipped"
            job.stage = "completed"
            job.progress = 100
            job.message = "already_exists"
        job.num_chunks = paper.num_chunks if paper is not None else 0
        if paper is not None:
            metadata = paper.ingest_metadata or {}
            job.warnings = list(metadata.get("warnings") or [])
            job.content_hash = paper.content_hash
        job.updated_at = utc_now()
        db.commit()
    except Exception as exc:
        db.rollback()
        job = db.query(UploadJob).filter(UploadJob.job_id == job_id).one_or_none()
        if job is not None:
            job.status = "failed"
            job.stage = "failed"
            job.message = f"{type(exc).__name__}: {exc}"
            job.error_code = _error_code(exc)
            job.updated_at = utc_now()
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


async def _save_upload(upload: UploadFile, *, job_id: str) -> tuple[Path, str, int, str]:
    filename = safe_filename(upload.filename)
    media_type = media_type_for_filename(filename)
    settings = get_settings()
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / f"{job_id}__{filename}"
    max_bytes = settings.ingest_max_file_mb * 1024 * 1024
    digest = hashlib.sha256()
    size = 0

    try:
        with destination.open("wb") as handle:
            while True:
                block = await upload.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                if size > max_bytes:
                    raise ValueError(
                        f"file exceeds {settings.ingest_max_file_mb} MB limit"
                    )
                digest.update(block)
                handle.write(block)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()

    if size == 0:
        destination.unlink(missing_ok=True)
        raise ValueError("empty upload")
    return destination, digest.hexdigest(), size, media_type


def _run_file_ingest_job(job_id: str, record: dict) -> None:
    db: Session = SessionLocal()
    try:
        _update_job(
            job_id,
            status="running",
            stage="parsing",
            progress=20,
            message="parsing_document",
        )
        paper_id, status = _ingest_one(
            db,
            record,
            force=False,
            progress=lambda stage, percent, message: _update_job(
                job_id,
                status="running",
                stage=stage,
                progress=percent,
                message=message,
            ),
        )
        db.commit()
        paper = db.query(Paper).filter(Paper.paper_id == paper_id).one()
        job = db.query(UploadJob).filter(UploadJob.job_id == job_id).one()
        job.status = "succeeded" if status == "ok" else "skipped"
        job.stage = "completed"
        job.progress = 100
        job.message = "ok" if status == "ok" else "already_exists"
        job.num_chunks = paper.num_chunks or 0
        job.warnings = list((paper.ingest_metadata or {}).get("warnings") or [])
        job.updated_at = utc_now()
        db.commit()
    except Exception as exc:
        db.rollback()
        job = db.query(UploadJob).filter(UploadJob.job_id == job_id).one_or_none()
        if job is not None:
            job.status = "failed"
            job.stage = "failed"
            job.message = f"{type(exc).__name__}: {exc}"
            job.error_code = _error_code(exc)
            job.updated_at = utc_now()
            db.commit()
    finally:
        db.close()


@router.post("/files", response_model=FileUploadBatchResponse)
async def upload_files(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
) -> FileUploadBatchResponse:
    """Queue up to 20 heterogeneous files for the unified ingest pipeline."""
    if not files or len(files) > 20:
        raise HTTPException(
            status_code=400,
            detail=_upload_error(
                "invalid_file_count",
                "请选择 1–20 个文件。",
                "拆分批次后重试。",
                False,
            ),
        )

    # Validate extensions before writing any bytes.
    try:
        for upload in files:
            media_type_for_filename(safe_filename(upload.filename))
    except UnsupportedDocumentError as exc:
        raise HTTPException(
            status_code=415,
            detail=_upload_error(
                "unsupported_file_type",
                "文件类型暂不支持。",
                str(exc),
                False,
            ),
        ) from exc

    items: list[UploadResponse] = []
    saved_paths: list[Path] = []
    for upload in files:
        job_id = uuid.uuid4().hex
        try:
            path, content_hash, file_size, media_type = await _save_upload(
                upload,
                job_id=job_id,
            )
            saved_paths.append(path)
        except ValueError as exc:
            for saved_path in saved_paths:
                saved_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=413 if "exceeds" in str(exc) else 400,
                detail=_upload_error(
                    "file_too_large" if "exceeds" in str(exc) else "empty_file",
                    "文件超过大小限制。" if "exceeds" in str(exc) else "文件内容为空。",
                    str(exc),
                    False,
                ),
            ) from exc

        filename = safe_filename(upload.filename)
        existing = (
            db.query(Paper)
            .filter(Paper.content_hash == content_hash)
            .one_or_none()
        )
        if _paper_is_ingested(existing):
            path.unlink(missing_ok=True)
            db.add(
                _new_upload_job(
                    job_id=job_id,
                    paper_id=existing.paper_id,
                    filename=filename,
                    title=existing.title,
                    status="skipped",
                    stage="completed",
                    progress=100,
                    num_chunks=existing.num_chunks or 0,
                    message="duplicate_content",
                    source_kind="upload",
                    media_type=media_type,
                    content_hash=content_hash,
                )
            )
            items.append(
                UploadResponse(
                    job_id=job_id,
                    paper_id=existing.paper_id,
                    status="skipped",
                    num_chunks=existing.num_chunks or 0,
                    message="duplicate_content",
                )
            )
            continue

        paper_id = f"local-{content_hash[:24]}"
        title = Path(filename).stem
        record = {
            "paper_id": paper_id,
            "title": title,
            "authors": [],
            "year": utc_now().year,
            "primary_category": "local",
            "categories": ["local_upload"],
            "source_path": str(path),
            "source_kind": "upload",
            "media_type": media_type,
            "content_hash": content_hash,
            "original_filename": filename,
            "ingest_metadata": {"file_size": file_size},
        }
        db.add(
            _new_upload_job(
                job_id=job_id,
                paper_id=paper_id,
                filename=filename,
                title=title,
                status="queued",
                stage="saved",
                progress=10,
                num_chunks=0,
                message="saved",
                source_kind="upload",
                media_type=media_type,
                content_hash=content_hash,
            )
        )
        items.append(
            UploadResponse(
                job_id=job_id,
                paper_id=paper_id,
                status="queued",
                num_chunks=0,
                message="saved",
            )
        )
        background_tasks.add_task(_run_file_ingest_job, job_id, record)

    db.commit()
    return FileUploadBatchResponse(total=len(items), items=items)


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
