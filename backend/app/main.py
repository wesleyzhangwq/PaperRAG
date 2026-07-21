"""FastAPI app entrypoint."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.observability import configure_logging
from app.db.mysql import init_db
from app.middleware.guardrails import APIKeyAuthMiddleware, FixedWindowRateLimitMiddleware
from app.middleware.request_context import RequestContextMiddleware
from app.routers import chat as chat_router
from app.routers import conversations as conversations_router
from app.routers import feedback as feedback_router
from app.routers import ingest as ingest_router
from app.routers import papers as papers_router
from app.routers import upload as upload_router

settings = get_settings()
configure_logging(json_logs=settings.observability_json_logs)

app = FastAPI(
    title="Cite Scope",
    version="0.1.0",
    description="Agentic research assistant for cited arXiv paper Q&A with MySQL, Qdrant, and cloud LLM/Embedding APIs.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    FixedWindowRateLimitMiddleware,
    enabled=settings.rate_limit_enabled,
    requests=settings.rate_limit_requests,
    window_seconds=settings.rate_limit_window_seconds,
    protected_path_prefixes=settings.rate_limit_path_list,
)
app.add_middleware(
    APIKeyAuthMiddleware,
    enabled=settings.api_auth_enabled,
    api_keys=settings.api_key_list,
    exempt_path_prefixes=settings.api_auth_exempt_path_list,
)
app.add_middleware(RequestContextMiddleware)

app.include_router(chat_router.router)
app.include_router(conversations_router.router)
app.include_router(feedback_router.router)
app.include_router(papers_router.router)
app.include_router(ingest_router.router)
app.include_router(upload_router.router)


@app.on_event("startup")
def _startup() -> None:
    init_db()


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok"}
