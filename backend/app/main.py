"""FastAPI app entrypoint."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.observability import configure_logging
from app.db.mysql import init_db
from app.middleware.request_context import RequestContextMiddleware
from app.routers import chat as chat_router
from app.routers import ingest as ingest_router
from app.routers import papers as papers_router
from app.routers import upload as upload_router

settings = get_settings()
configure_logging(json_logs=settings.observability_json_logs)

app = FastAPI(
    title="PaperRAG",
    version="0.1.0",
    description="arXiv RAG with MySQL + Qdrant + cloud LLM/Embedding APIs.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestContextMiddleware)

app.include_router(chat_router.router)
app.include_router(papers_router.router)
app.include_router(ingest_router.router)
app.include_router(upload_router.router)


@app.on_event("startup")
def _startup() -> None:
    init_db()


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {
        "status": "ok",
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "llm_api_base": settings.llm_api_base,
        "embedding_api_base": settings.embedding_api_base,
        "db": "mysql",
        "vector": "qdrant",
    }
