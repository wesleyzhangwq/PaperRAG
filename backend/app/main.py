"""FastAPI app entrypoint."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.db.mysql import init_db
from app.routers import chat as chat_router
from app.routers import ingest as ingest_router
from app.routers import papers as papers_router
from app.routers import upload as upload_router

settings = get_settings()

app = FastAPI(
    title="PaperRAG",
    version="0.1.0",
    description="arXiv RAG with MySQL + Qdrant + Ollama (gemma4:e4b + bge-m3).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        "ollama": settings.ollama_base_url,
        "db": "mysql",
        "vector": "qdrant",
    }
