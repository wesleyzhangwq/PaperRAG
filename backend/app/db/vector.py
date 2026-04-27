"""Vector store entrypoint (Qdrant)."""
from __future__ import annotations

from app.db.qdrant import QdrantVectorStore, get_qdrant_vector_store


def get_vector_store() -> QdrantVectorStore:
    return get_qdrant_vector_store()
