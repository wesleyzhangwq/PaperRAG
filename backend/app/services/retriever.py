"""Hybrid retrieval: Chroma similarity search with metadata filtering."""
from __future__ import annotations

from typing import Optional

from langchain_core.documents import Document

from app.core.config import get_settings
from app.db.chroma import get_vector_store
from app.schemas.chat import ChatFilter


def _build_where(flt: Optional[ChatFilter]) -> Optional[dict]:
    if not flt:
        return None
    conds: list[dict] = []
    if flt.category:
        conds.append({"primary_category": {"$eq": flt.category}})
    if flt.year_min is not None:
        conds.append({"year": {"$gte": flt.year_min}})
    if flt.year_max is not None:
        conds.append({"year": {"$lte": flt.year_max}})
    if flt.paper_ids:
        conds.append({"paper_id": {"$in": flt.paper_ids}})

    if not conds:
        return None
    if len(conds) == 1:
        return conds[0]
    return {"$and": conds}


def retrieve(
    query: str,
    flt: Optional[ChatFilter] = None,
    top_k: Optional[int] = None,
) -> list[tuple[Document, float]]:
    settings = get_settings()
    vs = get_vector_store()
    k = top_k or settings.retrieval_k
    where = _build_where(flt)

    docs_scores = vs.similarity_search_with_score(query=query, k=k, filter=where)
    return docs_scores
