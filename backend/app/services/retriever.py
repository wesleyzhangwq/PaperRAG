"""Vector retrieval with optional hybrid BM25 rerank and TTL cache."""
from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from typing import Optional

import orjson
from cachetools import TTLCache
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.core.config import get_settings
from app.core.context import request_id_ctx
from app.db.vector import get_vector_store
from app.schemas.chat import ChatFilter

log = logging.getLogger("app.services.retriever")

_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]", re.IGNORECASE)

_retrieve_cache: Optional[TTLCache[str, list[tuple[Document, float]]]] = None
_cache_lock = threading.Lock()


def _cache_instance() -> Optional[TTLCache[str, list[tuple[Document, float]]]]:
    global _retrieve_cache
    s = get_settings()
    if not s.cache_retrieval_enabled:
        _retrieve_cache = None
        return None
    if _retrieve_cache is None:
        _retrieve_cache = TTLCache(
            maxsize=max(1, s.cache_retrieval_max_entries),
            ttl=max(1, s.cache_retrieval_ttl_sec),
        )
    elif (
        _retrieve_cache.maxsize != s.cache_retrieval_max_entries
        or _retrieve_cache.ttl != s.cache_retrieval_ttl_sec
    ):
        _retrieve_cache = TTLCache(
            maxsize=max(1, s.cache_retrieval_max_entries),
            ttl=max(1, s.cache_retrieval_ttl_sec),
        )
    return _retrieve_cache


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


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _min_max_norm(values: list[float]) -> list[float]:
    if not values:
        return []
    mn, mx = min(values), max(values)
    if mx - mn < 1e-12:
        return [0.5] * len(values)
    return [(v - mn) / (mx - mn) for v in values]


def _hybrid_fuse(
    query: str,
    docs_scores: list[tuple[Document, float]],
    k: int,
    alpha: float,
) -> list[tuple[Document, float]]:
    if len(docs_scores) <= 1:
        return docs_scores[:k]
    docs = [d for d, _ in docs_scores]
    corpus = [_tokenize(d.page_content or "") for d in docs]
    if all(len(toks) == 0 for toks in corpus):
        return docs_scores[:k]
    bm25 = BM25Okapi(corpus)
    q_toks = _tokenize(query)
    bm_scores = list(bm25.get_scores(q_toks))
    vec_scores = [s for _, s in docs_scores]
    v_norm = _min_max_norm(vec_scores)
    b_norm = _min_max_norm(bm_scores)
    fused: list[tuple[Document, float]] = []
    for i, d in enumerate(docs):
        score = alpha * v_norm[i] + (1.0 - alpha) * b_norm[i]
        fused.append((d, float(score)))
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused[:k]


def _cache_key(
    query: str,
    where: Optional[dict],
    k: int,
    hybrid: bool,
    oversample: float,
    alpha: float,
    max_fetch: int,
) -> str:
    payload = {
        "q": query.strip(),
        "w": where,
        "k": k,
        "hybrid": hybrid,
        "oversample": oversample,
        "alpha": alpha,
        "max_fetch": max_fetch,
    }
    raw = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(raw).hexdigest()


def retrieve(
    query: str,
    flt: Optional[ChatFilter] = None,
    top_k: Optional[int] = None,
) -> list[tuple[Document, float]]:
    settings = get_settings()
    vs = get_vector_store()
    k = top_k or settings.retrieval_k
    where = _build_where(flt)
    hybrid = settings.hybrid_retrieval_enabled
    oversample = max(1.0, float(settings.hybrid_oversample))
    alpha = min(1.0, max(0.0, float(settings.hybrid_alpha)))
    max_fetch = max(k, min(int(settings.hybrid_max_fetch), int(k * oversample)))

    cache = _cache_instance()
    ck: Optional[str] = None
    if cache is not None:
        ck = _cache_key(
            query, where, k, hybrid, oversample, alpha, max_fetch
        )
        with _cache_lock:
            hit = cache.get(ck)
        if hit is not None:
            log.info(
                "retrieve_cache_hit",
                extra={
                    "event": "rag.retrieve",
                    "request_id": request_id_ctx.get(),
                    "phase": "cache_hit",
                    "chunks": len(hit),
                    "top_k": k,
                },
            )
            return list(hit)

    t0 = time.perf_counter()
    try:
        fetch_limit = max_fetch if hybrid else k
        raw = vs.similarity_search_with_score(
            query=query, k=k, filter=where, fetch_limit=fetch_limit
        )
        if hybrid and raw:
            out = _hybrid_fuse(query, raw, k, alpha)
        else:
            out = raw[:k]
    except Exception:
        log.exception(
            "retrieve_failed",
            extra={
                "event": "rag.retrieve",
                "request_id": request_id_ctx.get(),
                "phase": "error",
                "error_kind": "retrieve",
                "top_k": k,
            },
        )
        return []

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    log.info(
        "retrieve_done",
        extra={
            "event": "rag.retrieve",
            "request_id": request_id_ctx.get(),
            "phase": "done",
            "ms": elapsed_ms,
            "chunks": len(out),
            "top_k": k,
        },
    )
    if cache is not None and ck is not None:
        with _cache_lock:
            cache[ck] = list(out)
    return out
