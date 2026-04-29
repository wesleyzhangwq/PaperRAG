"""Special acceptance checks for 4 optimization areas.

Run:
  cd backend
  .venv/bin/python ../eval/special_acceptance.py
"""
from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from app.core.observability import JsonLogFormatter
from app.db.qdrant import AlibabaEmbeddingClient
from app.services import retriever as retriever_mod


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "eval" / "results" / "special_acceptance.json"


def _ndcg_at_k(ranked: list[str], relevant: set[str], k: int = 3) -> float:
    dcg = 0.0
    for i, pid in enumerate(ranked[:k], start=1):
        rel = 1.0 if pid in relevant else 0.0
        if rel:
            dcg += rel / math.log2(i + 1)
    ideal_len = min(k, len(relevant))
    if ideal_len == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_len + 1))
    return dcg / idcg if idcg > 0 else 0.0


def _recall_at_k(ranked: list[str], relevant: set[str], k: int = 3) -> float:
    if not relevant:
        return 0.0
    hit = sum(1 for pid in ranked[:k] if pid in relevant)
    return hit / len(relevant)


def check_retry_backoff() -> dict[str, Any]:
    from app.db import qdrant as qdrant_mod

    settings = qdrant_mod.get_settings()
    settings.http_retry_max_attempts = 4
    settings.http_retry_backoff_base_sec = 0.01

    client = AlibabaEmbeddingClient(
        model="text-embedding-v4",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="dummy",
    )

    class FakeResp:
        def __init__(self, status_code: int, body: dict[str, Any] | None = None):
            self.status_code = status_code
            self._body = body or {}
            self.text = json.dumps(self._body)

        def json(self) -> dict[str, Any]:
            return self._body

    seq = [
        FakeResp(429, {"error": "rate limited"}),
        FakeResp(503, {"error": "temporarily unavailable"}),
        FakeResp(
            200,
            {
                "output": {
                    "embeddings": [
                        {"embedding": [0.1, 0.2, 0.3]},
                    ]
                }
            },
        ),
    ]
    post_calls = {"n": 0}
    sleep_calls: list[float] = []

    orig_post = qdrant_mod.requests.post
    orig_sleep = qdrant_mod.time.sleep
    try:
        def fake_post(*args, **kwargs):
            idx = post_calls["n"]
            post_calls["n"] += 1
            if idx < len(seq):
                return seq[idx]
            return seq[-1]

        def fake_sleep(sec: float):
            sleep_calls.append(sec)

        qdrant_mod.requests.post = fake_post
        qdrant_mod.time.sleep = fake_sleep
        vec = client.embed_query("retry test")
    finally:
        qdrant_mod.requests.post = orig_post
        qdrant_mod.time.sleep = orig_sleep

    return {
        "post_calls": post_calls["n"],
        "sleep_calls": sleep_calls,
        "vector_len": len(vec),
        "pass": post_calls["n"] == 3 and len(sleep_calls) == 2 and vec == [0.1, 0.2, 0.3],
    }


@dataclass
class _FakeStore:
    delay_sec: float = 0.03
    calls: int = 0

    def similarity_search_with_score(self, query: str, k: int = 4, filter=None, fetch_limit=None):
        self.calls += 1
        time.sleep(self.delay_sec)
        docs = [
            (
                Document(
                    page_content=f"query={query}, chunk={i}",
                    metadata={"paper_id": f"P{i}", "title": f"T{i}", "page_num": i},
                ),
                1.0 - i * 0.05,
            )
            for i in range(max(k, 6))
        ]
        if fetch_limit is not None:
            return docs[:fetch_limit]
        return docs[:k]


def check_cache_hit_rate() -> dict[str, Any]:
    settings = retriever_mod.get_settings()
    store = _FakeStore()

    orig_get_vs = retriever_mod.get_vector_store
    orig_cache = retriever_mod._retrieve_cache
    try:
        retriever_mod.get_vector_store = lambda: store

        settings.hybrid_retrieval_enabled = True
        settings.hybrid_oversample = 2.0
        settings.hybrid_alpha = 0.7
        settings.hybrid_max_fetch = 16

        # With cache
        settings.cache_retrieval_enabled = True
        settings.cache_retrieval_ttl_sec = 300
        settings.cache_retrieval_max_entries = 64
        retriever_mod._retrieve_cache = None
        t0 = time.perf_counter()
        for _ in range(20):
            retriever_mod.retrieve("cache-check-query", top_k=4)
        t_cache = time.perf_counter() - t0
        calls_with_cache = store.calls

        # Without cache
        store.calls = 0
        settings.cache_retrieval_enabled = False
        retriever_mod._retrieve_cache = None
        t1 = time.perf_counter()
        for _ in range(20):
            retriever_mod.retrieve("cache-check-query", top_k=4)
        t_no_cache = time.perf_counter() - t1
        calls_no_cache = store.calls
    finally:
        retriever_mod.get_vector_store = orig_get_vs
        retriever_mod._retrieve_cache = orig_cache

    latency_gain = (t_no_cache - t_cache) / t_no_cache if t_no_cache > 0 else 0.0
    hit_rate = 1.0 - calls_with_cache / 20.0
    return {
        "calls_with_cache": calls_with_cache,
        "calls_without_cache": calls_no_cache,
        "hit_rate_estimate": round(hit_rate, 4),
        "latency_cache_sec": round(t_cache, 4),
        "latency_no_cache_sec": round(t_no_cache, 4),
        "latency_gain_ratio": round(latency_gain, 4),
        "pass": calls_with_cache < calls_no_cache and t_cache < t_no_cache,
    }


def check_hybrid_quality_gain() -> dict[str, Any]:
    # Synthetic benchmark where lexical relevance should help reorder candidates.
    query_cases = [
        {
            "query": "attention mechanism in transformer",
            "docs": [
                ("A", "cooking pasta and tomato sauce", 0.93),
                ("B", "attention mechanism in transformer model", 0.89),
                ("C", "graph shortest path algorithm", 0.86),
            ],
            "relevant": {"B"},
        },
        {
            "query": "retrieval augmented generation citation",
            "docs": [
                ("D", "general machine learning overview", 0.92),
                ("E", "retrieval augmented generation with citation grounding", 0.90),
                ("F", "image segmentation baseline", 0.82),
            ],
            "relevant": {"E"},
        },
        {
            "query": "time decay memory update",
            "docs": [
                ("G", "random projection in kernels", 0.91),
                ("H", "streaming memory relevance gated updates and time decay", 0.90),
                ("I", "sports analytics ranking", 0.85),
            ],
            "relevant": {"H"},
        },
    ]

    # Use a fixed alpha for benchmark reproducibility across env configs.
    alpha = 0.55
    k = 3

    ndcg_vec: list[float] = []
    ndcg_hyb: list[float] = []
    recall_vec: list[float] = []
    recall_hyb: list[float] = []
    top1_vec: list[float] = []
    top1_hyb: list[float] = []

    for case in query_cases:
        docs_scores = [
            (
                Document(page_content=text, metadata={"paper_id": pid}),
                score,
            )
            for pid, text, score in case["docs"]
        ]
        vec_ranked = [d.metadata["paper_id"] for d, _ in docs_scores[:k]]
        hyb = retriever_mod._hybrid_fuse(case["query"], docs_scores, k, alpha=alpha)
        hyb_ranked = [d.metadata["paper_id"] for d, _ in hyb]
        rel = case["relevant"]

        ndcg_vec.append(_ndcg_at_k(vec_ranked, rel, k))
        ndcg_hyb.append(_ndcg_at_k(hyb_ranked, rel, k))
        recall_vec.append(_recall_at_k(vec_ranked, rel, k))
        recall_hyb.append(_recall_at_k(hyb_ranked, rel, k))
        top1_vec.append(1.0 if vec_ranked[0] in rel else 0.0)
        top1_hyb.append(1.0 if hyb_ranked[0] in rel else 0.0)

    mean_ndcg_vec = statistics.mean(ndcg_vec)
    mean_ndcg_hyb = statistics.mean(ndcg_hyb)
    mean_recall_vec = statistics.mean(recall_vec)
    mean_recall_hyb = statistics.mean(recall_hyb)
    mean_top1_vec = statistics.mean(top1_vec)
    mean_top1_hyb = statistics.mean(top1_hyb)

    return {
        "queries": len(query_cases),
        "ndcg_at_3_vector": round(mean_ndcg_vec, 4),
        "ndcg_at_3_hybrid": round(mean_ndcg_hyb, 4),
        "recall_at_3_vector": round(mean_recall_vec, 4),
        "recall_at_3_hybrid": round(mean_recall_hyb, 4),
        "top1_acc_vector": round(mean_top1_vec, 4),
        "top1_acc_hybrid": round(mean_top1_hyb, 4),
        "ndcg_gain": round(mean_ndcg_hyb - mean_ndcg_vec, 4),
        "recall_gain": round(mean_recall_hyb - mean_recall_vec, 4),
        "top1_gain": round(mean_top1_hyb - mean_top1_vec, 4),
        "pass": (
            mean_ndcg_hyb >= mean_ndcg_vec
            and mean_recall_hyb >= mean_recall_vec
            and mean_top1_hyb >= mean_top1_vec
        ),
    }


def check_observability_json() -> dict[str, Any]:
    import logging

    formatter = JsonLogFormatter()
    rec = logging.LogRecord(
        name="app.services.retriever",
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg="retrieve_done",
        args=(),
        exc_info=None,
    )
    rec.event = "rag.retrieve"
    rec.request_id = "obs-check-001"
    rec.phase = "done"
    rec.ms = 12.34
    rec.chunks = 3
    line = formatter.format(rec)
    obj = json.loads(line)
    must = {"ts", "level", "logger", "message", "event", "request_id", "phase", "ms", "chunks"}
    return {
        "json_line": line,
        "keys": sorted(obj.keys()),
        "pass": must.issubset(obj.keys()) and obj["request_id"] == "obs-check-001",
    }


def main() -> None:
    report = {
        "retry_backoff": check_retry_backoff(),
        "cache_hit_rate": check_cache_hit_rate(),
        "hybrid_vs_vector": check_hybrid_quality_gain(),
        "observability_json": check_observability_json(),
    }
    report["all_pass"] = all(v.get("pass", False) for v in report.values())
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
