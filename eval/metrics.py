# eval/metrics.py
"""Standard RAG retrieval metrics: NDCG@k, Precision@k, Context Precision."""
from __future__ import annotations

import math


def ndcg_at_k(pred_pids: list[str], expected_pids: set[str], k: int = 5) -> float:
    if not expected_pids:
        return 0.0
    pred = pred_pids[:k]
    dcg = 0.0
    for i, pid in enumerate(pred):
        if pid in expected_pids:
            dcg += 1.0 / math.log2(i + 2)
    ideal_hits = min(len(expected_pids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def precision_at_k(pred_pids: list[str], expected_pids: set[str], k: int = 5) -> float:
    if not expected_pids:
        return 0.0
    pred = pred_pids[:k]
    if not pred:
        return 0.0
    hits = sum(1 for pid in pred if pid in expected_pids)
    return hits / len(pred)


def context_precision(cited_pids: list[str], retrieved_pids: list[str]) -> float:
    if not retrieved_pids:
        return 0.0
    cited_set = set(cited_pids)
    hits = sum(1 for pid in retrieved_pids if pid in cited_set)
    return hits / len(retrieved_pids)
