"""Pure RAG evaluation helpers.

This module is intentionally independent from the application runtime. It
accepts plain dictionaries produced by any retriever/generator runner and
computes stable retrieval/context metrics for resume-safe reporting.
"""
from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean
from typing import Iterable

from eval.metrics import ndcg_at_k, precision_at_k


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * p) - 1))
    return float(ordered[idx])


def unique_ranked_pids(retrieved_chunks: Iterable[dict]) -> list[str]:
    ranked: list[str] = []
    seen: set[str] = set()
    for chunk in retrieved_chunks:
        pid = str(chunk.get("paper_id") or "").strip()
        if pid and pid not in seen:
            ranked.append(pid)
            seen.add(pid)
    return ranked


def recall_at_k(pred_pids: list[str], expected_pids: set[str], k: int) -> float:
    if not expected_pids:
        return 0.0
    return len(set(pred_pids[:k]) & expected_pids) / len(expected_pids)


def hit_at_k(pred_pids: list[str], expected_pids: set[str], k: int) -> float:
    if not expected_pids:
        return 0.0
    return 1.0 if set(pred_pids[:k]) & expected_pids else 0.0


def first_relevant_rank(pred_pids: list[str], expected_pids: set[str]) -> int | None:
    if not expected_pids:
        return None
    for idx, pid in enumerate(pred_pids, start=1):
        if pid in expected_pids:
            return idx
    return None


def evaluate_retrieval_case(
    *,
    qid: str,
    query: str,
    expected_paper_ids: list[str],
    expected_mode: str,
    difficulty: str,
    qtype: str,
    retrieved_chunks: list[dict],
    k_values: list[int],
    context_k: int,
    latency_s: float | None = None,
) -> dict:
    expected = {str(pid) for pid in expected_paper_ids if str(pid).strip()}
    ranked_pids = unique_ranked_pids(retrieved_chunks)
    is_negative = qtype == "negative" or expected_mode in {"insufficient", "refuse"}

    row: dict = {
        "qid": qid,
        "query": query,
        "difficulty": difficulty,
        "type": qtype,
        "expected_mode": expected_mode,
        "expected_pids": sorted(expected),
        "ranked_pids": ranked_pids,
        "retrieved_chunk_count": len(retrieved_chunks),
        "retrieved_unique_paper_count": len(ranked_pids),
        "has_expected": bool(expected),
        "is_negative": is_negative,
    }
    if latency_s is not None:
        row["latency_s"] = round(float(latency_s), 4)

    scores = [chunk.get("score") for chunk in retrieved_chunks if chunk.get("score") is not None]
    if scores:
        row["max_score"] = max(float(score) for score in scores)

    if not expected:
        row["negative_context_count"] = len(retrieved_chunks[:context_k])
        row["negative_max_score"] = row.get("max_score")
        return row

    rank = first_relevant_rank(ranked_pids, expected)
    row["first_relevant_rank"] = rank
    row["mrr"] = 1.0 / rank if rank else 0.0

    for k in sorted(set(k_values)):
        row[f"hit_at_{k}"] = hit_at_k(ranked_pids, expected, k)
        row[f"recall_at_{k}"] = recall_at_k(ranked_pids, expected, k)
        row[f"precision_at_{k}"] = precision_at_k(ranked_pids, expected, k)
        row[f"ndcg_at_{k}"] = ndcg_at_k(ranked_pids, expected, k)
        row[f"all_expected_at_{k}"] = 1.0 if recall_at_k(ranked_pids, expected, k) >= 1.0 else 0.0

    context_chunks = retrieved_chunks[:context_k]
    context_chunk_pids = [
        str(chunk.get("paper_id") or "").strip()
        for chunk in context_chunks
        if str(chunk.get("paper_id") or "").strip()
    ]
    context_unique = unique_ranked_pids(context_chunks)
    relevant_context_chunks = sum(1 for pid in context_chunk_pids if pid in expected)
    context_chunk_precision = (
        relevant_context_chunks / len(context_chunk_pids)
        if context_chunk_pids
        else 0.0
    )
    context_unique_precision = (
        len(set(context_unique) & expected) / len(context_unique)
        if context_unique
        else 0.0
    )
    context_recall = len(set(context_unique) & expected) / len(expected)

    row["context_k"] = context_k
    row["context_chunk_precision"] = context_chunk_precision
    row["context_unique_precision"] = context_unique_precision
    row["context_recall"] = context_recall
    row["context_noise_rate"] = 1.0 - context_chunk_precision
    return row


def _avg(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return mean(values)


def _summarize_bucket(rows: list[dict], k_values: list[int]) -> dict:
    positives = [row for row in rows if row.get("has_expected")]
    out: dict = {
        "count": len(rows),
        "count_positive": len(positives),
        "count_negative": len(rows) - len(positives),
    }

    for k in sorted(set(k_values)):
        for prefix in ("hit", "recall", "precision", "ndcg", "all_expected"):
            key = f"{prefix}_at_{k}"
            out[key] = _round(_avg(positives, key))

    for key in (
        "mrr",
        "context_chunk_precision",
        "context_unique_precision",
        "context_recall",
        "context_noise_rate",
    ):
        out[key] = _round(_avg(positives, key))

    latencies = [float(row["latency_s"]) for row in rows if row.get("latency_s") is not None]
    out["latency_p50"] = _round(_percentile(latencies, 0.5), 4)
    out["latency_p90"] = _round(_percentile(latencies, 0.9), 4)
    out["latency_mean"] = _round(mean(latencies), 4) if latencies else None

    negatives = [row for row in rows if row.get("is_negative")]
    if negatives:
        with_context = sum(1 for row in negatives if row.get("negative_context_count", 0) > 0)
        out["negative_with_context_rate"] = _round(with_context / len(negatives))
        out["negative_max_score_mean"] = _round(_avg(negatives, "negative_max_score"))
    else:
        out["negative_with_context_rate"] = None
        out["negative_max_score_mean"] = None

    return out


def _breakdown(rows: list[dict], key: str, k_values: list[int]) -> dict:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(key) or "unknown")].append(row)
    return {
        bucket: _summarize_bucket(bucket_rows, k_values)
        for bucket, bucket_rows in sorted(buckets.items())
    }


def summarize_retrieval_cases(cases: list[dict], k_values: list[int]) -> dict:
    summary = _summarize_bucket(cases, k_values)
    summary["by_difficulty"] = _breakdown(cases, "difficulty", k_values)
    summary["by_type"] = _breakdown(cases, "type", k_values)
    return summary


def compare_metric(metric: str, *, baseline: float | None, candidate: float | None) -> dict:
    absolute_delta = None
    relative_delta_pct = None
    if baseline is not None and candidate is not None:
        absolute_delta = round(float(candidate) - float(baseline), 4)
        if abs(float(baseline)) > 1e-12:
            relative_delta_pct = round((absolute_delta / float(baseline)) * 100, 2)
    return {
        "metric": metric,
        "baseline": baseline,
        "candidate": candidate,
        "absolute_delta": absolute_delta,
        "relative_delta_pct": relative_delta_pct,
    }


def render_markdown_report(
    *,
    run_id: str,
    dataset_name: str,
    summary: dict,
    comparisons: list[dict] | None = None,
) -> str:
    lines = [
        f"# Pure RAG Evaluation Report: {run_id}",
        "",
        f"- Dataset: `{dataset_name}`",
        f"- Questions: {summary.get('count', 0)} total, {summary.get('count_positive', 0)} positive, {summary.get('count_negative', 0)} negative",
        "",
        "## Core Retrieval Metrics",
        "",
        f"- NDCG@5: {_format_metric(summary.get('ndcg_at_5'))}",
        f"- Recall@5: {_format_metric(summary.get('recall_at_5'))}",
        f"- MRR: {_format_metric(summary.get('mrr'))}",
        f"- Context chunk precision: {_format_metric(summary.get('context_chunk_precision'))}",
        f"- Context recall: {_format_metric(summary.get('context_recall'))}",
        f"- Latency P90: {_format_metric(summary.get('latency_p90'))}s",
        f"- Negative with context rate: {_format_metric(summary.get('negative_with_context_rate'))}",
        f"- Negative max score mean: {_format_metric(summary.get('negative_max_score_mean'))}",
        "",
        "## Resume-safe metrics",
        "",
        "- Use only metrics from this section in external materials unless a newer report supersedes it.",
        f"- NDCG@5: {_format_metric(summary.get('ndcg_at_5'))}",
        f"- Recall@5: {_format_metric(summary.get('recall_at_5'))}",
        f"- MRR: {_format_metric(summary.get('mrr'))}",
        f"- Context precision: {_format_metric(summary.get('context_chunk_precision'))}",
        f"- Context recall: {_format_metric(summary.get('context_recall'))}",
    ]

    if summary.get("generation_count"):
        lines.extend(
            [
                "",
                "## Fixed-Context Generation",
                "",
                f"- Generation cases: {summary.get('generation_count')}",
                f"- Mode accuracy: {_format_metric(summary.get('mode_accuracy'))}",
                f"- Citation support rate: {_format_metric(summary.get('citation_support_rate'))}",
                f"- Citation precision: {_format_metric(summary.get('citation_precision'))}",
                f"- Citation expected hit: {_format_metric(summary.get('citation_expected_hit'))}",
            ]
        )

    if comparisons:
        lines.extend(["", "## Comparisons", ""])
        for item in comparisons:
            rel = item.get("relative_delta_pct")
            rel_text = "n/a" if rel is None else f"{rel:+.2f}%"
            lines.append(
                f"- {item['metric']}: {_format_metric(item.get('baseline'))} -> "
                f"{_format_metric(item.get('candidate'))} ({rel_text})"
            )

    by_type = summary.get("by_type") or {}
    if by_type:
        lines.extend(["", "## Breakdown by Type", ""])
        lines.append("| Type | Count | NDCG@5 | Recall@5 | Context recall |")
        lines.append("|---|---:|---:|---:|---:|")
        for name, bucket in by_type.items():
            lines.append(
                f"| {name} | {bucket.get('count', 0)} | "
                f"{_format_metric(bucket.get('ndcg_at_5'))} | "
                f"{_format_metric(bucket.get('recall_at_5'))} | "
                f"{_format_metric(bucket.get('context_recall'))} |"
            )

    return "\n".join(lines) + "\n"


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"
