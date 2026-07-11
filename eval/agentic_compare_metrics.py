"""Metrics shared by traditional-vs-agentic RAG evaluation."""
from __future__ import annotations

import math
import re
from collections import defaultdict
from statistics import mean

CITATION_RE = re.compile(
    r"(?:\[?\s*arxiv:|https?://arxiv\.org/abs/)([0-9]{4}\.[0-9]{4,6})(?:v\d+)?\]?",
    re.IGNORECASE,
)
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.IGNORECASE | re.DOTALL)
ABSTENTION_TERMS = (
    "信息不足",
    "参考资料不足",
    "无法回答",
    "无法确定",
    "无法基于现有语料",
    "没有足够",
    "不在语料",
    "insufficient information",
    "cannot answer",
    "not enough information",
    "no evidence",
)


def extract_citation_pids(answer: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for pid in CITATION_RE.findall(answer or ""):
        if pid not in seen:
            found.append(pid)
            seen.add(pid)
    return found


def strip_thinking_for_metrics(answer: str) -> str:
    """Remove model reasoning blocks from eval-only answer metrics."""
    return THINK_BLOCK_RE.sub("", answer or "").strip()


def detect_abstention(answer: str) -> bool:
    text = strip_thinking_for_metrics(answer).strip().lower()
    if not text:
        return False

    content_lines = [
        line.strip().lstrip("#>*- ")
        for line in text.splitlines()
        if line.strip().lstrip("#>*- ")
    ]
    while content_lines and content_lines[0] in {"回答", "answer", "结论", "conclusion"}:
        content_lines.pop(0)
    if not content_lines:
        return False

    opening = content_lines[0]
    sentence_end = re.search(r"[。！？!?]", opening)
    if sentence_end:
        opening = opening[: sentence_end.end()]
    return any(term.lower() in opening for term in ABSTENTION_TERMS)


def _unique(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        clean = str(item or "").strip()
        if clean and clean not in seen:
            out.append(clean)
            seen.add(clean)
    return out


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


def answer_case_metrics(
    *,
    qid: str,
    qtype: str,
    difficulty: str,
    expected_paper_ids: list[str],
    expected_mode: str,
    answer: str,
    source_pids: list[str],
    context_pids: list[str] | None = None,
    latency_s: float | None = None,
    used_chunks: int | None = None,
    step_count: int | None = None,
    retrieval_step_count: int | None = None,
    error: str | None = None,
) -> dict:
    expected = set(_unique(expected_paper_ids))
    sources = set(_unique(source_pids))
    contexts = set(_unique(context_pids if context_pids is not None else source_pids))
    citations = extract_citation_pids(answer)
    cited = set(citations)
    is_negative = expected_mode in {"insufficient", "refuse"} or not expected
    abstained = detect_abstention(answer)

    row: dict = {
        "qid": qid,
        "type": qtype,
        "difficulty": difficulty,
        "expected_mode": expected_mode,
        "expected_pids": sorted(expected),
        "source_pids": sorted(sources),
        "context_pids": sorted(contexts),
        "has_expected": bool(expected),
        "is_negative": is_negative,
        "answer_abstained": abstained,
        "mode_correct": False if error else (abstained if is_negative else not abstained),
        "citation_pids": citations,
        "citation_count": len(citations),
    }
    if latency_s is not None:
        row["latency_s"] = round(float(latency_s), 4)
    if used_chunks is not None:
        row["used_chunks"] = int(used_chunks)
    if step_count is not None:
        row["step_count"] = int(step_count)
    if retrieval_step_count is not None:
        row["retrieval_step_count"] = int(retrieval_step_count)
    if error:
        row["error"] = error

    if expected:
        row["source_hit"] = 1.0 if sources & expected else 0.0
        row["source_recall"] = len(sources & expected) / len(expected)
        row["source_precision"] = len(sources & expected) / len(sources) if sources else 0.0
    else:
        row["source_hit"] = None
        row["source_recall"] = None
        row["source_precision"] = None

    if citations:
        row["citation_support_rate"] = len(cited & contexts) / len(cited)
        if expected:
            row["citation_precision"] = len(cited & expected) / len(cited)
            row["citation_expected_hit"] = 1.0 if cited & expected else 0.0
        else:
            row["citation_precision"] = None
            row["citation_expected_hit"] = None
    else:
        row["citation_support_rate"] = None
        row["citation_precision"] = 0.0 if expected else None
        row["citation_expected_hit"] = 0.0 if expected else None
    return row


def _avg(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return mean(values) if values else None


def _summarize_bucket(rows: list[dict]) -> dict:
    positives = [row for row in rows if row.get("has_expected")]
    negatives = [row for row in rows if row.get("is_negative")]
    latencies = [float(row["latency_s"]) for row in rows if row.get("latency_s") is not None]
    return {
        "count": len(rows),
        "count_positive": len(positives),
        "count_negative": len(negatives),
        "mode_accuracy": _round(_avg(rows, "mode_correct")),
        "source_hit": _round(_avg(positives, "source_hit")),
        "source_recall": _round(_avg(positives, "source_recall")),
        "source_precision": _round(_avg(positives, "source_precision")),
        "citation_support_rate": _round(_avg(rows, "citation_support_rate")),
        "citation_precision": _round(_avg(positives, "citation_precision")),
        "citation_expected_hit": _round(_avg(positives, "citation_expected_hit")),
        "used_chunks_mean": _round(_avg(rows, "used_chunks")),
        "step_count_mean": _round(_avg(rows, "step_count")),
        "retrieval_step_count_mean": _round(_avg(rows, "retrieval_step_count")),
        "latency_p50": _round(_percentile(latencies, 0.5)),
        "latency_p90": _round(_percentile(latencies, 0.9)),
        "latency_mean": _round(mean(latencies), 4) if latencies else None,
        "error_count": sum(1 for row in rows if row.get("error")),
    }


def _breakdown(rows: list[dict], key: str) -> dict:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(key) or "unknown")].append(row)
    return {name: _summarize_bucket(bucket) for name, bucket in sorted(buckets.items())}


def summarize_answer_cases(rows: list[dict]) -> dict:
    summary = _summarize_bucket(rows)
    summary["by_type"] = _breakdown(rows, "type")
    summary["by_difficulty"] = _breakdown(rows, "difficulty")
    return summary


def compare_system_summaries(
    *,
    baseline: dict,
    candidate: dict,
    keys: list[str] | None = None,
) -> list[dict]:
    keys = keys or [
        "mode_accuracy",
        "source_recall",
        "source_precision",
        "citation_support_rate",
        "citation_precision",
        "citation_expected_hit",
        "latency_p90",
        "used_chunks_mean",
    ]
    rows = []
    for key in keys:
        b = baseline.get(key)
        c = candidate.get(key)
        absolute = None
        relative = None
        if b is not None and c is not None:
            absolute = round(float(c) - float(b), 4)
            if abs(float(b)) > 1e-12:
                relative = round(absolute / float(b) * 100, 2)
        rows.append(
            {
                "metric": key,
                "baseline": b,
                "candidate": c,
                "absolute_delta": absolute,
                "relative_delta_pct": relative,
            }
        )
    return rows


def select_stratified_questions(
    questions: list[dict],
    *,
    per_type: int | None = None,
) -> list[dict]:
    if per_type is None:
        return list(questions)
    buckets: dict[str, list[dict]] = defaultdict(list)
    type_order: list[str] = []
    for question in questions:
        qtype = str(question.get("type") or "unknown")
        if qtype not in buckets:
            type_order.append(qtype)
        buckets[qtype].append(question)
    selected: list[dict] = []
    for qtype in type_order:
        selected.extend(buckets[qtype][:per_type])
    return selected


def select_proportional_questions(
    questions: list[dict],
    *,
    sample_size: int | None = None,
) -> list[dict]:
    if sample_size is None or sample_size >= len(questions):
        return list(questions)
    if sample_size <= 0:
        return []

    buckets: dict[str, list[dict]] = defaultdict(list)
    type_order: list[str] = []
    for question in questions:
        qtype = str(question.get("type") or "unknown")
        if qtype not in buckets:
            type_order.append(qtype)
        buckets[qtype].append(question)

    if sample_size < len(type_order):
        return list(questions[:sample_size])

    total = len(questions)
    allocations: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for qtype in type_order:
        exact = len(buckets[qtype]) / total * sample_size
        base = max(1, int(math.floor(exact)))
        base = min(base, len(buckets[qtype]))
        allocations[qtype] = base
        remainders.append((exact - math.floor(exact), qtype))

    allocated = sum(allocations.values())
    for _, qtype in sorted(remainders, reverse=True):
        if allocated >= sample_size:
            break
        if allocations[qtype] < len(buckets[qtype]):
            allocations[qtype] += 1
            allocated += 1

    if allocated > sample_size:
        for _, qtype in sorted(remainders):
            while allocated > sample_size and allocations[qtype] > 1:
                allocations[qtype] -= 1
                allocated -= 1
            if allocated == sample_size:
                break

    selected: list[dict] = []
    for qtype in type_order:
        selected.extend(buckets[qtype][: allocations[qtype]])
    return selected
