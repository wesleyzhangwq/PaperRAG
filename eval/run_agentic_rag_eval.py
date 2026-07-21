"""Compare fixed-context traditional RAG with the full Agentic RAG graph.

The traditional branch is intentionally plain: one local retrieval call, a
fixed top-k context window, and one answer-generation call. The agentic branch
uses the production LangGraph pipeline through ``run_agent_sync``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shlex
import subprocess
import sys
import time
from contextlib import ExitStack
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))

from eval.agentic_compare_metrics import (  # noqa: E402
    answer_case_metrics,
    compare_system_summaries,
    extract_citation_pids,
    select_proportional_questions,
    select_stratified_questions,
    strip_thinking_for_metrics,
    summarize_answer_cases,
)
from eval.costing import (  # noqa: E402
    DEFAULT_CATALOG_PATH,
    attribute_llm_costs,
    load_pricing_catalog,
)
from eval.run_rag_eval import (  # noqa: E402
    _build_context,
    build_retrieved_chunks,
    generate_fixed_context_answer,
    load_questions,
    postprocess_chunks,
    redact_sensitive_text,
)

RETRIEVAL_ACTIONS = {
    "retrieve_local",
    "retrieve_arxiv",
    "search_web",
    "get_paper_detail",
    "get_paper_chunks",
}
RESUME_CONTRACT_SCHEMA_VERSION = "1.0"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _evaluation_source_sha256() -> str:
    """Hash evaluation and runtime source while excluding generated outputs."""
    source_files = [
        path
        for path in (BACKEND_ROOT / "app").rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix in {".py", ".json", ".md", ".txt", ".yaml", ".yml"}
    ]
    source_files.extend(path for path in (PROJECT_ROOT / "eval").glob("*.py") if path.is_file())
    source_files.extend(
        path
        for path in (PROJECT_ROOT / "eval" / "pricing").rglob("*.json")
        if path.is_file()
    )
    digest = hashlib.sha256()
    for path in sorted(set(source_files)):
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def _git_snapshot() -> dict:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    status_lines = [line for line in run("status", "--porcelain").splitlines() if line]
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status_lines),
        "dirty_entry_count": len(status_lines),
    }


def _selected_qids(questions: list[dict]) -> list[str]:
    qids: list[str] = []
    seen: set[str] = set()
    for index, question in enumerate(questions, start=1):
        qid = str(question.get("qid") or "").strip()
        if not qid:
            raise ValueError(f"Selected question {index} has no qid.")
        if qid in seen:
            raise ValueError(f"Selected questions contain duplicate qid: {qid}")
        qids.append(qid)
        seen.add(qid)
    return qids


def _build_resume_contract(
    *,
    dataset_path: Path,
    pricing_catalog_path: Path,
    questions: list[dict],
    selection: dict,
    traditional: dict,
    agentic: dict,
    settings: dict,
    execution: dict,
    pricing: dict,
    git_commit: str,
) -> dict:
    """Freeze every input that could change the meaning of persisted rows."""
    return {
        "schema_version": RESUME_CONTRACT_SCHEMA_VERSION,
        "dataset": {
            "path": str(dataset_path.expanduser().resolve()),
            "sha256": _sha256_file(dataset_path),
            "selected_qids": _selected_qids(questions),
        },
        "selection": dict(selection),
        "traditional": dict(traditional),
        "agentic": dict(agentic),
        "settings": dict(settings),
        "execution": {
            key: execution.get(key)
            for key in (
                "concurrency",
                "warmup",
                "request_timeout_s",
                "external_api_allowed",
            )
        },
        "pricing": {
            **dict(pricing),
            "catalog_path": str(pricing_catalog_path.expanduser().resolve()),
            "catalog_sha256": _sha256_file(pricing_catalog_path),
        },
        "implementation": {
            "git_commit": git_commit,
            "runner_sha256": _sha256_file(Path(__file__)),
            "evaluation_source_sha256": _evaluation_source_sha256(),
        },
    }


def _resume_contract_mismatches(
    expected: Any,
    actual: Any,
    *,
    path: str = "resume_contract",
) -> list[str]:
    if type(expected) is not type(actual):
        return [f"{path} type changed"]
    if isinstance(expected, dict):
        mismatches: list[str] = []
        for key in sorted(set(expected) | set(actual)):
            child_path = f"{path}.{key}"
            if key not in expected:
                mismatches.append(f"{child_path} was added")
            elif key not in actual:
                mismatches.append(f"{child_path} is missing")
            else:
                mismatches.extend(
                    _resume_contract_mismatches(
                        expected[key], actual[key], path=child_path
                    )
                )
        return mismatches
    if isinstance(expected, list):
        if expected == actual:
            return []
        return [f"{path} changed (including order)"]
    return [] if expected == actual else [f"{path} changed"]


def _validate_resume_contract(manifest: dict, current_contract: dict) -> None:
    stored_contract = manifest.get("resume_contract")
    if not isinstance(stored_contract, dict):
        raise ValueError(
            "Cannot resume: manifest has no immutable resume_contract; start a new run."
        )
    mismatches = _resume_contract_mismatches(stored_contract, current_contract)
    if mismatches:
        detail = "; ".join(mismatches[:12])
        if len(mismatches) > 12:
            detail += f"; plus {len(mismatches) - 12} more"
        raise ValueError(f"Cannot resume: immutable evaluation inputs changed: {detail}")


def _attach_usage_cost(
    row: dict,
    records: list[dict],
    *,
    billing_origin: str,
    pricing_catalog: dict,
) -> None:
    attribution = attribute_llm_costs(
        records,
        billing_origin=billing_origin,
        catalog=pricing_catalog,
    )
    row["llm_usage"] = attribution.pop("calls")
    row["llm_usage_totals"] = attribution
    for key in (
        "input_tokens",
        "output_tokens",
        "cached_read_tokens",
        "cache_write_tokens",
        "total_tokens",
        "usage_status",
        "cache_usage_status",
        "cached_read_unknown_call_count",
        "cache_write_unknown_call_count",
        "cost_usd",
        "known_partial_cost_usd",
        "cost_status",
        "pricing_catalog_version",
        "billing_origin",
        "cost_scope",
    ):
        row[key] = attribution.get(key)


def _stage_timings(traces: list[Any] | None) -> list[dict]:
    timings = []
    for trace in traces or []:
        if isinstance(trace, dict):
            node = trace.get("node")
            action = trace.get("action")
            duration = trace.get("duration_ms")
        else:
            node = getattr(trace, "node", None)
            action = getattr(trace, "action", None)
            duration = getattr(trace, "duration_ms", None)
        if duration is None:
            continue
        timings.append(
            {
                "node": str(node or ""),
                "action": str(action or node or "unknown"),
                "duration_ms": round(float(duration), 3),
            }
        )
    return timings


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key not in fieldnames and not isinstance(value, (list, dict)):
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_cases_with_checkpoint(
    questions: list[dict],
    *,
    existing_rows: list[dict],
    run_case: Callable[[dict, int, int], dict],
    persist: Callable[[list[dict]], None],
) -> list[dict]:
    """Run missing cases in order and persist after each completed case."""
    selected_qids = _selected_qids(questions)
    selected_qid_set = set(selected_qids)
    rows_by_qid: dict[str, dict] = {}
    for row in existing_rows:
        qid = str(row.get("qid") or "").strip()
        if not qid:
            raise ValueError("Checkpoint row has no qid.")
        if qid not in selected_qid_set:
            raise ValueError(f"Checkpoint row qid is not in the frozen selection: {qid}")
        if qid in rows_by_qid:
            raise ValueError(f"Checkpoint contains duplicate qid: {qid}")
        rows_by_qid[qid] = dict(row)
    total = len(questions)

    def ordered_rows() -> list[dict]:
        return [rows_by_qid[qid] for qid in selected_qids if qid in rows_by_qid]

    for idx, item in enumerate(questions, start=1):
        qid = str(item["qid"])
        if qid in rows_by_qid:
            continue
        row = dict(run_case(item, idx, total))
        if str(row.get("qid") or "") != qid:
            raise ValueError(f"Case runner returned qid {row.get('qid')!r} for {qid!r}")
        rows_by_qid[qid] = row
        persist(ordered_rows())
    return ordered_rows()


def _question_meta(item: dict, idx: int) -> dict:
    return {
        "qid": item.get("qid") or f"q{idx}",
        "query": item["query"],
        "type": item.get("type", "unknown"),
        "difficulty": item.get("difficulty", "unknown"),
        "expected_paper_ids": item.get("expected_paper_ids") or [],
        "expected_mode": item.get("expected_mode", "answer"),
        "reference_answer": item.get("reference_answer", ""),
    }


def _top_context_pids(chunks: list[dict], context_k: int) -> list[str]:
    pids: list[str] = []
    seen: set[str] = set()
    for chunk in chunks[:context_k]:
        pid = str(chunk.get("paper_id") or "").strip()
        if pid and pid not in seen:
            pids.append(pid)
            seen.add(pid)
    return pids


def _document_pids(documents: list[Any]) -> list[str]:
    pids: list[str] = []
    seen: set[str] = set()
    for document in documents:
        pid = str((getattr(document, "metadata", {}) or {}).get("paper_id") or "").strip()
        if pid and pid not in seen:
            pids.append(pid)
            seen.add(pid)
    return pids


def _trace_action(trace: Any) -> str:
    if isinstance(trace, dict):
        return str(trace.get("action") or "")
    return str(getattr(trace, "action", "") or "")


def _trace_actions(traces: list[Any] | None) -> list[str]:
    return [_trace_action(trace) for trace in traces or [] if _trace_action(trace)]


def _error_row(*, system: str, meta: dict, latency_s: float, exc: Exception) -> dict:
    telemetry = {
        # A top-level terminal exception does not prove that a recovery path
        # was attempted.  Keep it outside the fallback recovery denominator.
        "fallback_attempted": False,
        "fallback_recovered": False,
        "re_retrieve_count": 0,
        "re_generate_count": 0,
        "degraded_answer": False,
        "terminal_failure": True,
        "failure_class": "terminal_exception",
        "failure_classes": ["terminal_exception"],
        "events": [
            {
                "stage": "request",
                "failure_class": "terminal_exception",
                "outcome": "terminal_failure",
            }
        ],
    }
    row = answer_case_metrics(
        qid=meta["qid"],
        qtype=meta["type"],
        difficulty=meta["difficulty"],
        expected_paper_ids=meta["expected_paper_ids"],
        expected_mode=meta["expected_mode"],
        answer="",
        source_pids=[],
        latency_s=latency_s,
        used_chunks=0,
        step_count=0,
        retrieval_step_count=0,
        error=type(exc).__name__,
        terminal_failure=True,
        request_completed=False,
        fallback_telemetry=telemetry,
    )
    row["system"] = system
    row["query"] = meta["query"]
    row["answer"] = ""
    row["mode_correct"] = False
    row["fallback_telemetry"] = telemetry
    row["stage_timings"] = []
    return row


def run_traditional_case(
    item: dict,
    *,
    idx: int,
    total: int,
    context_k: int,
    retrieval_top_k: int | None,
    context_strategy: str,
    mmr_lambda: float,
    billing_origin: str,
    pricing_catalog: dict,
) -> dict:
    from app.services.retriever import retrieve

    meta = _question_meta(item, idx)
    print(f"[traditional {idx}/{total}] {meta['qid']}", file=sys.stderr)
    t0 = time.perf_counter()
    from app.observability.llm_usage import collect_llm_usage

    with collect_llm_usage() as usage_collector:
        try:
            retrieve_t0 = time.perf_counter()
            results = retrieve(meta["query"], top_k=retrieval_top_k)
            retrieval_latency_s = time.perf_counter() - retrieve_t0
            retrieved_paper_ids = _document_pids([document for document, _ in results])
            chunks = postprocess_chunks(
                build_retrieved_chunks(results),
                strategy=context_strategy,
                context_k=context_k,
                mmr_lambda=mmr_lambda,
            )
            context = _build_context(chunks, context_k)
            generation_t0 = time.perf_counter()
            answer = generate_fixed_context_answer(meta["query"], context)
            generation_latency_s = time.perf_counter() - generation_t0
            total_latency_s = time.perf_counter() - t0
        except Exception as exc:
            row = _error_row(
                system="traditional_rag",
                meta=meta,
                latency_s=time.perf_counter() - t0,
                exc=exc,
            )
            _attach_usage_cost(
                row,
                usage_collector.snapshot(),
                billing_origin=billing_origin,
                pricing_catalog=pricing_catalog,
            )
            return row

    answer = redact_sensitive_text(answer)
    answer_for_metrics = strip_thinking_for_metrics(answer)
    row = answer_case_metrics(
        qid=meta["qid"],
        qtype=meta["type"],
        difficulty=meta["difficulty"],
        expected_paper_ids=meta["expected_paper_ids"],
        expected_mode=meta["expected_mode"],
        answer=answer_for_metrics,
        source_pids=retrieved_paper_ids,
        context_pids=_top_context_pids(chunks, context_k),
        latency_s=total_latency_s,
        used_chunks=min(len(chunks), context_k),
        step_count=2,
        retrieval_step_count=1,
        final_source_pids=extract_citation_pids(answer_for_metrics),
    )
    row.update(
        {
            "system": "traditional_rag",
            "query": meta["query"],
            "answer": answer,
            "answer_for_metrics": answer_for_metrics,
            "retrieval_latency_s": round(retrieval_latency_s, 4),
            "generation_latency_s": round(generation_latency_s, 4),
            "retrieval_source_count": len(retrieved_paper_ids),
            "retrieved_chunks": chunks[:context_k],
        }
    )
    _attach_usage_cost(
        row,
        usage_collector.snapshot(),
        billing_origin=billing_origin,
        pricing_catalog=pricing_catalog,
    )
    return row


def run_agentic_case(
    item: dict,
    *,
    idx: int,
    total: int,
    run_id: str,
    db: Any,
    local_only: bool = True,
    billing_origin: str = "unknown",
    pricing_catalog: dict | None = None,
) -> dict:
    from app.agent.graph import run_agent_eval_sync
    from app.observability.llm_usage import collect_llm_usage

    pricing_catalog = pricing_catalog or load_pricing_catalog()

    meta = _question_meta(item, idx)
    print(f"[agentic {idx}/{total}] {meta['qid']}", file=sys.stderr)
    t0 = time.perf_counter()
    with collect_llm_usage() as usage_collector:
        try:
            with ExitStack() as stack:
                if local_only:
                    stack.enter_context(
                        patch(
                            "app.agent.nodes.executor.retrieve_arxiv_tool",
                            SimpleNamespace(
                                invoke=lambda _params: "Evaluation external arXiv intentionally disabled."
                            ),
                        )
                    )
                    stack.enter_context(
                        patch(
                            "app.agent.nodes.executor.search_web_tool",
                            SimpleNamespace(
                                invoke=lambda _params: "Evaluation external web intentionally disabled."
                            ),
                        )
                    )
                response, retrieved_paper_ids, context_paper_ids = run_agent_eval_sync(
                    db,
                    meta["query"],
                    session_id=f"{run_id}-{meta['qid']}",
                    history=None,
                )
            latency_s = time.perf_counter() - t0
        except Exception as exc:
            row = _error_row(
                system="agentic_rag",
                meta=meta,
                latency_s=time.perf_counter() - t0,
                exc=exc,
            )
            _attach_usage_cost(
                row,
                usage_collector.snapshot(),
                billing_origin=billing_origin,
                pricing_catalog=pricing_catalog,
            )
            return row

    traces = response.step_traces or []
    actions = _trace_actions(traces)
    answer = redact_sensitive_text(response.answer or "")
    answer_for_metrics = strip_thinking_for_metrics(answer)
    final_source_pids = [
        str(getattr(source, "paper_id", "") or "").strip()
        for source in response.sources or []
        if str(getattr(source, "paper_id", "") or "").strip()
    ]
    telemetry = dict(response.fallback_telemetry or {})
    presentation = dict(response.presentation or {})
    degraded_answer = bool(response.degraded or telemetry.get("degraded_answer"))
    row = answer_case_metrics(
        qid=meta["qid"],
        qtype=meta["type"],
        difficulty=meta["difficulty"],
        expected_paper_ids=meta["expected_paper_ids"],
        expected_mode=meta["expected_mode"],
        answer=answer_for_metrics,
        source_pids=retrieved_paper_ids,
        context_pids=context_paper_ids,
        latency_s=latency_s,
        used_chunks=response.used_chunks,
        step_count=len(traces),
        retrieval_step_count=sum(1 for action in actions if action in RETRIEVAL_ACTIONS),
        final_source_pids=final_source_pids,
        presentation=presentation,
        sufficiency_result=response.sufficiency_result,
        removed_citation_pids=response.removed_citations,
        degraded_answer=degraded_answer,
        terminal_failure=bool(telemetry.get("terminal_failure")),
        fallback_telemetry=telemetry,
        request_completed=True,
    )
    row.update(
        {
            "system": "agentic_rag",
            "query": meta["query"],
            "answer": answer,
            "answer_for_metrics": answer_for_metrics,
            "retrieval_source_count": len(retrieved_paper_ids),
            "context_source_count": len(context_paper_ids),
            "actions": actions,
            "reflection_passed": (
                response.reflection_result or {}
            ).get("passed")
            if response.reflection_result is not None
            else None,
            "reflection_result": response.reflection_result,
            "presentation": presentation,
            "sufficiency_result": response.sufficiency_result,
            "removed_citations": list(response.removed_citations or []),
            "synthesis_context_paper_ids": list(response.synthesis_context_paper_ids or []),
            "fallback_telemetry": telemetry,
            "stage_timings": _stage_timings(traces),
        }
    )
    _attach_usage_cost(
        row,
        usage_collector.snapshot(),
        billing_origin=billing_origin,
        pricing_catalog=pricing_catalog,
    )
    return row


def _paired_rows(traditional_rows: list[dict], agentic_rows: list[dict]) -> list[dict]:
    agents = {row["qid"]: row for row in agentic_rows}
    pairs = []
    for trad in traditional_rows:
        agent = agents.get(trad["qid"])
        if not agent:
            continue
        pairs.append(
            {
                "qid": trad["qid"],
                "type": trad.get("type"),
                "difficulty": trad.get("difficulty"),
                "expected_pids": trad.get("expected_pids"),
                "traditional_source_recall": trad.get("source_recall"),
                "agentic_source_recall": agent.get("source_recall"),
                "traditional_citation_precision": trad.get("citation_precision"),
                "agentic_citation_precision": agent.get("citation_precision"),
                "traditional_citation_expected_hit": trad.get("citation_expected_hit"),
                "agentic_citation_expected_hit": agent.get("citation_expected_hit"),
                "traditional_mode_correct": trad.get("mode_correct"),
                "agentic_mode_correct": agent.get("mode_correct"),
                "traditional_latency_s": trad.get("latency_s"),
                "agentic_latency_s": agent.get("latency_s"),
                "traditional_used_chunks": trad.get("used_chunks"),
                "agentic_used_chunks": agent.get("used_chunks"),
                "agentic_step_count": agent.get("step_count"),
                "agentic_retrieval_step_count": agent.get("retrieval_step_count"),
                "traditional_error": trad.get("error"),
                "agentic_error": agent.get("error"),
            }
        )
    return pairs


def _settings_manifest() -> dict:
    from app.core.config import get_settings
    from app.observability.llm_usage import infer_provider

    settings = get_settings()

    def connection_sha256(url: str) -> str:
        parsed = urlparse(url)
        identity = (
            f"{parsed.scheme.lower()}://{(parsed.hostname or '').lower()}"
            f":{parsed.port or ''}{parsed.path.rstrip('/')}"
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    mysql_identity = settings.mysql_url or (
        f"mysql://{settings.mysql_host or 'unset'}:{settings.mysql_port}/"
        f"{settings.mysql_database}"
    )
    return {
        "llm_provider": infer_provider(settings.llm_api_base),
        "llm_model": settings.llm_model,
        "llm_connection_sha256": connection_sha256(settings.llm_api_base),
        "planner_model": settings.planner_model or settings.llm_model,
        "reflection_model": settings.reflection_model or settings.llm_model,
        "embedding_provider": infer_provider(settings.embedding_api_base),
        "embedding_model": settings.embedding_model,
        "embedding_connection_sha256": connection_sha256(settings.embedding_api_base),
        "mysql_connection_sha256": connection_sha256(mysql_identity),
        "qdrant_collection": settings.qdrant_collection,
        "qdrant_connection_sha256": connection_sha256(settings.qdrant_url),
        "retrieval_k": settings.retrieval_k,
        "final_context_k": settings.final_context_k,
        "hybrid_retrieval_enabled": settings.hybrid_retrieval_enabled,
        "hybrid_alpha": settings.hybrid_alpha,
        "hybrid_oversample": settings.hybrid_oversample,
        "hybrid_max_fetch": settings.hybrid_max_fetch,
        "cache_retrieval_enabled": settings.cache_retrieval_enabled,
        "cache_retrieval_ttl_sec": settings.cache_retrieval_ttl_sec,
        "cache_retrieval_max_entries": settings.cache_retrieval_max_entries,
        "cache_embedding_enabled": settings.cache_embedding_enabled,
        "cache_embedding_max_entries": settings.cache_embedding_max_entries,
        "http_retry_max_attempts": settings.http_retry_max_attempts,
        "http_retry_backoff_base_sec": settings.http_retry_backoff_base_sec,
        "llm_max_retries": settings.llm_max_retries,
        "agent_max_plan_steps": settings.agent_max_plan_steps,
        "agent_max_reflections": settings.agent_max_reflections,
        "agent_external_retrieval_enabled": settings.agent_external_retrieval_enabled,
        "agent_checkpoint_enabled": settings.agent_checkpoint_enabled,
        "tavily_web_search_configured": bool(settings.tavily_api_key),
        "arxiv_max_results": settings.arxiv_max_results,
        "llm_request_timeout_s": 120,
    }


def _render_metric_table(rows: list[dict]) -> str:
    lines = [
        "| Metric | Traditional RAG | Agentic RAG | Delta | Delta % |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {metric} | {baseline} | {candidate} | {absolute_delta} | {relative_delta_pct} |".format(
                **{k: "" if v is None else v for k, v in row.items()}
            )
        )
    return "\n".join(lines)


def _render_system_table(manifest: dict, summaries: dict) -> str:
    settings = manifest["settings"]
    traditional = summaries["traditional_rag"]
    agentic = summaries["agentic_rag"]
    rows = [
        ("Pipeline", "local retrieve -> fixed context answer", "LangGraph agent pipeline"),
        ("Retriever", "service retriever", "planner-routed tools"),
        ("Context strategy", manifest["traditional"]["context_strategy"], "evidence processing + citation gate"),
        ("Context k", manifest["traditional"]["context_k"], settings["final_context_k"]),
        ("Retrieval top k", manifest["traditional"]["retrieval_top_k"], settings["retrieval_k"]),
        ("Max reflections", "0", settings["agent_max_reflections"]),
        ("Mean steps", traditional.get("step_count_mean"), agentic.get("step_count_mean")),
        ("Mean retrieval steps", traditional.get("retrieval_step_count_mean"), agentic.get("retrieval_step_count_mean")),
        (
            "External retrieval",
            "disabled",
            "disabled" if manifest["agentic"].get("local_only", True) else "enabled",
        ),
    ]
    lines = [
        "| Parameter | Traditional RAG | Agentic RAG |",
        "|---|---:|---:|",
    ]
    for key, left, right in rows:
        lines.append(f"| {key} | {left} | {right} |")
    return "\n".join(lines)


def render_report(
    *,
    run_id: str,
    manifest: dict,
    summaries: dict,
    comparisons: list[dict],
) -> str:
    question_count = manifest["question_count"]
    dataset_name = Path(manifest["dataset"]).name
    agentic = summaries["agentic_rag"]
    cost_note = (
        f"Known for {agentic.get('cost_known_n', 0)}/{agentic.get('count', 0)} tasks."
        if agentic.get("cost_known_n")
        else "Unknown: provider usage, exact model pricing, or billing origin was not fully verified."
    )
    production_metrics = "\n".join(
        [
            "| Metric | Value | Sample |",
            "|---|---:|---:|",
            f"| task_success_rate | {agentic.get('task_success_rate')} | n={agentic.get('count')} |",
            f"| mode_accuracy | {agentic.get('mode_accuracy')} | n={agentic.get('count')} |",
            f"| citation_support_rate | {agentic.get('citation_support_rate')} | citation cases n={agentic.get('citation_support_n')} |",
            f"| expected_source_hit_rate | {agentic.get('expected_source_hit_rate')} | positives={agentic.get('count_positive')} |",
            f"| terminal_failure_rate | {agentic.get('terminal_failure_rate')} | n={agentic.get('count')} |",
            f"| latency p50 / p90 / p95 / mean (s) | {agentic.get('latency_p50')} / {agentic.get('latency_p90')} / {agentic.get('latency_p95')} / {agentic.get('latency_mean')} | n={agentic.get('latency_n')} |",
            f"| fallback_recovery_rate | {agentic.get('fallback_recovery_rate')} | attempted={agentic.get('fallback_attempted_count')} |",
            f"| cost mean / p50 / p95 (USD, LLM-only) | {agentic.get('cost_per_task_mean_usd')} / {agentic.get('cost_per_task_p50_usd')} / {agentic.get('cost_per_task_p95_usd')} | known n={agentic.get('cost_known_n')} |",
        ]
    )
    return "\n\n".join(
        [
            f"# Traditional RAG vs Agentic RAG Eval: {run_id}",
            (
                f"- Dataset: `{dataset_name}`\n"
                f"- Questions: {question_count}\n"
                f"- Selection: {manifest['selection']}\n"
                f"- Dataset SHA-256: `{manifest['dataset_sha256']}`\n"
                f"- Git: `{manifest['git']['commit']}` (dirty={manifest['git']['dirty']})\n"
                f"- Concurrency: {manifest['execution']['concurrency']}; warmup={manifest['execution']['warmup']}\n"
                "- Retrieval metrics use raw local paper IDs; citation support uses final context paper IDs.\n"
                "- Strict task success uses structured presentation/sufficiency/source/citation-gate/degraded state.\n"
                "- Answer metrics do not use an external judge. Small-n P95 is descriptive, not a stable production SLO."
            ),
            "## Production Metrics\n" + production_metrics,
            "## Cost Attribution\n"
            + f"- Catalog: `{manifest['pricing']['catalog_version']}`\n"
            + f"- Official source: {manifest['pricing']['source']}\n"
            + f"- Billing origin: `{manifest['pricing']['billing_origin']}`\n"
            + f"- Scope: `LLM-only`; embedding and rerank are excluded.\n"
            + f"- Status: {cost_note}",
            "## Key Parameters\n" + _render_system_table(manifest, summaries),
            "## Overall Metrics\n" + _render_metric_table(comparisons),
            "## Traditional Summary\n```json\n"
            + json.dumps(summaries["traditional_rag"], ensure_ascii=False, indent=2)
            + "\n```",
            "## Agentic Summary\n```json\n"
            + json.dumps(summaries["agentic_rag"], ensure_ascii=False, indent=2)
            + "\n```",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare fixed-context traditional RAG with full Agentic RAG."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(PROJECT_ROOT / "eval/datasets/questions_501_test_200.jsonl"),
    )
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "eval/results/agentic"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Select this many questions using original type proportions.",
    )
    parser.add_argument(
        "--per-type",
        type=int,
        default=None,
        help="Select this many questions per type before applying --limit.",
    )
    parser.add_argument("--context-k", type=int, default=5)
    parser.add_argument("--retrieval-top-k", type=int, default=None)
    parser.add_argument(
        "--traditional-context-strategy",
        choices=["raw", "paper_dedup", "mmr_dedup"],
        default="raw",
    )
    parser.add_argument("--mmr-lambda", type=float, default=0.65)
    parser.add_argument(
        "--allow-external",
        action="store_true",
        help="Allow arXiv/web tools in the Agentic branch (disabled by default for comparable local RAG evaluation).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted run, keeping successful per-question rows and rerunning missing/error rows.",
    )
    parser.add_argument(
        "--billing-origin",
        choices=["unknown", "minimax_paygo"],
        default="unknown",
        help="Verified billing origin for official-price attribution; compatible endpoints are not inferred.",
    )
    parser.add_argument(
        "--pricing-catalog",
        type=str,
        default=str(DEFAULT_CATALOG_PATH),
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser().resolve()
    pricing_catalog_path = Path(args.pricing_catalog).expanduser().resolve()
    pricing_catalog = load_pricing_catalog(pricing_catalog_path)
    run_id = args.run_id or f"agentic-rag-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path(args.output_dir) / run_id
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Run directory already has outputs: {run_dir}. Use --resume to continue it.")
    run_dir.mkdir(parents=True, exist_ok=True)

    all_questions = load_questions(dataset_path)
    if args.sample_size is not None and args.per_type is not None:
        raise ValueError("Use either --sample-size or --per-type, not both.")
    if args.sample_size is not None:
        questions = select_proportional_questions(all_questions, sample_size=args.sample_size)
    else:
        questions = select_stratified_questions(all_questions, per_type=args.per_type)
    if args.limit is not None:
        questions = questions[: args.limit]
    if not questions:
        raise ValueError("No questions selected for evaluation.")

    selection = {
        "source_count": len(all_questions),
        "sample_size": args.sample_size,
        "per_type": args.per_type,
        "limit": args.limit,
    }
    traditional_config = {
        "retriever": "service",
        "context_k": args.context_k,
        "retrieval_top_k": args.retrieval_top_k,
        "context_strategy": args.traditional_context_strategy,
        "mmr_lambda": args.mmr_lambda,
    }
    agentic_config = {
        "entrypoint": "app.agent.graph.run_agent_eval_sync",
        "retrieval_actions": sorted(RETRIEVAL_ACTIONS),
        "local_only": not args.allow_external,
    }
    settings_manifest = _settings_manifest()
    git_snapshot = _git_snapshot()
    execution_config = {
        "command": shlex.join([sys.executable, *sys.argv]),
        "concurrency": 1,
        "warmup": False,
        "request_timeout_s": 120,
        "external_api_allowed": bool(args.allow_external),
    }
    pricing_config = {
        "catalog_path": str(pricing_catalog_path),
        "catalog_sha256": _sha256_file(pricing_catalog_path),
        "catalog_version": pricing_catalog.get("catalog_version"),
        "catalog_provider": pricing_catalog.get("provider"),
        "catalog_billing_origin": pricing_catalog.get("billing_origin"),
        "currency": pricing_catalog.get("currency"),
        "source": (pricing_catalog.get("source") or {}).get("url"),
        "billing_origin": args.billing_origin,
        "cost_scope": "llm_only",
    }
    resume_contract = _build_resume_contract(
        dataset_path=dataset_path,
        pricing_catalog_path=pricing_catalog_path,
        questions=questions,
        selection=selection,
        traditional=traditional_config,
        agentic=agentic_config,
        settings=settings_manifest,
        execution=execution_config,
        pricing=pricing_config,
        git_commit=str(git_snapshot["commit"]),
    )

    manifest_path = run_dir / "manifest.json"
    if args.resume:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Cannot resume without manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_resume_contract(manifest, resume_contract)
    else:
        manifest = {
            "schema_version": "2.1",
            "run_id": run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "dataset": str(dataset_path),
            "dataset_sha256": _sha256_file(dataset_path),
            "question_count": len(questions),
            "selected_qids": _selected_qids(questions),
            "selection": selection,
            "traditional": traditional_config,
            "agentic": agentic_config,
            "settings": settings_manifest,
            "git": git_snapshot,
            "execution": execution_config,
            "pricing": pricing_config,
            "resume_contract": resume_contract,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    t0 = time.perf_counter()

    def persist_traditional(rows: list[dict]) -> None:
        _write_jsonl(run_dir / "traditional_per_question.jsonl", rows)
        _write_csv(run_dir / "traditional_per_question.csv", rows)

    traditional_rows = run_cases_with_checkpoint(
        questions,
        existing_rows=[row for row in _load_jsonl(run_dir / "traditional_per_question.jsonl") if not row.get("error")],
        run_case=lambda item, idx, total: run_traditional_case(
            item,
            idx=idx,
            total=total,
            context_k=args.context_k,
            retrieval_top_k=args.retrieval_top_k,
            context_strategy=args.traditional_context_strategy,
            mmr_lambda=args.mmr_lambda,
            billing_origin=str(manifest.get("pricing", {}).get("billing_origin", "unknown")),
            pricing_catalog=pricing_catalog,
        ),
        persist=persist_traditional,
    )

    from app.db.mysql import SessionLocal

    db = SessionLocal()
    try:
        def persist_agentic(rows: list[dict]) -> None:
            _write_jsonl(run_dir / "agentic_per_question.jsonl", rows)
            _write_csv(run_dir / "agentic_per_question.csv", rows)

        agentic_rows = run_cases_with_checkpoint(
            questions,
            existing_rows=[row for row in _load_jsonl(run_dir / "agentic_per_question.jsonl") if not row.get("error")],
            run_case=lambda item, idx, total: run_agentic_case(
                item,
                idx=idx,
                total=total,
                run_id=run_id,
                db=db,
                local_only=bool(manifest["agentic"].get("local_only", True)),
                billing_origin=str(manifest.get("pricing", {}).get("billing_origin", "unknown")),
                pricing_catalog=pricing_catalog,
            ),
            persist=persist_agentic,
        )
    finally:
        db.close()

    elapsed_s = time.perf_counter() - t0
    summaries = {
        "traditional_rag": summarize_answer_cases(traditional_rows),
        "agentic_rag": summarize_answer_cases(agentic_rows),
    }
    summaries["elapsed_total_s"] = round(elapsed_s, 4)
    comparisons = compare_system_summaries(
        baseline=summaries["traditional_rag"],
        candidate=summaries["agentic_rag"],
    )
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    paired = _paired_rows(traditional_rows, agentic_rows)

    _write_jsonl(run_dir / "traditional_per_question.jsonl", traditional_rows)
    _write_jsonl(run_dir / "agentic_per_question.jsonl", agentic_rows)
    _write_jsonl(run_dir / "per_question.jsonl", agentic_rows)
    _write_jsonl(run_dir / "paired_per_question.jsonl", paired)
    _write_csv(run_dir / "traditional_per_question.csv", traditional_rows)
    _write_csv(run_dir / "agentic_per_question.csv", agentic_rows)
    _write_csv(run_dir / "paired_per_question.csv", paired)
    summary_payload = {
        "schema_version": "2.0",
        "task_metrics": summaries["agentic_rag"],
        "summaries": summaries,
        "comparisons": comparisons,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(
            summary_payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "report.md").write_text(
        render_report(
            run_id=run_id,
            manifest=manifest,
            summaries=summaries,
            comparisons=comparisons,
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
    print(f"Saved traditional-vs-agentic eval to {run_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
