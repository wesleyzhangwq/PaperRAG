"""Compare fixed-context traditional RAG with the full Agentic RAG graph.

The traditional branch is intentionally plain: one local retrieval call, a
fixed top-k context window, and one answer-generation call. The agentic branch
uses the production LangGraph pipeline through ``run_agent_sync``.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from contextlib import ExitStack
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))

from eval.agentic_compare_metrics import (  # noqa: E402
    answer_case_metrics,
    compare_system_summaries,
    select_proportional_questions,
    select_stratified_questions,
    strip_thinking_for_metrics,
    summarize_answer_cases,
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
    rows_by_qid = {
        str(row.get("qid") or ""): dict(row)
        for row in existing_rows
        if str(row.get("qid") or "")
    }
    total = len(questions)

    def ordered_rows() -> list[dict]:
        return [rows_by_qid[str(item["qid"])] for item in questions if str(item["qid"]) in rows_by_qid]

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
        error=f"{type(exc).__name__}: {redact_sensitive_text(str(exc))}",
    )
    row["system"] = system
    row["query"] = meta["query"]
    row["answer"] = ""
    row["mode_correct"] = False
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
) -> dict:
    from app.services.retriever import retrieve

    meta = _question_meta(item, idx)
    print(f"[traditional {idx}/{total}] {meta['qid']}", file=sys.stderr)
    t0 = time.perf_counter()
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
        return _error_row(
            system="traditional_rag",
            meta=meta,
            latency_s=time.perf_counter() - t0,
            exc=exc,
        )

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
    return row


def run_agentic_case(
    item: dict,
    *,
    idx: int,
    total: int,
    run_id: str,
    db: Any,
    local_only: bool = True,
) -> dict:
    from app.agent.graph import run_agent_eval_sync

    meta = _question_meta(item, idx)
    print(f"[agentic {idx}/{total}] {meta['qid']}", file=sys.stderr)
    t0 = time.perf_counter()
    try:
        with ExitStack() as stack:
            if local_only:
                stack.enter_context(
                    patch(
                        "app.agent.nodes.executor.retrieve_arxiv_tool",
                        SimpleNamespace(invoke=lambda _params: ""),
                    )
                )
                stack.enter_context(
                    patch(
                        "app.agent.nodes.executor.search_web_tool",
                        SimpleNamespace(
                            invoke=lambda _params: "Web search is not configured."
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
        return _error_row(
            system="agentic_rag",
            meta=meta,
            latency_s=time.perf_counter() - t0,
            exc=exc,
        )

    traces = response.step_traces or []
    actions = _trace_actions(traces)
    answer = redact_sensitive_text(response.answer or "")
    answer_for_metrics = strip_thinking_for_metrics(answer)
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
        }
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

    settings = get_settings()
    return {
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "retrieval_k": settings.retrieval_k,
        "final_context_k": settings.final_context_k,
        "hybrid_retrieval_enabled": settings.hybrid_retrieval_enabled,
        "hybrid_alpha": settings.hybrid_alpha,
        "hybrid_oversample": settings.hybrid_oversample,
        "hybrid_max_fetch": settings.hybrid_max_fetch,
        "cache_retrieval_enabled": settings.cache_retrieval_enabled,
        "agent_max_plan_steps": settings.agent_max_plan_steps,
        "agent_max_reflections": settings.agent_max_reflections,
        "agent_external_retrieval_enabled": settings.agent_external_retrieval_enabled,
        "agent_checkpoint_enabled": settings.agent_checkpoint_enabled,
        "tavily_web_search_configured": bool(settings.tavily_api_key),
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
    return "\n\n".join(
        [
            f"# Traditional RAG vs Agentic RAG Eval: {run_id}",
            (
                f"- Dataset: `{dataset_name}`\n"
                f"- Questions: {question_count}\n"
                f"- Selection: {manifest['selection']}\n"
                "- Retrieval metrics use raw local paper IDs; citation support uses final context paper IDs.\n"
                "- Answer metrics do not use an external judge."
            ),
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
        default=str(PROJECT_ROOT / "eval/datasets/questions_v3_200.jsonl"),
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
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
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

    manifest_path = run_dir / "manifest.json"
    if args.resume:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Cannot resume without manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataset") != str(dataset_path) or manifest.get("question_count") != len(questions):
            raise ValueError("Resume configuration does not match the existing manifest.")
    else:
        manifest = {
            "run_id": run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "dataset": str(dataset_path),
            "question_count": len(questions),
            "selection": {
                "source_count": len(all_questions),
                "sample_size": args.sample_size,
                "per_type": args.per_type,
                "limit": args.limit,
            },
            "traditional": {
                "retriever": "service",
                "context_k": args.context_k,
                "retrieval_top_k": args.retrieval_top_k,
                "context_strategy": args.traditional_context_strategy,
                "mmr_lambda": args.mmr_lambda,
            },
            "agentic": {
                "entrypoint": "app.agent.graph.run_agent_eval_sync",
                "retrieval_actions": sorted(RETRIEVAL_ACTIONS),
                "local_only": not args.allow_external,
            },
            "settings": _settings_manifest(),
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
    _write_jsonl(run_dir / "paired_per_question.jsonl", paired)
    _write_csv(run_dir / "traditional_per_question.csv", traditional_rows)
    _write_csv(run_dir / "agentic_per_question.csv", agentic_rows)
    _write_csv(run_dir / "paired_per_question.csv", paired)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "summaries": summaries,
                "comparisons": comparisons,
            },
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

    print(json.dumps({"summaries": summaries, "comparisons": comparisons}, ensure_ascii=False, indent=2))
    print(f"Saved traditional-vs-agentic eval to {run_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
