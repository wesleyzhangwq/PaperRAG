"""Deterministic, no-network failure-injection benchmark for the production graph."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shlex
import subprocess
import sys
import time
from collections import Counter
from contextlib import ExitStack, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))

from app.observability.llm_usage import collect_llm_usage  # noqa: E402
from eval.agentic_compare_metrics import (  # noqa: E402
    answer_case_metrics,
    summarize_answer_cases,
)
from eval.costing import attribute_llm_costs, load_pricing_catalog  # noqa: E402


PAPER_ID = "1706.03762"
VALID_DOCS = [
    (
        Document(
            page_content="The Transformer uses self-attention for sequence modeling.",
            metadata={
                "paper_id": PAPER_ID,
                "title": "Attention Is All You Need",
                "score": 0.99,
            },
        ),
        0.99,
    )
]
ANSWER = f"Transformer 使用自注意力完成序列建模 [arxiv:{PAPER_ID}]。"
INSUFFICIENT = "现有论文证据不足，无法完成回答。"
INTENT_JSON = '{"type":"simple","entities":["Transformer"],"complexity":"low"}'
PLAN_LOCAL = '[{"action":"retrieve_local","params":{"query":"Transformer","top_k":3},"reason":"fixture"}]'
PLAN_EXTERNAL = (
    '[{"action":"retrieve_local","params":{"query":"Transformer","top_k":3},"reason":"fixture"},'
    '{"action":"retrieve_arxiv","params":{"query":"latest Transformer","max_results":2},"reason":"fixture"},'
    '{"action":"search_web","params":{"query":"latest Transformer","max_results":2},"reason":"fixture"}]'
)
PLAN_SUPPLEMENT = '[{"action":"retrieve_local","params":{"query":"self attention","top_k":3},"reason":"supplement"}]'
SUFFICIENT = '{"sufficient":true,"reason":"enough","missing_aspects":[]}'
INSUFFICIENT_JSON = '{"sufficient":false,"reason":"missing","missing_aspects":["evidence"]}'
GROUNDED_PASS = (
    '{"passed":true,"citation_ok":true,"completeness_ok":true,'
    '"logic_ok":true,"issues":[],"fix_strategy":null}'
)
GROUNDED_RETRIEVE = (
    '{"passed":false,"citation_ok":true,"completeness_ok":false,'
    '"logic_ok":true,"issues":["need more evidence"],"fix_strategy":"re_retrieve"}'
)
GROUNDED_GENERATE = (
    '{"passed":false,"citation_ok":true,"completeness_ok":false,'
    '"logic_ok":true,"issues":["answer incomplete"],"fix_strategy":"re_generate"}'
)


class ScriptedInvokeLlm:
    def __init__(self, outputs: list[str | BaseException]) -> None:
        self.outputs = list(outputs)

    def invoke(self, _prompt: Any) -> AIMessage:
        if not self.outputs:
            raise AssertionError("deterministic invoke script exhausted")
        value = self.outputs.pop(0)
        if isinstance(value, BaseException):
            raise value
        return AIMessage(
            content=value,
            usage_metadata={
                "input_tokens": 20,
                "output_tokens": 8,
                "total_tokens": 28,
                "input_token_details": {"cache_read": 2},
            },
        )


class ScriptedStreamLlm:
    def __init__(self, outputs: list[str | BaseException]) -> None:
        self.outputs = list(outputs)

    def stream(self, _prompt: Any):
        if not self.outputs:
            raise AssertionError("deterministic stream script exhausted")
        value = self.outputs.pop(0)
        if isinstance(value, BaseException):
            raise value
        yield AIMessageChunk(content=value)
        yield AIMessageChunk(
            content="",
            usage_metadata={
                "input_tokens": 40,
                "output_tokens": 12,
                "total_tokens": 52,
                "input_token_details": {"cache_read": 4},
            },
        )


class SequenceCallable:
    def __init__(self, outputs: list[Any]) -> None:
        self.outputs = list(outputs)

    def __call__(self, *_args: Any, **_kwargs: Any) -> Any:
        if not self.outputs:
            raise AssertionError("deterministic retrieval script exhausted")
        value = self.outputs.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


class FakeQuery:
    def __init__(self, paper: Any, *, fail: bool = False) -> None:
        self.paper = paper
        self.fail = fail

    def filter(self, *_args: Any, **_kwargs: Any) -> "FakeQuery":
        return self

    def one_or_none(self) -> Any:
        if self.fail:
            raise RuntimeError("deterministic database failure")
        return self.paper


class FakeDb:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.paper = SimpleNamespace(
            paper_id=PAPER_ID,
            title="Attention Is All You Need",
            authors=["Vaswani et al."],
            year=2017,
            primary_category="cs.CL",
            doi=None,
            abstract="Transformer architecture based on attention.",
        )

    def query(self, _model: Any) -> FakeQuery:
        return FakeQuery(self.paper, fail=self.fail)


SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "local_retrieval_empty",
        "planner": [PLAN_LOCAL, PLAN_SUPPLEMENT],
        "retrieve": [[], VALID_DOCS],
        "sufficiency": [SUFFICIENT],
        "synthesis": [ANSWER],
        "groundedness": [GROUNDED_PASS],
        "expected_outcome": "recovered",
        "required_class": "local_retrieval_empty",
    },
    {
        "id": "local_retrieval_exception",
        "planner": [PLAN_LOCAL, PLAN_SUPPLEMENT],
        "retrieve": [RuntimeError("fixture local failure"), VALID_DOCS],
        "sufficiency": [SUFFICIENT],
        "synthesis": [ANSWER],
        "groundedness": [GROUNDED_PASS],
        "expected_outcome": "recovered",
        "required_class": "local_retrieval_exception",
    },
    {
        "id": "arxiv_web_unavailable",
        "query": "latest Transformer evidence",
        "planner": [PLAN_EXTERNAL],
        "retrieve": [VALID_DOCS],
        "sufficiency": [SUFFICIENT],
        "synthesis": [ANSWER],
        "groundedness": [GROUNDED_PASS],
        "arxiv_error": RuntimeError("fixture arxiv failure"),
        "web_error": RuntimeError("fixture web failure"),
        "expected_outcome": "recovered",
        "required_class": "arxiv_service_unavailable",
        "required_class_2": "web_service_unavailable",
    },
    {
        "id": "planner_output_unparseable",
        "planner": ["not valid json"],
        "retrieve": [VALID_DOCS],
        "sufficiency": [SUFFICIENT],
        "synthesis": [ANSWER],
        "groundedness": [GROUNDED_PASS],
        "expected_outcome": "recovered",
        "required_class": "planner_output_unparseable",
    },
    {
        "id": "sufficiency_output_unparseable",
        "planner": [PLAN_LOCAL],
        "retrieve": [VALID_DOCS],
        "sufficiency": ["not valid json"],
        "synthesis": [ANSWER],
        "groundedness": [GROUNDED_PASS],
        "expected_outcome": "safe_degraded",
        "required_class": "sufficiency_output_unparseable",
    },
    {
        "id": "llm_timeout",
        "planner": [TimeoutError("fixture timeout")],
        "retrieve": [VALID_DOCS],
        "sufficiency": [SUFFICIENT],
        "synthesis": [ANSWER],
        "groundedness": [GROUNDED_PASS],
        "expected_outcome": "recovered",
        "required_class": "llm_timeout",
    },
    {
        "id": "groundedness_re_retrieve",
        "planner": [PLAN_LOCAL, PLAN_SUPPLEMENT],
        "retrieve": [VALID_DOCS, VALID_DOCS],
        "sufficiency": [SUFFICIENT, SUFFICIENT],
        "synthesis": [ANSWER, ANSWER],
        "groundedness": [GROUNDED_RETRIEVE, GROUNDED_PASS],
        "expected_outcome": "recovered",
        "required_re_retrieve": 1,
    },
    {
        "id": "groundedness_re_generate",
        "planner": [PLAN_LOCAL],
        "retrieve": [VALID_DOCS],
        "sufficiency": [SUFFICIENT],
        "synthesis": ["初稿不完整。", ANSWER],
        "groundedness": [GROUNDED_GENERATE, GROUNDED_PASS],
        "expected_outcome": "recovered",
        "required_re_generate": 1,
    },
    {
        "id": "supplement_budget_exhausted",
        "planner": [PLAN_LOCAL, PLAN_SUPPLEMENT],
        "retrieve": [[], []],
        "sufficiency": [INSUFFICIENT_JSON, INSUFFICIENT_JSON],
        "synthesis": [INSUFFICIENT],
        "groundedness": [GROUNDED_PASS],
        "expected_outcome": "safe_degraded",
        "required_class": "supplement_retrieval_budget_exhausted",
    },
    {
        "id": "terminal_database_failure",
        "planner": [PLAN_LOCAL],
        "retrieve": [VALID_DOCS],
        "sufficiency": [SUFFICIENT],
        "synthesis": [ANSWER],
        "groundedness": [GROUNDED_PASS],
        "db_fail": True,
        "expected_outcome": "terminal_failure",
    },
]


def _git_snapshot() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    status = [line for line in run("status", "--porcelain").splitlines() if line]
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(status), "dirty_entry_count": len(status)}


def _clean_git_snapshot() -> dict[str, Any]:
    snapshot = _git_snapshot()
    if snapshot["dirty"]:
        raise RuntimeError(
            "Refusing to generate production evidence from a dirty worktree; "
            "commit the runner, configuration, and dataset first."
        )
    return snapshot


def _scenario_hash() -> str:
    serializable = [
        {
            key: (
                [type(value).__name__ if isinstance(value, BaseException) else str(value) for value in raw]
                if isinstance(raw, list)
                else type(raw).__name__ if isinstance(raw, BaseException)
                else raw
            )
            for key, raw in scenario.items()
        }
        for scenario in SCENARIOS
    ]
    payload = json.dumps(serializable, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stage_timings(traces: list[dict] | None) -> list[dict[str, Any]]:
    return [
        {
            "node": str(trace.get("node") or ""),
            "action": str(trace.get("action") or trace.get("node") or "unknown"),
            "duration_ms": round(float(trace.get("duration_ms") or 0), 3),
        }
        for trace in traces or []
    ]


def _outcome(row: dict[str, Any]) -> str:
    if row.get("terminal_failure"):
        return "terminal_failure"
    if row.get("degraded_answer"):
        return "safe_degraded"
    if row.get("fallback_recovered"):
        return "recovered"
    return "normal"


def _scenario_passed(row: dict[str, Any], scenario: dict[str, Any]) -> bool:
    telemetry = row.get("fallback_telemetry") or {}
    classes = set(telemetry.get("failure_classes") or [])
    checks = [_outcome(row) == scenario["expected_outcome"]]
    if scenario.get("required_class"):
        checks.append(scenario["required_class"] in classes)
    if scenario.get("required_class_2"):
        checks.append(scenario["required_class_2"] in classes)
    if scenario.get("required_re_retrieve") is not None:
        checks.append(int(row.get("re_retrieve_count") or 0) >= int(scenario["required_re_retrieve"]))
    if scenario.get("required_re_generate") is not None:
        checks.append(int(row.get("re_generate_count") or 0) >= int(scenario["required_re_generate"]))
    return all(checks)


def run_scenario(scenario: dict[str, Any], *, catalog: dict[str, Any]) -> dict[str, Any]:
    from app.agent.graph import run_agent_eval_sync

    intent_llm = ScriptedInvokeLlm([INTENT_JSON])
    planner_llm = ScriptedInvokeLlm(list(scenario["planner"]))
    sufficiency_llm = ScriptedInvokeLlm(list(scenario["sufficiency"]))
    synthesis_llm = ScriptedStreamLlm(list(scenario["synthesis"]))
    groundedness_llm = ScriptedInvokeLlm(list(scenario["groundedness"]))
    retrieve = SequenceCallable(list(scenario["retrieve"]))
    db = FakeDb(fail=bool(scenario.get("db_fail")))
    arxiv_tool = SimpleNamespace(
        invoke=(
            lambda _params: (_ for _ in ()).throw(scenario["arxiv_error"])
            if scenario.get("arxiv_error")
            else "Evaluation external arXiv intentionally disabled."
        )
    )
    web_tool = SimpleNamespace(
        invoke=(
            lambda _params: (_ for _ in ()).throw(scenario["web_error"])
            if scenario.get("web_error")
            else "Evaluation external web intentionally disabled."
        )
    )
    query = str(scenario.get("query") or "Explain Transformer evidence")
    started = time.perf_counter()

    with collect_llm_usage() as collector:
        try:
            with ExitStack() as stack:
                stack.enter_context(patch("app.agent.graph.open_sync_checkpointer", return_value=nullcontext(None)))
                stack.enter_context(patch("app.agent.nodes.intent._get_llm", return_value=intent_llm))
                stack.enter_context(patch("app.agent.nodes.planner._get_llm", return_value=planner_llm))
                stack.enter_context(patch("app.tools.evaluate_docs._get_llm", return_value=sufficiency_llm))
                stack.enter_context(patch("app.agent.nodes.synthesis._get_llm", return_value=synthesis_llm))
                stack.enter_context(patch("app.agent.nodes.groundedness._get_llm", return_value=groundedness_llm))
                stack.enter_context(patch("app.agent.nodes.executor.retrieve", side_effect=retrieve))
                stack.enter_context(patch("app.agent.nodes.executor.retrieve_arxiv_tool", arxiv_tool))
                stack.enter_context(patch("app.agent.nodes.executor.search_web_tool", web_tool))
                if scenario["id"] == "arxiv_web_unavailable":
                    stack.enter_context(
                        patch(
                            "app.agent.nodes.route.get_settings",
                            return_value=SimpleNamespace(tavily_api_key="fixture", arxiv_max_results=2),
                        )
                    )
                response, retrieved_ids, context_ids = run_agent_eval_sync(
                    db,
                    query,
                    session_id=f"failure-injection-{scenario['id']}",
                )
            terminal_error = None
        except Exception as exc:
            response = None
            retrieved_ids = []
            context_ids = []
            terminal_error = type(exc).__name__
        latency_s = time.perf_counter() - started

    if response is None:
        telemetry = {
            "fallback_attempted": False,
            "fallback_recovered": False,
            "re_retrieve_count": 0,
            "re_generate_count": 0,
            "degraded_answer": False,
            "terminal_failure": True,
            "failure_class": "terminal_exception",
            "failure_classes": ["terminal_exception"],
            "events": [{"stage": "request", "failure_class": "terminal_exception", "outcome": "terminal_failure"}],
        }
        answer = ""
        presentation = None
        final_sources: list[str] = []
        removed: list[str] = []
        degraded = False
        traces: list[dict] = []
        sufficiency_result = None
    else:
        telemetry = dict(response.fallback_telemetry or {})
        answer = response.answer
        presentation = response.presentation
        final_sources = [source.paper_id for source in response.sources]
        removed = list(response.removed_citations or [])
        degraded = bool(response.degraded or telemetry.get("degraded_answer"))
        traces = list(response.step_traces or [])
        sufficiency_result = response.sufficiency_result

    row = answer_case_metrics(
        qid=scenario["id"],
        qtype="failure_injection",
        difficulty="deterministic",
        expected_paper_ids=[PAPER_ID],
        expected_mode="answer",
        answer=answer,
        source_pids=retrieved_ids,
        context_pids=context_ids,
        final_source_pids=final_sources,
        latency_s=latency_s,
        step_count=len(traces),
        retrieval_step_count=sum(
            1 for trace in traces if trace.get("action") in {"retrieve_local", "retrieve_arxiv", "search_web"}
        ),
        presentation=presentation,
        sufficiency_result=sufficiency_result,
        removed_citation_pids=removed,
        degraded_answer=degraded,
        terminal_failure=bool(telemetry.get("terminal_failure")),
        fallback_telemetry=telemetry,
        request_completed=response is not None,
        error=terminal_error,
    )
    cost = attribute_llm_costs(
        collector.snapshot(),
        billing_origin="deterministic_fixture",
        catalog=catalog,
    )
    row.update(
        {
            "system": "production_graph_deterministic_fixture",
            "query": query,
            "answer": answer,
            "expected_outcome": scenario["expected_outcome"],
            "actual_outcome": _outcome(row),
            "fallback_telemetry": telemetry,
            "stage_timings": _stage_timings(traces),
            "llm_usage": cost.pop("calls"),
            "llm_usage_totals": cost,
            "cost_status": cost["cost_status"],
            "cost_usd": cost["cost_usd"],
            "pricing_catalog_version": cost["pricing_catalog_version"],
            "cost_scope": cost["cost_scope"],
        }
    )
    row["scenario_passed"] = _scenario_passed(row, scenario)
    return row


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _render_report(manifest: dict[str, Any], summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    matrix = [
        "| Scenario | Expected | Actual | Passed | Failure classes |",
        "|---|---|---|---:|---|",
    ]
    for row in rows:
        classes = ", ".join((row.get("fallback_telemetry") or {}).get("failure_classes") or [])
        matrix.append(
            f"| {row['qid']} | {row['expected_outcome']} | {row['actual_outcome']} | "
            f"{str(bool(row['scenario_passed'])).lower()} | {classes} |"
        )
    metrics = summary["task_metrics"]
    return "\n\n".join(
        [
            f"# PaperRAG Failure-Injection Benchmark: {manifest['run_id']}",
            (
                f"- Production entrypoint: `{manifest['entrypoint']}`\n"
                f"- Deterministic scenarios: {manifest['sample_count']}\n"
                f"- Concurrency: 1; warmup: false; external API calls: 0\n"
                f"- Git: `{manifest['git']['commit']}` (dirty={manifest['git']['dirty']})\n"
                f"- Scenario dataset SHA-256: `{manifest['dataset_sha256']}`\n"
                "- Provider responses and retrieval are deterministic fixtures; no result is presented as a real-provider quality or cost measurement."
            ),
            "## Outcome\n"
            + f"- scenario_pass_rate: {summary['scenario_pass_rate']} ({summary['scenario_passed_count']}/{summary['scenario_count']})\n"
            + f"- safe_degraded_count: {summary['outcome_counts'].get('safe_degraded', 0)}\n"
            + f"- terminal_failure_count: {summary['outcome_counts'].get('terminal_failure', 0)}\n\n"
            + "| Metric | Value | Sample |\n"
            + "|---|---:|---:|\n"
            + f"| task_success_rate | {metrics.get('task_success_rate')} | n={metrics.get('count')} |\n"
            + f"| mode_accuracy | {metrics.get('mode_accuracy')} | n={metrics.get('count')} |\n"
            + f"| citation_support_rate | {metrics.get('citation_support_rate')} | citation cases n={metrics.get('citation_support_n')} |\n"
            + f"| expected_source_hit_rate | {metrics.get('expected_source_hit_rate')} | positives={metrics.get('count_positive')} |\n"
            + f"| terminal_failure_rate | {metrics.get('terminal_failure_rate')} | n={metrics.get('count')} |\n"
            + f"| fallback_recovery_rate | {metrics.get('fallback_recovery_rate')} | attempted={metrics.get('fallback_attempted_count')} |\n"
            + f"| latency p50 / p90 / p95 / mean (s) | {metrics.get('latency_p50')} / {metrics.get('latency_p90')} / {metrics.get('latency_p95')} / {metrics.get('latency_mean')} | n={metrics.get('latency_n')} |\n"
            + f"| cost mean / p50 / p95 (USD, LLM-only) | {metrics.get('cost_per_task_mean_usd')} / {metrics.get('cost_per_task_p50_usd')} / {metrics.get('cost_per_task_p95_usd')} | known n={metrics.get('cost_known_n')} |\n\n"
            + "Cost is unknown by design because usage is synthetic and the billing origin is unverified.",
            "## Scenario Matrix\n" + "\n".join(matrix),
            "## Metric Definitions\n"
            + "`fallback_recovery_rate = strict task-success fallbacks / fallback_attempts`. "
            + "Safe degraded responses and expected terminal injections do not count as recovered. "
            + "P95 is descriptive for this small deterministic scenario set.",
        ]
    ) + "\n"


def main() -> int:
    from app.agent.nodes.sufficiency import MAX_SUFFICIENCY_ROUNDS
    from app.core.config import get_settings

    parser = argparse.ArgumentParser(description="Run deterministic production-graph failure injection.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "eval" / "results" / "production"),
    )
    args = parser.parse_args()
    git_snapshot = _clean_git_snapshot()
    run_id = args.run_id or f"failure-injection-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path(args.output_dir) / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory already has outputs: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    catalog = load_pricing_catalog()
    settings = get_settings()
    started_at = datetime.now(timezone.utc).isoformat()
    rows = [run_scenario(scenario, catalog=catalog) for scenario in SCENARIOS]
    task_metrics = summarize_answer_cases(rows)
    passed = sum(1 for row in rows if row["scenario_passed"])
    summary = {
        "schema_version": "1.0",
        "scenario_count": len(rows),
        "scenario_passed_count": passed,
        "scenario_pass_rate": round(passed / len(rows), 4) if rows else None,
        "outcome_counts": dict(Counter(row["actual_outcome"] for row in rows)),
        "failure_class_counts": dict(
            Counter(
                failure_class
                for row in rows
                for failure_class in (row.get("fallback_telemetry") or {}).get("failure_classes") or []
            )
        ),
        "task_metrics": task_metrics,
    }
    manifest = {
        "schema_version": "1.0",
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "entrypoint": "app.agent.graph.run_agent_eval_sync",
        "git": git_snapshot,
        "dataset": "embedded deterministic failure scenarios",
        "dataset_sha256": _scenario_hash(),
        "sample_count": len(rows),
        "model": "deterministic provider fixture",
        "provider": "deterministic fixture",
        "non_sensitive_config": {
            "concurrency": 1,
            "warmup": False,
            "timeout_s": 120,
            "external_api_calls": 0,
            "agent_max_reflections": settings.agent_max_reflections,
            "sufficiency_round_budget": MAX_SUFFICIENCY_ROUNDS,
        },
        "pricing_catalog_version": catalog.get("catalog_version"),
        "pricing_source": (catalog.get("source") or {}).get("url"),
        "billing_origin": "deterministic_fixture_unpriced",
        "cost_scope": "llm_only",
        "command": shlex.join([sys.executable, *sys.argv]),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }
    _write_jsonl(run_dir / "per_question.jsonl", rows)
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "report.md").write_text(_render_report(manifest, summary, rows), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved deterministic failure benchmark to {run_dir}", file=sys.stderr)
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
