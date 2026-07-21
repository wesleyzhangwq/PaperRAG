"""Write a reproducible zero-request report when real-provider billing is unsafe.

This command deliberately performs no model, embedding, rerank, database, or
retrieval request.  It exists so a blocked production benchmark has the same
four immutable artifacts as a completed benchmark instead of being silently
omitted or replaced by guessed measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.observability.llm_usage import infer_provider  # noqa: E402
from eval.costing import DEFAULT_CATALOG_PATH, load_pricing_catalog  # noqa: E402


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status),
        "dirty_entry_count": len(status),
    }


def _clean_git_snapshot() -> dict[str, Any]:
    snapshot = _git_snapshot()
    if snapshot["dirty"]:
        raise RuntimeError(
            "Refusing to generate production evidence from a dirty worktree; "
            "commit the runner, configuration, and dataset first."
        )
    return snapshot


def _empty_task_metrics() -> dict[str, Any]:
    empty_latency = {"n": 0, "p50": None, "p90": None, "p95": None, "mean": None}
    return {
        "count": 0,
        "count_positive": 0,
        "count_negative": 0,
        "task_success_rate": None,
        "mode_accuracy": None,
        "citation_support_rate": None,
        "citation_support_n": 0,
        "expected_source_hit_rate": None,
        "terminal_failure_rate": None,
        "latency_p50": None,
        "latency_p90": None,
        "latency_p95": None,
        "latency_mean": None,
        "latency_n": 0,
        "latency_by_path": {
            "normal": dict(empty_latency),
            "fallback": dict(empty_latency),
            "terminal": dict(empty_latency),
        },
        "stage_latency": {},
        "fallback_attempted_count": 0,
        "fallback_recovered_count": 0,
        "fallback_recovery_rate": None,
        "cost_known_n": 0,
        "cost_unknown_n": 0,
        "cost_per_task_mean_usd": None,
        "cost_per_task_p50_usd": None,
        "cost_per_task_p95_usd": None,
        "successful_task_cost_mean_usd": None,
        "fallback_task_cost_mean_usd": None,
        "by_type": {},
        "by_difficulty": {},
    }


def _render_report(manifest: dict[str, Any], summary: dict[str, Any]) -> str:
    models = ", ".join(manifest["models"])
    unmapped = ", ".join(
        manifest["pricing"]["models_without_safe_billable_mapping"]
    )
    return "\n\n".join(
        [
            f"# PaperRAG Real-Provider Benchmark: {manifest['run_id']}",
            (
                "- Status: **blocked; no external request was sent**\n"
                f"- Sample size: 0; external requests: {manifest['execution']['external_requests']}\n"
                f"- Configured provider: `{manifest['provider']}`; models: `{models}`\n"
                f"- Git: `{manifest['git']['commit']}` (dirty={manifest['git']['dirty']})\n"
                f"- Dataset SHA-256: `{manifest['dataset_sha256']}`"
            ),
            "## Blocking Evidence\n"
            + "- The configured endpoint's billing origin is unverified; a compatible host does not prove official pay-as-you-go billing.\n"
            + "- MiniMax publishes context-tiered M3 input, output, and cache-read rates, but the configured endpoint and returned usage dimensions cannot be mapped safely to that billing contract.\n"
            + f"- The local fail-closed catalog therefore has no safe billable mapping for configured model(s): `{unmapped}`.\n"
            + "- A safe, exact USD cap therefore cannot be enforced before sending a request. The benchmark was not started.",
            "## Metrics\n"
            + "All quality, latency, fallback-recovery, token, and USD cost metrics are `unknown` with `n=0`. "
            + "Zero requests is not reported as zero cost per task.",
            "## Reproduction\n"
            + f"- Command: `{manifest['execution']['command']}`\n"
            + f"- Pricing catalog: `{manifest['pricing']['catalog_version']}`\n"
            + f"- Official pricing source: {manifest['pricing']['source']}\n"
            + "- Cost scope, if unblocked: `LLM-only`; embedding and rerank would remain excluded.",
            "## Machine Summary\n```json\n"
            + json.dumps(summary, ensure_ascii=False, indent=2)
            + "\n```",
        ]
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write a no-request blocked real-provider benchmark artifact."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--dataset",
        default=str(PROJECT_ROOT / "eval" / "datasets" / "questions_v3_200.jsonl"),
    )
    parser.add_argument(
        "--pricing-catalog",
        default=str(DEFAULT_CATALOG_PATH),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "eval" / "results" / "production"),
    )
    args = parser.parse_args()
    git_snapshot = _clean_git_snapshot()

    dataset_path = Path(args.dataset)
    catalog_path = Path(args.pricing_catalog)
    run_dir = Path(args.output_dir) / args.run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory already has outputs: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    catalog = load_pricing_catalog(catalog_path)
    models = sorted(
        {
            str(settings.llm_model),
            str(settings.planner_model or settings.llm_model),
            str(settings.reflection_model or settings.llm_model),
        }
    )
    priced_models = catalog.get("models") or {}
    unmapped_models = [model for model in models if model not in priced_models]
    timestamp = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": "1.0",
        "run_id": args.run_id,
        "status": "blocked",
        "started_at": None,
        "blocked_at": timestamp,
        "finished_at": timestamp,
        "entrypoint_if_unblocked": "eval/run_agentic_rag_eval.py",
        "git": git_snapshot,
        "dataset": str(dataset_path),
        "dataset_sha256": _sha256_file(dataset_path),
        "sample_count": 0,
        "provider": infer_provider(settings.llm_api_base),
        "models": models,
        "non_sensitive_config": {
            "concurrency": 1,
            "warmup": False,
            "request_timeout_s": 120,
            "external_api_allowed": False,
        },
        "execution": {
            "command": shlex.join([sys.executable, *sys.argv]),
            "external_requests": 0,
            "real_provider_invoked": False,
        },
        "pricing": {
            "catalog_path": str(catalog_path),
            "catalog_version": catalog.get("catalog_version"),
            "source": (catalog.get("source") or {}).get("url"),
            "billing_origin": "unverified",
            "models_without_safe_billable_mapping": unmapped_models,
            "mapping_status": "incomplete_or_ambiguous",
            "mapping_limitations": [
                "configured_endpoint_billing_origin_unverified",
                "m3_context_tier_and_cache_usage_dimensions_not_safely_attributable",
            ],
            "cost_scope": "llm_only",
        },
        "blockers": [
            "billing_origin_unverified",
            "provider_usage_dimensions_insufficient_for_exact_cost",
            "safe_exact_cost_cap_unavailable",
        ],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    summary = {
        "schema_version": "1.0",
        "status": "blocked",
        "sample_count": 0,
        "external_requests": 0,
        "metrics_status": "unknown",
        "blocked_reasons": list(manifest["blockers"]),
        "task_metrics": _empty_task_metrics(),
    }

    (run_dir / "per_question.jsonl").write_text("", encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "report.md").write_text(
        _render_report(manifest, summary), encoding="utf-8"
    )
    print(f"Saved blocked real-provider benchmark to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
