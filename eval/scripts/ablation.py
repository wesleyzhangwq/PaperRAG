"""Ablation runner: sweep one parameter at a time, record results.

Usage:
    cd backend
    python ../eval/scripts/ablation.py --experiment hybrid_alpha
    python ../eval/scripts/ablation.py --experiment retrieval_k
    python ../eval/scripts/ablation.py --experiment chunk_size
    python ../eval/scripts/ablation.py --experiment final_context_k
    python ../eval/scripts/ablation.py --all
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))

EXPERIMENTS = {
    "hybrid_alpha": {
        "env_var": "HYBRID_ALPHA",
        "values": [0.0, 0.3, 0.5, 0.72, 0.85, 1.0],
        "description": "Vector vs BM25 weight (0=pure BM25, 1=pure vector)",
    },
    "retrieval_k": {
        "env_var": "RETRIEVAL_K",
        "values": [4, 8, 12, 16],
        "description": "Number of chunks retrieved",
    },
    "chunk_size": {
        "env_var": "CHUNK_SIZE",
        "values": [400, 600, 800, 1000, 1200],
        "description": "Target chunk size in characters",
        "requires_reingest": True,
    },
    "final_context_k": {
        "env_var": "FINAL_CONTEXT_K",
        "values": [2, 3, 5, 8],
        "description": "Chunks passed to LLM for answer generation",
    },
    "hybrid_oversample": {
        "env_var": "HYBRID_OVERSAMPLE",
        "values": [1.5, 2.0, 2.5, 3.0],
        "description": "Oversample multiplier for hybrid retrieval",
    },
}


def run_single(
    experiment: str,
    env_var: str,
    value,
    dataset_path: Path,
    use_judge: bool,
) -> dict:
    """Run a single eval with an env var override, return metrics dict."""
    os.environ[env_var] = str(value)

    # Force settings reload
    from app.core.config import get_settings
    get_settings.cache_clear()

    from eval.run_eval import load_questions, run_eval

    questions = load_questions(dataset_path)
    metrics = run_eval(questions=questions, use_judge=use_judge)
    breakdown = metrics.pop("breakdown", {})
    metrics.pop("per_question", None)

    result = {
        "experiment": experiment,
        "param": env_var,
        "value": value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **metrics,
    }
    for diff in ["easy", "medium", "hard"]:
        bk = breakdown.get(f"d_{diff}", {})
        result[f"ndcg_5_{diff}"] = bk.get("ndcg_5", "")
        result[f"mrr_{diff}"] = bk.get("mrr", "")

    return result


def run_experiment(
    name: str,
    config: dict,
    dataset_path: Path,
    output_dir: Path,
    use_judge: bool,
) -> list[dict]:
    env_var = config["env_var"]
    values = config["values"]

    if config.get("requires_reingest"):
        print(
            f"\n⚠️  Experiment '{name}' requires re-ingesting data for each value.",
            file=sys.stderr,
        )
        print(
            "  Run with --skip-reingest to use current chunks (useful for quick testing).",
            file=sys.stderr,
        )

    original_value = os.environ.get(env_var)
    results = []

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Experiment: {name} — {config['description']}", file=sys.stderr)
    print(f"Sweeping {env_var}: {values}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    for i, val in enumerate(values):
        print(f"\n--- {env_var}={val} ({i+1}/{len(values)}) ---", file=sys.stderr)
        t0 = time.time()
        try:
            row = run_single(name, env_var, val, dataset_path, use_judge)
            results.append(row)
            elapsed = time.time() - t0
            print(
                f"  NDCG@5={row.get('ndcg_5', '?'):.4f}  "
                f"MRR={row.get('mrr', '?'):.4f}  "
                f"Lat_p90={row.get('latency_p90', '?'):.3f}s  "
                f"({elapsed:.1f}s)",
                file=sys.stderr,
            )
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)

    if original_value is not None:
        os.environ[env_var] = original_value
    elif env_var in os.environ:
        del os.environ[env_var]

    # Force settings reload to restore original
    from app.core.config import get_settings
    get_settings.cache_clear()

    csv_path = output_dir / f"ablation_{name}.csv"
    if results:
        output_dir.mkdir(parents=True, exist_ok=True)
        header = list(results[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to: {csv_path}", file=sys.stderr)

    return results


def print_comparison(results: list[dict], param: str) -> None:
    """Print a comparison table to stderr."""
    if not results:
        return
    print(f"\n{'─'*70}", file=sys.stderr)
    print(f"{'Value':>10} {'NDCG@5':>8} {'MRR':>8} {'Recall':>8} {'Lat_p90':>10}", file=sys.stderr)
    print(f"{'─'*70}", file=sys.stderr)
    best_ndcg = max(r.get("ndcg_5", 0) for r in results)
    for r in results:
        ndcg = r.get("ndcg_5", 0)
        marker = " ★" if ndcg == best_ndcg else ""
        print(
            f"{str(r['value']):>10} {ndcg:>8.4f} {r.get('mrr', 0):>8.4f} "
            f"{r.get('recall_5', 0):>8.4f} {r.get('latency_p90', 0):>10.3f}s{marker}",
            file=sys.stderr,
        )
    print(f"{'─'*70}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--experiment", type=str, choices=list(EXPERIMENTS.keys()))
    parser.add_argument("--all", action="store_true", help="Run all experiments sequentially")
    parser.add_argument("--dataset", type=str, default=str(PROJECT_ROOT / "eval/datasets/questions_v2.jsonl"))
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "eval/results"))
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--skip-reingest", action="store_true")
    args = parser.parse_args()

    if not args.experiment and not args.all:
        parser.error("Specify --experiment NAME or --all")

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)

    experiments_to_run = list(EXPERIMENTS.keys()) if args.all else [args.experiment]

    for name in experiments_to_run:
        config = EXPERIMENTS[name]
        if config.get("requires_reingest") and args.skip_reingest:
            print(f"\nSkipping {name} (requires re-ingest)", file=sys.stderr)
            continue

        results = run_experiment(
            name=name,
            config=config,
            dataset_path=dataset_path,
            output_dir=output_dir,
            use_judge=args.judge,
        )
        print_comparison(results, config["env_var"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
