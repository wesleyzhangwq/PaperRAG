"""Retrieval-only evaluation: measures retrieval quality WITHOUT calling the LLM.

Cost: only embedding API calls (~¥0.01 per 70 queries).
Speed: ~2-5 minutes for 70 questions (vs 2+ hours with full pipeline).

Usage:
    cd backend
    python ../eval/run_retrieval_eval.py --run-id "retrieval-baseline"
    python ../eval/run_retrieval_eval.py --run-id "retrieval-baseline" --detail-json ../eval/results/retrieval-detail.json

Ablation mode (sweep one parameter):
    python ../eval/run_retrieval_eval.py --sweep retrieval_k     --values 4 8 12 16 20
    python ../eval/run_retrieval_eval.py --sweep hybrid_alpha    --values 0.0 0.3 0.5 0.72 0.85 1.0
    python ../eval/run_retrieval_eval.py --sweep final_context_k --values 2 3 5 8
    python ../eval/run_retrieval_eval.py --sweep hybrid_oversample --values 1.5 2.0 2.5 3.0
    python ../eval/run_retrieval_eval.py --sweep-all
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))

from eval.metrics import ndcg_at_k, precision_at_k  # noqa: E402

# ---------------------------------------------------------------------------
# Sweep parameter definitions
# ---------------------------------------------------------------------------
SWEEP_PARAMS = {
    "retrieval_k": {
        "env_var": "RETRIEVAL_K",
        "values": [4, 8, 12, 16, 20],
        "label": "检索块数",
    },
    "hybrid_alpha": {
        "env_var": "HYBRID_ALPHA",
        "values": [0.0, 0.3, 0.5, 0.72, 0.85, 1.0],
        "label": "向量/BM25权重",
    },
    "final_context_k": {
        "env_var": "FINAL_CONTEXT_K",
        "values": [2, 3, 5, 8],
        "label": "LLM上下文块数",
    },
    "hybrid_oversample": {
        "env_var": "HYBRID_OVERSAMPLE",
        "values": [1.5, 2.0, 2.5, 3.0, 4.0],
        "label": "过采样因子",
    },
}


def load_questions(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        # Skip placeholder questions
        if item.get("query") in ("<question text>", "..."):
            continue
        rows.append(item)
    if not rows:
        raise ValueError(f"No valid questions in {path}")
    return rows


def recall_at_k(pred_pids: list[str], expected_pids: set[str], k: int = 5) -> float:
    if not expected_pids:
        return 0.0
    pred = pred_pids[:k]
    hits = sum(1 for pid in pred if pid in expected_pids)
    return hits / len(expected_pids)


def mrr_score(pred_pids: list[str], expected_pids: set[str]) -> float:
    if not expected_pids:
        return 0.0
    for idx, pid in enumerate(pred_pids, start=1):
        if pid in expected_pids:
            return 1.0 / idx
    return 0.0


def _reset_settings_and_cache():
    """Clear all caches so env var changes take effect."""
    from app.core.config import get_settings
    get_settings.cache_clear()
    # Clear Qdrant/embedding singletons
    from app.db.qdrant import get_qdrant_vector_store, get_embeddings
    get_qdrant_vector_store.cache_clear()
    get_embeddings.cache_clear()


def run_retrieval_eval(
    questions: list[dict],
    k: int = 5,
) -> dict:
    """Evaluate retrieval quality by directly calling retrieve() — no LLM."""
    from app.services.retriever import retrieve

    per_question = []
    latencies = []

    for i, item in enumerate(questions):
        query = item["query"]
        expected_pids = set(item.get("expected_paper_ids") or [])
        expected_mode = item.get("expected_mode", "answer")
        difficulty = item.get("difficulty", "unknown")
        qtype = item.get("type", "unknown")
        qid = item.get("qid", f"q{i}")

        # For negative questions, expected behavior is to return nothing relevant
        is_negative = qtype == "negative" or expected_mode == "refuse"

        t0 = time.time()
        try:
            results = retrieve(query)
        except Exception as e:
            print(f"  [{i+1}/{len(questions)}] ERROR: {e}", file=sys.stderr)
            per_question.append({
                "qid": qid,
                "difficulty": difficulty,
                "type": qtype,
                "error": str(e),
                "ndcg": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "rr": 0.0,
                "has_expected": bool(expected_pids),
                "latency": 0.0,
            })
            continue

        elapsed = time.time() - t0
        latencies.append(elapsed)

        # Extract paper_ids from retrieved documents
        pred_pids = []
        seen = set()
        for doc, score in results:
            pid = (doc.metadata or {}).get("paper_id", "")
            if pid and pid not in seen:
                pred_pids.append(pid)
                seen.add(pid)

        # Compute metrics
        ndcg = ndcg_at_k(pred_pids, expected_pids, k) if expected_pids else 0.0
        prec = precision_at_k(pred_pids, expected_pids, k) if expected_pids else 0.0
        rec = recall_at_k(pred_pids, expected_pids, k) if expected_pids else 0.0
        rr = mrr_score(pred_pids, expected_pids) if expected_pids else 0.0

        status_icon = "✓" if (ndcg > 0 or is_negative) else "✗"
        print(
            f"  [{i+1}/{len(questions)}] {status_icon} {qid} ({qtype}/{difficulty}) "
            f"NDCG={ndcg:.3f} MRR={rr:.3f} "
            f"retrieved={len(pred_pids)} pids, "
            f"expected={list(expected_pids)[:3]}, "
            f"got=[{', '.join(pred_pids[:3])}] "
            f"({elapsed:.2f}s)",
            file=sys.stderr,
        )

        per_question.append({
            "qid": qid,
            "query": query[:80],
            "difficulty": difficulty,
            "type": qtype,
            "ndcg": round(ndcg, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "rr": round(rr, 4),
            "has_expected": bool(expected_pids),
            "expected_pids": list(expected_pids),
            "predicted_pids": pred_pids[:k],
            "num_chunks_retrieved": len(results),
            "latency": round(elapsed, 3),
            "is_negative": is_negative,
        })

    # Aggregate metrics
    retrieval_items = [r for r in per_question if r.get("has_expected")]
    n = max(1, len(retrieval_items))

    overall = {
        "count": len(per_question),
        "count_with_expected": len(retrieval_items),
        "ndcg_5": round(sum(r["ndcg"] for r in retrieval_items) / n, 4),
        "precision_5": round(sum(r["precision"] for r in retrieval_items) / n, 4),
        "recall_5": round(sum(r["recall"] for r in retrieval_items) / n, 4),
        "mrr": round(sum(r["rr"] for r in retrieval_items) / n, 4),
        "latency_p50": round(sorted(latencies)[len(latencies)//2], 3) if latencies else 0,
        "latency_p90": round(sorted(latencies)[int(len(latencies)*0.9)] if latencies else 0, 3),
        "latency_mean": round(sum(latencies)/len(latencies), 3) if latencies else 0,
    }

    # Breakdowns
    breakdown = {}
    for key_fn, prefix in [
        (lambda r: r["difficulty"], "d"),
        (lambda r: r["type"], "t"),
    ]:
        buckets = defaultdict(list)
        for r in per_question:
            buckets[key_fn(r)].append(r)
        for bname, items in sorted(buckets.items()):
            b_retrieval = [r for r in items if r.get("has_expected")]
            bn = max(1, len(b_retrieval))
            breakdown[f"{prefix}_{bname}"] = {
                "count": len(items),
                "ndcg_5": round(sum(r["ndcg"] for r in b_retrieval) / bn, 4) if b_retrieval else 0,
                "precision_5": round(sum(r["precision"] for r in b_retrieval) / bn, 4) if b_retrieval else 0,
                "recall_5": round(sum(r["recall"] for r in b_retrieval) / bn, 4) if b_retrieval else 0,
                "mrr": round(sum(r["rr"] for r in b_retrieval) / bn, 4) if b_retrieval else 0,
            }

    return {
        **overall,
        "breakdown": breakdown,
        "per_question": per_question,
    }


def print_results(metrics: dict, run_id: str = ""):
    """Pretty-print evaluation results."""
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"  Retrieval Evaluation Results{f' ({run_id})' if run_id else ''}", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)

    print(f"\n  Overall ({metrics['count']} questions, {metrics['count_with_expected']} with expected):", file=sys.stderr)
    print(f"    NDCG@5     = {metrics['ndcg_5']:.4f}", file=sys.stderr)
    print(f"    MRR        = {metrics['mrr']:.4f}", file=sys.stderr)
    print(f"    Precision@5= {metrics['precision_5']:.4f}", file=sys.stderr)
    print(f"    Recall@5   = {metrics['recall_5']:.4f}", file=sys.stderr)
    print(f"    Latency p90= {metrics['latency_p90']:.3f}s", file=sys.stderr)

    breakdown = metrics.get("breakdown", {})
    # Difficulty breakdown
    print(f"\n  By Difficulty:", file=sys.stderr)
    print(f"    {'Level':<10} {'Count':>6} {'NDCG@5':>8} {'MRR':>8} {'Recall':>8}", file=sys.stderr)
    print(f"    {'─'*42}", file=sys.stderr)
    for d in ["easy", "medium", "hard"]:
        bk = breakdown.get(f"d_{d}", {})
        if bk:
            print(f"    {d:<10} {bk['count']:>6} {bk['ndcg_5']:>8.4f} {bk['mrr']:>8.4f} {bk['recall_5']:>8.4f}", file=sys.stderr)

    # Type breakdown
    print(f"\n  By Type:", file=sys.stderr)
    print(f"    {'Type':<20} {'Count':>6} {'NDCG@5':>8} {'MRR':>8} {'Recall':>8}", file=sys.stderr)
    print(f"    {'─'*54}", file=sys.stderr)
    for t in ["concept_locate", "method_detail", "fact_extract", "comparison", "trend_synthesis", "negative"]:
        bk = breakdown.get(f"t_{t}", {})
        if bk:
            print(f"    {t:<20} {bk['count']:>6} {bk['ndcg_5']:>8.4f} {bk['mrr']:>8.4f} {bk['recall_5']:>8.4f}", file=sys.stderr)

    print(f"\n{'='*70}\n", file=sys.stderr)


def run_sweep(
    param_name: str,
    values: list,
    env_var: str,
    questions: list[dict],
    output_dir: Path,
) -> list[dict]:
    """Sweep a single parameter and return comparison results."""
    original = os.environ.get(env_var)
    results = []

    print(f"\n{'='*70}", file=sys.stderr)
    print(f"  Sweep: {param_name} ({env_var}) = {values}", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)

    for i, val in enumerate(values):
        os.environ[env_var] = str(val)
        _reset_settings_and_cache()

        print(f"\n--- {env_var}={val} ({i+1}/{len(values)}) ---", file=sys.stderr)
        t0 = time.time()

        metrics = run_retrieval_eval(questions)
        breakdown = metrics.pop("breakdown", {})
        metrics.pop("per_question", None)
        elapsed = time.time() - t0

        row = {
            "param": param_name,
            "value": val,
            "ndcg_5": metrics["ndcg_5"],
            "mrr": metrics["mrr"],
            "precision_5": metrics["precision_5"],
            "recall_5": metrics["recall_5"],
            "latency_p90": metrics["latency_p90"],
            "latency_mean": metrics["latency_mean"],
            "elapsed_total": round(elapsed, 1),
        }
        # Add difficulty breakdowns
        for d in ["easy", "medium", "hard"]:
            bk = breakdown.get(f"d_{d}", {})
            row[f"ndcg_5_{d}"] = bk.get("ndcg_5", "")
            row[f"mrr_{d}"] = bk.get("mrr", "")
        results.append(row)

        print(
            f"  → NDCG@5={row['ndcg_5']:.4f}  MRR={row['mrr']:.4f}  "
            f"Recall={row['recall_5']:.4f}  Lat={row['latency_p90']:.3f}s  "
            f"({elapsed:.1f}s total)",
            file=sys.stderr,
        )

    # Restore original
    if original is not None:
        os.environ[env_var] = original
    elif env_var in os.environ:
        del os.environ[env_var]
    _reset_settings_and_cache()

    # Save CSV
    if results:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"sweep_{param_name}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\n  Saved: {csv_path}", file=sys.stderr)

    # Print comparison table
    print(f"\n{'─'*70}", file=sys.stderr)
    print(f"  {'Value':>10} {'NDCG@5':>8} {'MRR':>8} {'Recall':>8} {'Lat_p90':>10}", file=sys.stderr)
    print(f"  {'─'*46}", file=sys.stderr)
    best_ndcg = max(r["ndcg_5"] for r in results)
    for r in results:
        marker = " ★" if r["ndcg_5"] == best_ndcg else ""
        print(
            f"  {str(r['value']):>10} {r['ndcg_5']:>8.4f} {r['mrr']:>8.4f} "
            f"{r['recall_5']:>8.4f} {r['latency_p90']:>10.3f}s{marker}",
            file=sys.stderr,
        )
    print(f"  {'─'*46}", file=sys.stderr)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Retrieval-only evaluation (no LLM cost)"
    )
    parser.add_argument("--run-id", type=str, default="retrieval-eval")
    parser.add_argument("--dataset", type=str,
                        default=str(PROJECT_ROOT / "eval/datasets/questions_v2.jsonl"))
    parser.add_argument("--output-dir", type=str,
                        default=str(PROJECT_ROOT / "eval/results"))
    parser.add_argument("--detail-json", type=str, default=None,
                        help="Save per-question details to JSON")
    parser.add_argument("--k", type=int, default=5)

    # Sweep mode
    parser.add_argument("--sweep", type=str, choices=list(SWEEP_PARAMS.keys()),
                        help="Sweep a single parameter")
    parser.add_argument("--values", nargs="+", type=float,
                        help="Custom values for sweep (overrides default)")
    parser.add_argument("--sweep-all", action="store_true",
                        help="Sweep all parameters sequentially")

    args = parser.parse_args()
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)

    questions = load_questions(dataset_path)
    print(f"\nLoaded {len(questions)} questions from {dataset_path.name}", file=sys.stderr)

    # --- Sweep mode ---
    if args.sweep or args.sweep_all:
        params_to_sweep = list(SWEEP_PARAMS.keys()) if args.sweep_all else [args.sweep]
        all_sweep_results = {}

        for pname in params_to_sweep:
            pconf = SWEEP_PARAMS[pname]
            values = args.values if (args.values and not args.sweep_all) else pconf["values"]
            results = run_sweep(
                param_name=pname,
                values=values,
                env_var=pconf["env_var"],
                questions=questions,
                output_dir=output_dir,
            )
            all_sweep_results[pname] = results

        # Print summary of all sweeps
        if args.sweep_all:
            print(f"\n{'='*70}", file=sys.stderr)
            print(f"  SWEEP SUMMARY — Best values per parameter", file=sys.stderr)
            print(f"{'='*70}", file=sys.stderr)
            for pname, results in all_sweep_results.items():
                best = max(results, key=lambda r: r["ndcg_5"])
                print(
                    f"  {pname:<20} best={best['value']:<8} "
                    f"NDCG@5={best['ndcg_5']:.4f}  MRR={best['mrr']:.4f}",
                    file=sys.stderr,
                )
            print(f"\n  Recommended next step:", file=sys.stderr)
            combo = {pname: max(results, key=lambda r: r["ndcg_5"])["value"]
                     for pname, results in all_sweep_results.items()}
            env_parts = " ".join(
                f"{SWEEP_PARAMS[p]['env_var']}={v}" for p, v in combo.items()
            )
            print(f"    {env_parts} python ../eval/run_retrieval_eval.py --run-id optimized", file=sys.stderr)
            print(f"{'='*70}\n", file=sys.stderr)

        return 0

    # --- Single run mode ---
    print(f"\nRunning retrieval evaluation (run_id={args.run_id})...", file=sys.stderr)
    t0 = time.time()
    metrics = run_retrieval_eval(questions, k=args.k)
    total_time = time.time() - t0

    print_results(metrics, args.run_id)
    print(f"  Total time: {total_time:.1f}s", file=sys.stderr)

    # Save summary CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "retrieval_summary.csv"
    breakdown = metrics.pop("breakdown", {})
    per_question = metrics.pop("per_question", [])

    row = {
        "run_id": args.run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_time_s": round(total_time, 1),
        **metrics,
    }
    # Add difficulty breakdowns
    for d in ["easy", "medium", "hard"]:
        bk = breakdown.get(f"d_{d}", {})
        row[f"ndcg_5_{d}"] = bk.get("ndcg_5", "")
        row[f"mrr_{d}"] = bk.get("mrr", "")

    file_exists = summary_csv.exists()
    with summary_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()), extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"  Summary appended to: {summary_csv}", file=sys.stderr)

    # Save detail JSON
    if args.detail_json:
        detail = {
            "run_id": args.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "breakdown": breakdown,
            "per_question": per_question,
        }
        detail_path = Path(args.detail_json)
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        detail_path.write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  Details saved to: {detail_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
