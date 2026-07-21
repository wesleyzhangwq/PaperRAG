"""Run Cite Scope eval: call /chat for each question, compute metrics, append CSV.

Usage:
    cd backend
    python ../eval/run_eval.py [--dataset PATH] [--run-id ID] [--judge] [--notes TEXT]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import re
import signal
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from starlette.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))

from backend.app.main import app  # noqa: E402
from eval.metrics import context_precision, ndcg_at_k, precision_at_k  # noqa: E402

try:
    import tiktoken
except Exception:
    tiktoken = None

CITATION_RE = re.compile(r"\[arxiv:([0-9]{4}\.[0-9]{4,6})\]")


def load_questions(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No questions found in {path}")
    return rows


def first_relevant_rank(pred_pids: list[str], expected_pids: set[str]) -> int | None:
    if not expected_pids:
        return None
    for idx, pid in enumerate(pred_pids, start=1):
        if pid in expected_pids:
            return idx
    return None


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = max(0, int(len(sorted_vals) * p) - 1)
    return float(sorted_vals[idx])


def estimate_tokens(text: str) -> int:
    text = text or ""
    if not text:
        return 0
    if tiktoken is not None:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


def _compute_bucket_metrics(bucket: list[dict]) -> dict:
    """Compute aggregate metrics for a bucket of per-question results."""
    retrieval_items = [r for r in bucket if r.get("has_expected")]
    n = max(1, len(retrieval_items))

    ndcg = sum(r.get("ndcg", 0) for r in retrieval_items) / n if retrieval_items else 0
    prec = sum(r.get("precision", 0) for r in retrieval_items) / n if retrieval_items else 0
    recall = sum(r.get("recall", 0) for r in retrieval_items) / n if retrieval_items else 0
    mrr = sum(r.get("rr", 0) for r in retrieval_items) / n if retrieval_items else 0

    latencies = [r["latency"] for r in bucket if "latency" in r]
    lat_p90 = percentile(latencies, 0.9)

    return {
        "count": len(bucket),
        "ndcg_5": round(ndcg, 4),
        "precision_5": round(prec, 4),
        "recall_5": round(recall, 4),
        "mrr": round(mrr, 4),
        "latency_p90": round(lat_p90, 3),
    }


def run_eval(questions: list[dict], use_judge: bool = False) -> dict:
    per_question: list[dict] = []

    judge_fn = None
    if use_judge:
        from eval.judge import judge_answer
        judge_fn = judge_answer

    with TestClient(app) as client:
        for i, item in enumerate(questions):
            query = item["query"]
            expected_pids = set(item.get("expected_paper_ids") or [])
            difficulty = item.get("difficulty", "unknown")
            qtype = item.get("type", "unknown")

            print(
                f"  [{i+1}/{len(questions)}] {item.get('qid', '?')} "
                f"({qtype}/{difficulty}): {query[:60]}...",
                file=sys.stderr,
            )

            t0 = time.time()
            PER_QUESTION_TIMEOUT = 300  # 5 minutes max per question

            # Retry up to 2 times on transient errors (SSL, connection, 500)
            resp = None
            MAX_RETRIES = 2
            for attempt in range(MAX_RETRIES + 1):
                try:
                    # Use thread-pool to enforce per-question timeout
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        future = pool.submit(client.post, "/chat", json={"query": query})
                        resp = future.result(timeout=PER_QUESTION_TIMEOUT)
                    if resp.status_code == 500 and attempt < MAX_RETRIES:
                        print(f"    Retry {attempt+1}: server error 500", file=sys.stderr)
                        time.sleep(2)
                        continue
                    break
                except concurrent.futures.TimeoutError:
                    print(f"    TIMEOUT ({PER_QUESTION_TIMEOUT}s) on attempt {attempt+1}", file=sys.stderr)
                    resp = None
                    break  # Don't retry timeouts — move to next question
                except Exception as e:
                    if attempt < MAX_RETRIES:
                        print(f"    Retry {attempt+1}: {type(e).__name__}: {e}", file=sys.stderr)
                        time.sleep(3)
                    else:
                        print(f"    FAILED after {MAX_RETRIES+1} attempts: {e}", file=sys.stderr)
                        resp = None

            latency = time.time() - t0

            result: dict = {
                "qid": item.get("qid"),
                "difficulty": difficulty,
                "type": qtype,
                "latency": latency,
                "has_expected": bool(expected_pids),
            }

            if resp is None or resp.status_code != 200:
                status = resp.status_code if resp else "exception"
                print(f"    ERROR: status {status}", file=sys.stderr)
                result.update({"ndcg": 0, "precision": 0, "recall": 0, "rr": 0})
                per_question.append(result)
                continue

            body = resp.json()
            answer = body.get("answer", "")
            sources = body.get("sources") or []
            pred_pids = [s.get("paper_id") for s in sources if s.get("paper_id")]
            cited_pids = CITATION_RE.findall(answer)
            tokens = estimate_tokens(query) + estimate_tokens(answer)

            result["tokens"] = tokens
            result["cited_count"] = len(cited_pids)
            result["source_count"] = len(pred_pids)

            if expected_pids:
                rank = first_relevant_rank(pred_pids[:5], expected_pids)
                result["ndcg"] = ndcg_at_k(pred_pids, expected_pids, k=5)
                result["precision"] = precision_at_k(pred_pids, expected_pids, k=5)
                top5_hits = len(set(pred_pids[:5]) & expected_pids)
                result["recall"] = top5_hits / max(1, len(expected_pids))
                result["rr"] = 1.0 / rank if rank else 0.0
                result["ctx_precision"] = context_precision(cited_pids, pred_pids[:5])

            if expected_pids and item.get("expected_mode") == "answer":
                is_insufficient = "不足" in answer or "没有" in answer or "无法" in answer
                result["mode_correct"] = not is_insufficient
            elif item.get("expected_mode") == "insufficient":
                is_insufficient = "不足" in answer or "没有" in answer or "无法" in answer
                result["mode_correct"] = is_insufficient

            if judge_fn and expected_pids:
                context_text = "\n---\n".join(
                    s.get("snippet", "") for s in sources if s.get("snippet")
                )
                try:
                    scores = judge_fn(query, context_text, answer)
                    result["faithfulness"] = scores.faithfulness
                    result["relevance"] = scores.relevance
                    result["correctness"] = scores.correctness
                except Exception as e:
                    print(f"    Judge error: {e}", file=sys.stderr)

            per_question.append(result)

    overall = _compute_bucket_metrics(per_question)

    by_difficulty: dict[str, list] = defaultdict(list)
    by_type: dict[str, list] = defaultdict(list)
    for r in per_question:
        by_difficulty[r["difficulty"]].append(r)
        by_type[r["type"]].append(r)

    breakdown = {}
    for d, bucket in sorted(by_difficulty.items()):
        m = _compute_bucket_metrics(bucket)
        breakdown[f"d_{d}"] = m
    for t, bucket in sorted(by_type.items()):
        m = _compute_bucket_metrics(bucket)
        breakdown[f"t_{t}"] = m

    tokens_list = [r["tokens"] for r in per_question if "tokens" in r]
    ctx_prec_list = [r["ctx_precision"] for r in per_question if "ctx_precision" in r]
    mode_correct = [r["mode_correct"] for r in per_question if "mode_correct" in r]

    judge_keys = ["faithfulness", "relevance", "correctness"]
    judge_avgs = {}
    for k in judge_keys:
        vals = [r[k] for r in per_question if k in r]
        if vals:
            judge_avgs[f"{k}_avg"] = round(statistics.mean(vals), 2)

    return {
        **overall,
        "ctx_precision": round(statistics.mean(ctx_prec_list), 4) if ctx_prec_list else None,
        "tokens_per_request": round(statistics.mean(tokens_list), 1) if tokens_list else None,
        "mode_accuracy": round(sum(mode_correct) / max(1, len(mode_correct)), 4) if mode_correct else None,
        **judge_avgs,
        "breakdown": breakdown,
        "per_question": per_question,
    }


SUMMARY_HEADER = [
    "run_id", "timestamp", "dataset", "count",
    "ndcg_5", "precision_5", "recall_5", "mrr", "ctx_precision",
    "ndcg_5_easy", "ndcg_5_medium", "ndcg_5_hard",
    "mrr_easy", "mrr_medium", "mrr_hard",
    "faithfulness_avg", "relevance_avg", "correctness_avg",
    "mode_accuracy", "latency_p90", "tokens_per_request", "notes",
]


def append_summary(path: Path, row: dict) -> None:
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_HEADER, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Cite Scope eval.")
    parser.add_argument("--dataset", type=str, default=str(PROJECT_ROOT / "eval/datasets/questions_v2.jsonl"))
    parser.add_argument("--summary-csv", type=str, default=str(PROJECT_ROOT / "eval/results/summary.csv"))
    parser.add_argument("--detail-json", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--notes", type=str, default="")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    summary_path = Path(args.summary_csv)
    questions = load_questions(dataset_path)

    print(f"Running eval: {len(questions)} questions from {dataset_path.name}", file=sys.stderr)
    metrics = run_eval(questions=questions, use_judge=args.judge)

    run_id = args.run_id or f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    breakdown = metrics.pop("breakdown", {})
    per_question = metrics.pop("per_question", [])

    row = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_path.name,
        **metrics,
    }
    for diff in ["easy", "medium", "hard"]:
        bk = breakdown.get(f"d_{diff}", {})
        row[f"ndcg_5_{diff}"] = bk.get("ndcg_5", "")
        row[f"mrr_{diff}"] = bk.get("mrr", "")
    row["notes"] = args.notes

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    append_summary(summary_path, row)

    if args.detail_json:
        detail_path = Path(args.detail_json)
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        detail = {
            "run_id": run_id,
            "timestamp": row["timestamp"],
            "metrics": metrics,
            "breakdown": breakdown,
            "per_question": per_question,
        }
        detail_path.write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Detail saved to: {detail_path}", file=sys.stderr)

    print(json.dumps(row, ensure_ascii=False, indent=2, default=str))
    print(f"\nAppended summary to: {summary_path}", file=sys.stderr)

    print("\n--- Breakdown by difficulty ---", file=sys.stderr)
    for k, v in sorted(breakdown.items()):
        if k.startswith("d_"):
            print(f"  {k}: NDCG={v['ndcg_5']:.4f}  MRR={v['mrr']:.4f}  n={v['count']}", file=sys.stderr)

    print("\n--- Breakdown by type ---", file=sys.stderr)
    for k, v in sorted(breakdown.items()):
        if k.startswith("t_"):
            print(f"  {k}: NDCG={v['ndcg_5']:.4f}  MRR={v['mrr']:.4f}  n={v['count']}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
