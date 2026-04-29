from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from starlette.testclient import TestClient

# Make backend package importable when run as script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))

from backend.app.main import app  # noqa: E402

try:
    import tiktoken
except Exception:  # pragma: no cover - fallback when tiktoken unavailable
    tiktoken = None


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


def run_eval(questions: list[dict], top_k: int, final_k: int) -> dict:
    latencies: list[float] = []
    hit_count = 0
    rr_values: list[float] = []
    recall_values: list[float] = []
    insufficient_count = 0
    answer_correct_count = 0
    tokens_per_req: list[int] = []

    with TestClient(app) as client:
        for item in questions:
            query = item["query"]
            expected_pids = set(item.get("expected_paper_ids") or [])
            expected_mode = item.get("expected_mode", "answer")

            t0 = time.time()
            resp = client.post("/chat", json={"query": query, "top_k": top_k, "final_k": final_k})
            latency = time.time() - t0
            latencies.append(latency)

            if resp.status_code != 200:
                rr_values.append(0.0)
                recall_values.append(0.0)
                tokens_per_req.append(estimate_tokens(query))
                continue

            body = resp.json()
            answer = body.get("answer", "")
            sources = body.get("sources") or []
            pred_pids = [s.get("paper_id") for s in sources if s.get("paper_id")]
            tokens_per_req.append(estimate_tokens(query) + estimate_tokens(answer))

            # insufficient tracking (actual behavior)
            actual_insufficient = "参考资料不足" in answer
            if actual_insufficient:
                insufficient_count += 1

            # retrieval metrics only for items with expected paper ids
            if expected_pids:
                rank = first_relevant_rank(pred_pids[:5], expected_pids)
                top5_hits = len(set(pred_pids[:5]) & expected_pids)
                recall_values.append(top5_hits / max(1, len(expected_pids)))
                if rank is not None:
                    hit_count += 1
                    rr_values.append(1.0 / rank)
                else:
                    rr_values.append(0.0)
                # Heuristic correctness for labeled answer queries
                if expected_mode == "answer":
                    is_correct = (not actual_insufficient) and (top5_hits > 0)
                else:
                    is_correct = actual_insufficient
            else:
                # keep denominator stable for MRR by appending 0 for non-labeled
                rr_values.append(0.0)
                # not counted in recall denominator
                if expected_mode == "insufficient":
                    is_correct = actual_insufficient
                else:
                    is_correct = (not actual_insufficient) and (len(pred_pids) > 0)

            if is_correct:
                answer_correct_count += 1

    total = len(questions)
    retrieval_labeled = sum(1 for q in questions if (q.get("expected_paper_ids") or []))
    retrieval_den = max(1, retrieval_labeled)
    recall_avg = round(sum(recall_values) / retrieval_den, 4) if recall_values else 0.0

    return {
        "answer_correctness": round(answer_correct_count / total, 4),
        "tokens_per_request": round(statistics.mean(tokens_per_req), 2) if tokens_per_req else 0.0,
        "recall": recall_avg,
        "latency_p90": round(percentile(latencies, 0.9), 3),
        "hit_at_5": round(hit_count / retrieval_den, 4),
        "mrr": round(sum(rr_values) / retrieval_den, 4),
        "insufficient_ratio": round(insufficient_count / total, 4),
    }


def append_summary(path: Path, row: dict) -> None:
    header = [
        "run_id",
        "timestamp",
        "dataset",
        "strategy",
        "answer_correctness",
        "tokens_per_request",
        "recall",
        "latency_p90",
        "hit_at_5",
        "mrr",
        "insufficient_ratio",
        "notes",
    ]
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PaperRAG eval and append summary CSV.")
    parser.add_argument("--dataset", type=str, default=str(PROJECT_ROOT / "eval/datasets/questions_v1.jsonl"))
    parser.add_argument("--summary-csv", type=str, default=str(PROJECT_ROOT / "eval/results/summary.csv"))
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--strategy", type=str, default=os.getenv("CHUNK_STRATEGY", "v2"))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--final-k", type=int, default=3)
    parser.add_argument("--notes", type=str, default="")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    summary_path = Path(args.summary_csv)
    questions = load_questions(dataset_path)

    metrics = run_eval(questions=questions, top_k=args.top_k, final_k=args.final_k)
    run_id = args.run_id or f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    row = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_path.name,
        "strategy": args.strategy,
        **metrics,
        "notes": args.notes,
    }
    append_summary(summary_path, row)

    print(json.dumps(row, ensure_ascii=False, indent=2))
    print(f"Appended summary to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
