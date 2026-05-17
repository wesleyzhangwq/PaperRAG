from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
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


def run_eval(questions: list[dict], top_k: int, final_k: int, use_judge: bool = False) -> dict:
    latencies: list[float] = []
    rr_values: list[float] = []
    recall_values: list[float] = []
    ndcg_values: list[float] = []
    precision_values: list[float] = []
    ctx_precision_values: list[float] = []
    tokens_per_req: list[int] = []

    judge_faithfulness: list[int] = []
    judge_relevance: list[int] = []
    judge_correctness: list[int] = []

    judge_fn = None
    if use_judge:
        from eval.judge import judge_answer
        judge_fn = judge_answer

    with TestClient(app) as client:
        for item in questions:
            query = item["query"]
            expected_pids = set(item.get("expected_paper_ids") or [])

            t0 = time.time()
            resp = client.post(
                "/chat",
                json={"query": query, "top_k": top_k, "final_k": final_k},
                params={"mode": "pipeline"},
            )
            latency = time.time() - t0
            latencies.append(latency)

            if resp.status_code != 200:
                rr_values.append(0.0)
                recall_values.append(0.0)
                ndcg_values.append(0.0)
                precision_values.append(0.0)
                tokens_per_req.append(estimate_tokens(query))
                continue

            body = resp.json()
            answer = body.get("answer", "")
            sources = body.get("sources") or []
            pred_pids = [s.get("paper_id") for s in sources if s.get("paper_id")]
            tokens_per_req.append(estimate_tokens(query) + estimate_tokens(answer))

            cited_pids = CITATION_RE.findall(answer)

            if expected_pids:
                rank = first_relevant_rank(pred_pids[:5], expected_pids)
                ndcg_values.append(ndcg_at_k(pred_pids, expected_pids, k=5))
                precision_values.append(precision_at_k(pred_pids, expected_pids, k=5))
                top5_hits = len(set(pred_pids[:5]) & expected_pids)
                recall_values.append(top5_hits / max(1, len(expected_pids)))
                rr_values.append(1.0 / rank if rank is not None else 0.0)

            ctx_precision_values.append(context_precision(cited_pids, pred_pids[:5]))

            if judge_fn is not None:
                context_text = "\n---\n".join(
                    s.get("snippet", "") for s in sources if s.get("snippet")
                )
                try:
                    scores = judge_fn(query, context_text, answer)
                    judge_faithfulness.append(scores.faithfulness)
                    judge_relevance.append(scores.relevance)
                    judge_correctness.append(scores.correctness)
                except Exception as e:
                    print(f"Judge error for query '{query[:50]}': {e}", file=sys.stderr)

    retrieval_labeled = sum(1 for q in questions if (q.get("expected_paper_ids") or []))
    retrieval_den = max(1, retrieval_labeled)

    result = {
        "ndcg_5": round(sum(ndcg_values) / retrieval_den, 4) if ndcg_values else 0.0,
        "precision_5": round(sum(precision_values) / retrieval_den, 4) if precision_values else 0.0,
        "recall_5": round(sum(recall_values) / retrieval_den, 4) if recall_values else 0.0,
        "mrr": round(sum(rr_values) / retrieval_den, 4) if rr_values else 0.0,
        "context_precision": round(statistics.mean(ctx_precision_values), 4) if ctx_precision_values else 0.0,
        "latency_p90": round(percentile(latencies, 0.9), 3),
        "tokens_per_request": round(statistics.mean(tokens_per_req), 2) if tokens_per_req else 0.0,
    }

    if judge_faithfulness:
        result["faithfulness_avg"] = round(statistics.mean(judge_faithfulness), 2)
        result["relevance_avg"] = round(statistics.mean(judge_relevance), 2)
        result["correctness_avg"] = round(statistics.mean(judge_correctness), 2)

    return result


def append_summary(path: Path, row: dict) -> None:
    header = [
        "run_id", "timestamp", "dataset",
        "ndcg_5", "precision_5", "recall_5", "mrr", "context_precision",
        "faithfulness_avg", "relevance_avg", "correctness_avg",
        "latency_p90", "tokens_per_request", "notes",
    ]
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PaperRAG eval and append summary CSV.")
    parser.add_argument("--dataset", type=str, default=str(PROJECT_ROOT / "eval/datasets/questions_v1.jsonl"))
    parser.add_argument("--summary-csv", type=str, default=str(PROJECT_ROOT / "eval/results/summary.csv"))
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--final-k", type=int, default=3)
    parser.add_argument("--judge", action="store_true", help="Enable LLM-as-Judge scoring")
    parser.add_argument("--notes", type=str, default="")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    summary_path = Path(args.summary_csv)
    questions = load_questions(dataset_path)

    metrics = run_eval(
        questions=questions,
        top_k=args.top_k,
        final_k=args.final_k,
        use_judge=args.judge,
    )
    run_id = args.run_id or f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    row = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_path.name,
        **metrics,
        "notes": args.notes,
    }
    append_summary(summary_path, row)

    print(json.dumps(row, ensure_ascii=False, indent=2))
    print(f"Appended summary to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
