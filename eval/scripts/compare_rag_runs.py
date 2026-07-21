"""Compare two pure-RAG runs with paired question-level bootstrap intervals."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DEFAULT_METRICS = [
    "ndcg_at_5",
    "recall_at_5",
    "mrr",
    "context_chunk_precision",
    "context_recall",
    "latency_s",
]


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile from an empty list")
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _rows_by_qid(rows: list[dict]) -> dict[str, dict]:
    indexed = {str(row.get("qid") or ""): row for row in rows}
    if "" in indexed or len(indexed) != len(rows):
        raise ValueError("Run contains missing or duplicate question ids")
    return indexed


def compare_metric_rows(
    baseline_rows: list[dict],
    candidate_rows: list[dict],
    *,
    metric: str,
    bootstrap_samples: int = 10_000,
    seed: int = 20260711,
) -> dict:
    baseline = _rows_by_qid(baseline_rows)
    candidate = _rows_by_qid(candidate_rows)
    qids = sorted(set(baseline) & set(candidate))
    paired_qids = [
        qid
        for qid in qids
        if baseline[qid].get(metric) is not None
        and candidate[qid].get(metric) is not None
    ]
    deltas = [
        float(candidate[qid][metric]) - float(baseline[qid][metric])
        for qid in paired_qids
    ]
    if not deltas:
        return {"metric": metric, "count": 0}

    rng = random.Random(seed)
    boot_means = []
    for _ in range(max(1, bootstrap_samples)):
        boot_means.append(sum(rng.choice(deltas) for _ in deltas) / len(deltas))

    baseline_mean = sum(float(baseline[qid][metric]) for qid in paired_qids) / len(deltas)
    candidate_mean = sum(float(candidate[qid][metric]) for qid in paired_qids) / len(deltas)
    mean_delta = sum(deltas) / len(deltas)
    epsilon = 1e-12
    lower_is_better = metric in {"latency_s"}
    signed_outcomes = [-delta if lower_is_better else delta for delta in deltas]
    return {
        "metric": metric,
        "count": len(deltas),
        "baseline_mean": round(baseline_mean, 6),
        "candidate_mean": round(candidate_mean, 6),
        "mean_delta": round(mean_delta, 6),
        "relative_delta_pct": round(mean_delta / abs(baseline_mean) * 100, 2)
        if abs(baseline_mean) > epsilon
        else None,
        "ci95_low": round(_percentile(boot_means, 0.025), 6),
        "ci95_high": round(_percentile(boot_means, 0.975), 6),
        "wins": sum(delta > epsilon for delta in signed_outcomes),
        "ties": sum(abs(delta) <= epsilon for delta in signed_outcomes),
        "losses": sum(delta < -epsilon for delta in signed_outcomes),
    }


def compare_runs(
    baseline_rows: list[dict],
    candidate_rows: list[dict],
    *,
    metrics: list[str] = DEFAULT_METRICS,
    bootstrap_samples: int = 10_000,
    seed: int = 20260711,
) -> dict:
    baseline_ids = set(_rows_by_qid(baseline_rows))
    candidate_ids = set(_rows_by_qid(candidate_rows))
    if baseline_ids != candidate_ids:
        raise ValueError(
            "Question id mismatch between runs: "
            f"baseline_only={sorted(baseline_ids - candidate_ids)[:5]}, "
            f"candidate_only={sorted(candidate_ids - baseline_ids)[:5]}"
        )
    return {
        "question_count": len(baseline_ids),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "metrics": [
            compare_metric_rows(
                baseline_rows,
                candidate_rows,
                metric=metric,
                bootstrap_samples=bootstrap_samples,
                seed=seed + idx,
            )
            for idx, metric in enumerate(metrics)
        ],
    }


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _render_markdown(comparison: dict, baseline: Path, candidate: Path) -> str:
    lines = [
        "# Paired RAG Run Comparison",
        "",
        f"- Baseline: `{baseline}`",
        f"- Candidate: `{candidate}`",
        f"- Questions: {comparison['question_count']}",
        f"- Bootstrap samples: {comparison['bootstrap_samples']}",
        "",
        "| Metric | Baseline | Candidate | Delta | 95% CI | W/T/L |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparison["metrics"]:
        if not row.get("count"):
            continue
        lines.append(
            f"| {row['metric']} | {row['baseline_mean']:.4f} | {row['candidate_mean']:.4f} | "
            f"{row['mean_delta']:+.4f} | [{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] | "
            f"{row['wins']}/{row['ties']}/{row['losses']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260711)
    args = parser.parse_args()

    comparison = compare_runs(
        _load_jsonl(args.baseline),
        _load_jsonl(args.candidate),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output / "comparison.md").write_text(
        _render_markdown(comparison, args.baseline, args.candidate),
        encoding="utf-8",
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
