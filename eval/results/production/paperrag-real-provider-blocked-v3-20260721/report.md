# PaperRAG Real-Provider Benchmark: paperrag-real-provider-blocked-v3-20260721

- Status: **blocked; no external request was sent**
- Sample size: 0; external requests: 0
- Configured provider: `minimax`; models: `MiniMax-M3`
- Git: `320ccb7a3917833f2729d5e1cb53deaf5d1452bc` (dirty=False)
- Dataset SHA-256: `30f3fec3850e80a8c66cc9aa55504784d32c716c89eff5a98b87e471a16e96ba`

## Blocking Evidence
- The configured endpoint's billing origin is unverified; a compatible host does not prove official pay-as-you-go billing.
- MiniMax publishes context-tiered M3 input, output, and cache-read rates, but the configured endpoint and returned usage dimensions cannot be mapped safely to that billing contract.
- The local fail-closed catalog therefore has no safe billable mapping for configured model(s): `MiniMax-M3`.
- A safe, exact USD cap therefore cannot be enforced before sending a request. The benchmark was not started.

## Metrics
All quality, latency, fallback-recovery, token, and USD cost metrics are `unknown` with `n=0`. Zero requests is not reported as zero cost per task.

## Reproduction
- Command: `/Users/wesz_station/Projects/PaperRAG/backend/.venv/bin/python eval/write_blocked_real_benchmark.py --run-id paperrag-real-provider-blocked-v3-20260721`
- Pricing catalog: `minimax-paygo-2026-07-21`
- Official pricing source: https://platform.minimax.io/docs/guides/pricing-paygo
- Cost scope, if unblocked: `LLM-only`; embedding and rerank would remain excluded.

## Machine Summary
```json
{
  "schema_version": "1.0",
  "status": "blocked",
  "sample_count": 0,
  "external_requests": 0,
  "metrics_status": "unknown",
  "blocked_reasons": [
    "billing_origin_unverified",
    "provider_usage_dimensions_insufficient_for_exact_cost",
    "safe_exact_cost_cap_unavailable"
  ],
  "task_metrics": {
    "count": 0,
    "count_positive": 0,
    "count_negative": 0,
    "task_success_rate": null,
    "mode_accuracy": null,
    "citation_support_rate": null,
    "citation_support_n": 0,
    "expected_source_hit_rate": null,
    "terminal_failure_rate": null,
    "latency_p50": null,
    "latency_p90": null,
    "latency_p95": null,
    "latency_mean": null,
    "latency_n": 0,
    "latency_by_path": {
      "normal": {
        "n": 0,
        "p50": null,
        "p90": null,
        "p95": null,
        "mean": null
      },
      "fallback": {
        "n": 0,
        "p50": null,
        "p90": null,
        "p95": null,
        "mean": null
      },
      "terminal": {
        "n": 0,
        "p50": null,
        "p90": null,
        "p95": null,
        "mean": null
      }
    },
    "stage_latency": {},
    "fallback_attempted_count": 0,
    "fallback_recovered_count": 0,
    "fallback_recovery_rate": null,
    "cost_known_n": 0,
    "cost_unknown_n": 0,
    "cost_per_task_mean_usd": null,
    "cost_per_task_p50_usd": null,
    "cost_per_task_p95_usd": null,
    "successful_task_cost_mean_usd": null,
    "fallback_task_cost_mean_usd": null,
    "by_type": {},
    "by_difficulty": {}
  }
}
```
