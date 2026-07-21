# PaperRAG Failure-Injection Benchmark: paperrag-failure-injection-v3-20260721

- Production entrypoint: `app.agent.graph.run_agent_eval_sync`
- Deterministic scenarios: 10
- Concurrency: 1; warmup: false; external API calls: 0
- Git: `320ccb7a3917833f2729d5e1cb53deaf5d1452bc` (dirty=False)
- Scenario dataset SHA-256: `5acd204c8b9f75ecde8388fe8c3b07f3f80e8bf94c9275291840deaf030df600`
- Provider responses and retrieval are deterministic fixtures; no result is presented as a real-provider quality or cost measurement.

## Outcome
- scenario_pass_rate: 1.0 (10/10)
- safe_degraded_count: 2
- terminal_failure_count: 1

| Metric | Value | Sample |
|---|---:|---:|
| task_success_rate | 0.7 | n=10 |
| mode_accuracy | 0.7 | n=10 |
| citation_support_rate | 1.0 | citation cases n=8 |
| expected_source_hit_rate | 0.8 | positives=10 |
| terminal_failure_rate | 0.1 | n=10 |
| fallback_recovery_rate | 0.7778 | attempted=9 |
| latency p50 / p90 / p95 / mean (s) | 0.0104 / 0.0116 / 0.0137 / 0.0105 | n=10 |
| cost mean / p50 / p95 (USD, LLM-only) | None / None / None | known n=0 |

Cost is unknown by design because usage is synthetic and the billing origin is unverified.

## Scenario Matrix
| Scenario | Expected | Actual | Passed | Failure classes |
|---|---|---|---:|---|
| local_retrieval_empty | recovered | recovered | true | local_retrieval_empty, evidence_insufficient |
| local_retrieval_exception | recovered | recovered | true | local_retrieval_exception, evidence_insufficient |
| arxiv_web_unavailable | recovered | recovered | true | arxiv_service_unavailable, web_service_unavailable |
| planner_output_unparseable | recovered | recovered | true | planner_output_unparseable |
| sufficiency_output_unparseable | safe_degraded | safe_degraded | true | sufficiency_output_unparseable |
| llm_timeout | recovered | recovered | true | llm_timeout |
| groundedness_re_retrieve | recovered | recovered | true | groundedness_check_failed |
| groundedness_re_generate | recovered | recovered | true | groundedness_check_failed |
| supplement_budget_exhausted | safe_degraded | safe_degraded | true | local_retrieval_empty, evidence_insufficient, supplement_retrieval_budget_exhausted |
| terminal_database_failure | terminal_failure | terminal_failure | true | terminal_exception |

## Metric Definitions
`fallback_recovery_rate = strict task-success fallbacks / fallback_attempts`. Safe degraded responses and expected terminal injections do not count as recovered. P95 is descriptive for this small deterministic scenario set.
