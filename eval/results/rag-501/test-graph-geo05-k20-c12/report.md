# Pure RAG Evaluation Report: test-graph-geo05-k20-c12

- Dataset: `questions_501_test_200.jsonl`
- Questions: 200 total, 180 positive, 20 negative

## Core Retrieval Metrics

- NDCG@5: 0.7012
- Recall@5: 0.7475
- MRR: 0.7357
- Context chunk precision: 0.1844
- Context recall: 0.7475
- Latency P90: 0.4895s
- Negative with context rate: 1.0000
- Negative max score mean: 0.7400

## Resume-safe metrics

- Use only metrics from this section in external materials unless a newer report supersedes it.
- NDCG@5: 0.7012
- Recall@5: 0.7475
- MRR: 0.7357
- Context precision: 0.1844
- Context recall: 0.7475

## Comparisons

- ndcg_at_5: 0.7063 -> 0.7012 (-0.72%)
- recall_at_5: 0.7531 -> 0.7475 (-0.74%)
- mrr: 0.7411 -> 0.7357 (-0.73%)
- context_chunk_precision: 0.2407 -> 0.1844 (-23.39%)
- context_recall: 0.7531 -> 0.7475 (-0.74%)
- negative_max_score_mean: 0.7400 -> 0.7400 (+0.00%)
- latency_p90: 0.4178 -> 0.4895 (+17.16%)

## Breakdown by Type

| Type | Count | NDCG@5 | Recall@5 | Context recall |
|---|---:|---:|---:|---:|
| comparison | 30 | 0.3262 | 0.3500 | 0.3500 |
| concept_locate | 60 | 0.8696 | 0.9333 | 0.9333 |
| fact_extract | 30 | 0.8587 | 0.9000 | 0.9000 |
| method_detail | 40 | 0.7868 | 0.8750 | 0.8750 |
| negative | 20 | n/a | n/a | n/a |
| trend_synthesis | 20 | 0.3507 | 0.3026 | 0.3026 |

## Graph RAG Merge Gates

- Overall: FAIL

| Gate | Baseline | Candidate | Threshold | Pass |
|---|---:|---:|---:|:---:|
| comparison_recall_at_5 | 0.3500 | 0.3500 | 0.0500 | FAIL |
| trend_synthesis_recall_at_5 | 0.3026 | 0.3026 | 0.0500 | FAIL |
| overall_ndcg_at_5 | 0.7063 | 0.7012 | -0.0100 | PASS |
| fixed_context_citation_support | 1.0000 | n/a | 0.0000 | FAIL |
| graph_expansion_p95_ms | 0.0000 | 69.5600 | 800.0000 | PASS |
