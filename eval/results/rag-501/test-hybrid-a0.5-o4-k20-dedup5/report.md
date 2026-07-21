# Pure RAG Evaluation Report: test-hybrid-a0.5-o4-k20-dedup5

- Dataset: `questions_501_test_200.jsonl`
- Questions: 200 total, 180 positive, 20 negative

## Core Retrieval Metrics

- NDCG@5: 0.7063
- Recall@5: 0.7531
- MRR: 0.7411
- Context chunk precision: 0.2407
- Context recall: 0.7531
- Latency P90: 0.4178s
- Negative with context rate: 1.0000
- Negative max score mean: 0.7400

## Resume-safe metrics

- Use only metrics from this section in external materials unless a newer report supersedes it.
- NDCG@5: 0.7063
- Recall@5: 0.7531
- MRR: 0.7411
- Context precision: 0.2407
- Context recall: 0.7531

## Breakdown by Type

| Type | Count | NDCG@5 | Recall@5 | Context recall |
|---|---:|---:|---:|---:|
| comparison | 30 | 0.3288 | 0.3500 | 0.3500 |
| concept_locate | 60 | 0.8830 | 0.9500 | 0.9500 |
| fact_extract | 30 | 0.8587 | 0.9000 | 0.9000 |
| method_detail | 40 | 0.7879 | 0.8750 | 0.8750 |
| negative | 20 | n/a | n/a | n/a |
| trend_synthesis | 20 | 0.3507 | 0.3026 | 0.3026 |
