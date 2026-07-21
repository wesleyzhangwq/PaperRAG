# Pure RAG Evaluation Report: test-dense-k20-dedup5

- Dataset: `questions_501_test_200.jsonl`
- Questions: 200 total, 180 positive, 20 negative

## Core Retrieval Metrics

- NDCG@5: 0.7037
- Recall@5: 0.7370
- MRR: 0.7385
- Context chunk precision: 0.2192
- Context recall: 0.7370
- Latency P90: 0.3902s
- Negative with context rate: 1.0000
- Negative max score mean: 0.5365

## Resume-safe metrics

- Use only metrics from this section in external materials unless a newer report supersedes it.
- NDCG@5: 0.7037
- Recall@5: 0.7370
- MRR: 0.7385
- Context precision: 0.2192
- Context recall: 0.7370

## Breakdown by Type

| Type | Count | NDCG@5 | Recall@5 | Context recall |
|---|---:|---:|---:|---:|
| comparison | 30 | 0.2777 | 0.3000 | 0.3000 |
| concept_locate | 60 | 0.9112 | 0.9833 | 0.9833 |
| fact_extract | 30 | 0.8210 | 0.8333 | 0.8333 |
| method_detail | 40 | 0.8031 | 0.8500 | 0.8500 |
| negative | 20 | n/a | n/a | n/a |
| trend_synthesis | 20 | 0.3450 | 0.2826 | 0.2826 |
