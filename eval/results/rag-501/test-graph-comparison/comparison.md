# Paired RAG Run Comparison

- Baseline: `eval/results/rag-501/test-hybrid-graph-control-rerun/per_question.jsonl`
- Candidate: `eval/results/rag-501/test-graph-geo05-k20-c12/per_question.jsonl`
- Questions: 200
- Bootstrap samples: 10000

| Metric | Baseline | Candidate | Delta | 95% CI | W/T/L |
|---|---:|---:|---:|---:|---:|
| ndcg_at_5 | 0.7063 | 0.7012 | -0.0051 | [-0.0122, -0.0002] | 0/176/4 |
| recall_at_5 | 0.7531 | 0.7475 | -0.0056 | [-0.0167, +0.0000] | 0/179/1 |
| mrr | 0.7411 | 0.7357 | -0.0054 | [-0.0127, -0.0003] | 0/176/4 |
| context_chunk_precision | 0.2407 | 0.1844 | -0.0563 | [-0.0831, -0.0325] | 0/152/28 |
| context_recall | 0.7531 | 0.7475 | -0.0056 | [-0.0167, +0.0000] | 0/179/1 |
| latency_s | 0.3545 | 0.4382 | +0.0837 | [+0.0660, +0.1026] | 11/0/189 |
