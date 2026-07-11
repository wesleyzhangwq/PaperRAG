# Paired RAG Run Comparison

- Baseline: `eval/results/rag-501/test-dense-k20-dedup5/per_question.jsonl`
- Candidate: `eval/results/rag-501/test-hybrid-a0.5-o4-k20-dedup5/per_question.jsonl`
- Questions: 200
- Bootstrap samples: 10000

| Metric | Baseline | Candidate | Delta | 95% CI | W/T/L |
|---|---:|---:|---:|---:|---:|
| ndcg_at_5 | 0.7037 | 0.7063 | +0.0027 | [-0.0250, +0.0322] | 15/150/15 |
| recall_at_5 | 0.7370 | 0.7531 | +0.0161 | [-0.0106, +0.0433] | 9/168/3 |
| mrr | 0.7385 | 0.7411 | +0.0026 | [-0.0301, +0.0376] | 11/156/13 |
| context_chunk_precision | 0.2192 | 0.2407 | +0.0216 | [+0.0013, +0.0417] | 25/148/7 |
| context_recall | 0.7370 | 0.7531 | +0.0161 | [-0.0097, +0.0428] | 9/168/3 |
| latency_s | 0.3581 | 0.3790 | +0.0209 | [+0.0003, +0.0445] | 76/0/124 |
