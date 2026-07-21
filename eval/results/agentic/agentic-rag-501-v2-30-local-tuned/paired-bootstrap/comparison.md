# Paired RAG Run Comparison

- Baseline: `eval/results/agentic/agentic-rag-501-v2-30-local-tuned/traditional_per_question.jsonl`
- Candidate: `eval/results/agentic/agentic-rag-501-v2-30-local-tuned/agentic_per_question.jsonl`
- Questions: 30
- Bootstrap samples: 10000

| Metric | Baseline | Candidate | Delta | 95% CI | W/T/L |
|---|---:|---:|---:|---:|---:|
| mode_correct | 0.7333 | 0.9333 | +0.2000 | [+0.0333, +0.3667] | 7/22/1 |
| source_hit | 0.9259 | 0.9630 | +0.0370 | [+0.0000, +0.1111] | 1/26/0 |
| source_recall | 0.8074 | 0.8704 | +0.0630 | [+0.0000, +0.1556] | 3/24/0 |
| source_precision | 0.1472 | 0.2479 | +0.1007 | [+0.0150, +0.1922] | 20/1/6 |
| citation_support_rate | 0.9833 | 1.0000 | +0.0167 | [+0.0000, +0.0500] | 1/29/0 |
| citation_precision | 0.5969 | 0.6012 | +0.0043 | [-0.1136, +0.1191] | 8/14/5 |
| citation_expected_hit | 0.8889 | 0.9259 | +0.0370 | [+0.0000, +0.1111] | 1/26/0 |
| latency_s | 9.7493 | 98.0507 | +88.3014 | [+77.0742, +99.9522] | 0/0/30 |
| used_chunks | 4.6667 | 4.9333 | +0.2667 | [+0.0000, +0.6000] | 4/25/1 |
| step_count | 2.0000 | 18.2667 | +16.2667 | [+15.1000, +17.4008] | 30/0/0 |
| retrieval_step_count | 1.0000 | 3.7333 | +2.7333 | [+2.2667, +3.2333] | 29/1/0 |
