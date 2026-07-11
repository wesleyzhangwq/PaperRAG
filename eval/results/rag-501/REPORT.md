# 501-Paper Pure RAG Evaluation

## Scope

- Corpus: 501 MySQL papers, 54,467 chunks; Qdrant was rebuilt from the exact MySQL chunk IDs with `BAAI/bge-m3`.
- Questions: 50-question development set and a paper-disjoint frozen 200-question test set.
- Frozen test mix: 60 concept, 40 method, 30 fact, 30 comparison, 20 trend, and 20 out-of-corpus negative questions.
- Evaluation target: local retrieval and fixed top-5 context only. LangGraph planning, routing, reflection, answer generation, Graph RAG, and external search are excluded.
- Positive labels: expected paper IDs plus evidence chunk IDs derived from exported MySQL text.
- Negative audit: zero title/abstract phrase matches across 501 papers; manual review found zero directly relevant results among dense top-3 for all 20 negatives.

The development and test paper sets are disjoint. Parameters were selected on the 50-question development set before either frozen test arm was run.

## Systems

| System | Retrieval configuration | Purpose |
|---|---|---|
| Lexical floor | Paper-level BM25 over title, abstract, categories, and representative evidence | Deliberately rough lower bound |
| Dense baseline | bge-m3 dense retrieval, `top_k=20`, paper deduplication, final context 5 | Credible comparison baseline |
| Tuned hybrid | bge-m3 fetch 80, normalized BM25/vector fusion with vector weight `0.5`, retain 20, paper deduplication, final context 5 | Development-selected candidate |

## Development Selection

| Configuration | NDCG@5 | Recall@5 | MRR | Context precision | P90 |
|---|---:|---:|---:|---:|---:|
| Lexical floor | 0.2523 | 0.4526 | 0.1956 | 0.1022 | 0.216s |
| Dense baseline | 0.7536 | 0.7967 | 0.8148 | 0.2393 | 0.427s |
| Hybrid `alpha=.3`, oversample 2.5 | 0.7389 | 0.7930 | 0.8000 | 0.2515 | 1.018s |
| Hybrid `alpha=.5`, oversample 2.5 | 0.7592 | 0.7930 | 0.8315 | 0.2607 | 0.504s |
| Hybrid `alpha=.5`, oversample 4 | **0.7624** | 0.7930 | **0.8370** | **0.2730** | 0.413s |

The last configuration was frozen because it had the best development NDCG@5 and MRR, materially higher context precision, and no observed latency penalty relative to dense-only. The development bootstrap intervals crossed zero, so this selection was treated as a candidate, not a proven improvement.

## Frozen Test Results

| Metric | Dense baseline | Tuned hybrid | Delta |
|---|---:|---:|---:|
| Hit@5 | 0.8222 | **0.8500** | +0.0278 |
| NDCG@5 | 0.7037 | **0.7063** | +0.0027 (+0.38%) |
| Recall@5 | 0.7370 | **0.7531** | +0.0161 (+2.19%) |
| MRR | 0.7385 | **0.7411** | +0.0026 (+0.35%) |
| Context chunk precision | 0.2192 | **0.2407** | +0.0216 (+9.84%) |
| P90 retrieval latency | **0.390s** | 0.418s | +0.028s (+7.1%) |

Paired 10,000-sample bootstrap over 180 positive questions:

| Metric delta | 95% CI | Wins / ties / losses | Interpretation |
|---|---:|---:|---|
| NDCG@5 +0.0027 | [-0.0250, +0.0322] | 15 / 150 / 15 | Ranking is effectively flat |
| Recall@5 +0.0161 | [-0.0106, +0.0433] | 9 / 168 / 3 | Positive point estimate, uncertain |
| Context precision +0.0216 | **[+0.0013, +0.0417]** | 25 / 148 / 7 | Reliable reduction in irrelevant context |
| Latency +0.0209s | [+0.0003, +0.0445] | 76 / 0 / 124 | Candidate is measurably slower on mean latency |

Fused hybrid scores are normalized within each candidate set, so their negative-query score magnitudes are not directly comparable with raw dense cosine scores. Negative questions are retained for later generation/refusal evaluation and are not used in the positive retrieval averages.

## By Question Type

| Type | NDCG@5 dense -> hybrid | Recall@5 dense -> hybrid | Context precision dense -> hybrid |
|---|---:|---:|---:|
| Concept | 0.9112 -> 0.8830 | 0.9833 -> 0.9500 | 0.2614 -> 0.2550 |
| Method | 0.8031 -> 0.7879 | 0.8500 -> 0.8750 | 0.2204 -> 0.2737 |
| Fact | 0.8210 -> 0.8587 | 0.8333 -> 0.9000 | 0.2050 -> 0.2450 |
| Comparison | 0.2777 -> 0.3288 | 0.3000 -> 0.3500 | 0.1200 -> 0.1444 |
| Trend | 0.3450 -> 0.3507 | 0.2826 -> 0.3026 | 0.2600 -> 0.2700 |

Hybrid fusion helps fact extraction and multi-paper comparison most, but slightly harms already-strong concept lookup. A production follow-up should route simple concept queries to dense-only and use hybrid expansion for detail or multi-paper queries.

## Resume-Safe Facts

- Built an evidence-backed pure-RAG benchmark over 501 papers and 54,467 chunks, with 50 development and 200 paper-disjoint frozen test questions.
- Tuned candidate expansion, BM25/vector fusion, and paper-level context deduplication against a credible bge-m3 dense-only baseline.
- On the frozen test set, improved Hit@5 from 82.22% to 85.00%, Recall@5 from 73.70% to 75.31%, and context precision from 21.92% to 24.07% (+9.84% relative), while P90 retrieval latency rose from 0.390s to 0.418s.
- Do not claim Graph RAG gains until Neo4j citation edges are populated and a separately frozen Graph RAG run passes its acceptance gates.
