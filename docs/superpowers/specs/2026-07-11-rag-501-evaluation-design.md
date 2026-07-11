# 501-Paper Pure RAG Evaluation Design

## Goal

Rebuild the pure-RAG benchmark against the repaired production corpus of 501 MySQL papers and 54,467 Qdrant chunks, then measure a lexical floor, a credible dense-only baseline, and a tuned hybrid RAG candidate without involving the LangGraph agent or Neo4j.

## Evidence Boundary

- MySQL `papers` and `chunks` are the corpus source of truth.
- Qdrant alias `paperrag-active` must resolve to the same 501 papers and 54,467 chunk IDs before any run is accepted.
- The historical `questions_v3_200.jsonl` is incompatible because its 97 expected paper IDs have zero overlap with the repaired corpus.
- Historical 9,704-chunk and 30-question Agentic-RAG results remain audit artifacts only and must not be presented as current-corpus results.
- Graph RAG is excluded until Semantic Scholar citation data has been synchronized to Neo4j.

## Dataset Design

Create two paper-disjoint datasets from a deterministic, category-stratified corpus split:

| Split | Questions | Purpose |
|---|---:|---|
| Development | 50 | Parameter and context-strategy selection only |
| Frozen test | 200 | One-time final baseline/candidate comparison |

The 200-question frozen test distribution is:

| Type | Count |
|---|---:|
| `concept_locate` | 60 |
| `method_detail` | 40 |
| `fact_extract` | 30 |
| `comparison` | 30 |
| `trend_synthesis` | 20 |
| `negative` | 20 |

The 50-question development distribution is 15 / 10 / 8 / 7 / 5 / 5 in the same order.

Because 460 of 501 MySQL papers have no abstract, question generation uses deterministic representative chunks from the stored paper body. Every positive question stores `evidence_chunk_ids`; these IDs must exist in MySQL and belong to every expected paper.

Development and test paper IDs must be disjoint. Negative questions have no expected paper IDs and are used only to measure retrieval exposure; pure retrieval metrics must not be described as refusal accuracy.

## Corpus Export

Export one record per ready MySQL paper with:

```json
{
  "paper_id": "2511.16043",
  "title": "...",
  "year": 2025,
  "primary_category": "cs.AI",
  "categories": ["cs.AI"],
  "chunk_count": 120,
  "evidence_chunks": [
    {"chunk_id": "2511.16043::0", "chunk_index": 0, "page_num": 1, "text": "..."}
  ],
  "evidence_text": "..."
}
```

Representative chunks are selected deterministically and capped so prompts remain bounded. Prefer early overview chunks and chunks containing method, experiment, result, or conclusion signals. Normalize whitespace and redact secret-like strings.

## Benchmark Arms

1. **Lexical floor**: paper-level BM25 over title, category, and representative evidence text. This is deliberately coarse and is reported as a lower bound.
2. **Dense-only baseline**: production `bge-m3` embeddings with hybrid reranking disabled, fixed raw context, and no Graph expansion.
3. **Tuned hybrid candidate**: Qdrant dense oversampling followed by BM25 candidate reranking, with development-set selection across `top_k`, `alpha`, oversampling, paper deduplication, and MMR.

The resume comparison uses dense-only versus tuned hybrid. The lexical floor is retained in the technical report to prevent a straw-man baseline from carrying the main claim.

## Tuning Policy

- Tune only on the 50-question development set.
- Use coordinate sweeps rather than an unconstrained Cartesian search.
- Select by NDCG@5 first, then Recall@5, while rejecting candidates whose P90 latency exceeds 1.5 times the dense baseline or whose context precision materially collapses.
- Freeze the selected configuration before running the 200-question test.
- Do not revisit parameters after observing frozen-test results.

## Metrics

Primary metrics:

- Hit@5
- NDCG@5
- Recall@5
- MRR
- Context chunk precision and noise rate
- P50 and P90 retrieval latency

Report overall and by question type. Use a deterministic paired bootstrap over per-question rows for 95% confidence intervals on dense-only to tuned-hybrid metric deltas.

## Quality Gates

Dataset acceptance requires:

- exact row counts and type distributions;
- unique `qid` and normalized query text;
- no placeholder or fallback-template questions;
- all expected paper IDs present in MySQL and Qdrant;
- all evidence chunk IDs present in MySQL and owned by the expected paper;
- zero paper overlap between development and test positive questions;
- no title, arXiv ID, or explicit answer leakage in generated queries;
- persisted generation manifest including corpus identity, seed, model, and timestamps.

Run acceptance requires:

- every dataset row produces a result;
- manifests record corpus alias, resolved collection, vector model, and retrieval settings;
- baseline and candidate use the same frozen test rows and metric code;
- no Graph, arXiv, web, planner, reflection, or agent execution path is called.

## Resume Update

After validation:

- update `facts.md` with 501 papers, 54,467 chunks, the new dataset design, exact baseline/candidate settings, metrics, and result paths;
- mark the old 9,704-chunk and 30-question Agentic-RAG figures historical and remove them from the current master resume;
- update only `master/resume-master.html` unless the user later requests the result-oriented variant;
- make no Graph-RAG performance claim before Neo4j evaluation exists;
- render and verify the master remains one A4 page.

