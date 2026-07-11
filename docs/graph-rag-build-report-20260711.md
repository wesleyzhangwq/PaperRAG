# Graph RAG Build and Evaluation Report

## Decision

The Neo4j projection and production Graph RAG path are complete and operational,
but the candidate does not pass the quality gates on the 501-paper frozen
evaluation. `GRAPH_RAG_ENABLED` therefore remains `false` by default and the
Graph result must not be presented as a retrieval-quality improvement.

## Data and Synchronization

- MySQL authority: 501 successfully ingested local papers and 54,467 chunks.
- Semantic Scholar API requests use `x-api-key`; the key is stored only in the
  Git-ignored root `.env` and is absent from tracked files.
- Local request interval: 1.3 seconds. The implementation enforces at least
  1.01 seconds globally, retries 429/5xx responses, and honors `Retry-After`.
- External projection bound: at most 200 references and 200 citations per
  local source paper. Local papers are synchronized as sources, so local-to-local
  reference coverage does not depend on importing every external citation.
- Synchronization is resumable: completed `ok` rows are skipped; each attempted
  paper is committed independently; `--force` rebuilds selected completed rows.
- Final MySQL graph status: 499 `ok`, 2 `unresolved`, 0 `failed`, 0 `pending`.
  Unresolved papers still receive local-only Paper and Category projections.

Two transient 429 failures were recovered by increasing the local interval and
retrying only failed IDs. Early projections written before external-author
bounding were rebuilt, and batch-end orphan cleanup removed stale external
Paper, Author, and Category nodes.

## Neo4j Audit

| Item | Count |
|---|---:|
| Paper nodes | 28,417 |
| Local Paper nodes | 501 |
| CITES relationships | 40,132 |
| Local-to-local CITES relationships | 1,620 |
| AUTHORED_BY relationships | 59,429 |
| IN_CATEGORY relationships | 1,450 |
| Missing local paper IDs | 0 |
| Extra local paper IDs | 0 |
| Isolated local papers | 0 |
| External-paper author edges | 0 |
| Orphan projection nodes | 0 |

The production smoke test used a comparison question, found 12 graph candidates,
completed graph expansion without fallback, and returned only local Qdrant
chunks as answer evidence.

## Score Calibration

The first production-path run directly mixed independently normalized seed and
second-pass scores. Local second-pass maxima displaced stronger global seed
chunks, reducing NDCG@5 to 0.6172. The corrected implementation stores the raw
second-pass semantic score and fuses it with the Neo4j path score using a
weighted geometric mean. This removed the severe displacement failure.

Development-only checks compared graph candidate limits 12, 32, and 64. Limit
12 preserved baseline NDCG/Recall; larger pools reduced comparison recall. The
frozen candidate therefore used limit 12 and score alpha 0.5.

## Frozen 200-Question Result

Both arms use bge-m3, hybrid alpha 0.5, oversample 4, initial top-k 20, and
paper-deduplicated top-5 context. The Graph arm additionally traverses at most
two hops, considers 12 graph candidates, and performs a filtered Qdrant second
pass.

| Metric | Tuned hybrid | +Graph | Delta |
|---|---:|---:|---:|
| Hit@5 | 0.8500 | 0.8444 | -0.0056 |
| NDCG@5 | 0.7063 | 0.7012 | -0.0051 |
| Recall@5 | 0.7531 | 0.7475 | -0.0056 |
| MRR | 0.7411 | 0.7357 | -0.0054 |
| Context chunk precision | 0.2407 | 0.1844 | -0.0563 |
| P90 total retrieval latency | 0.4178s | 0.4895s | +0.0717s |
| Graph expansion P95 | n/a | 69.56ms | within 800ms gate |
| Graph fallback rate | n/a | 0.0 | no fallback |
| Comparison Recall@5 | 0.3500 | 0.3500 | 0.0000 |
| Trend Recall@5 | 0.3026 | 0.3026 | 0.0000 |

Paired 10,000-sample bootstrap found NDCG@5 delta -0.0051 with 95% CI
[-0.0122, -0.0002], and mean latency delta +0.0837s with 95% CI
[+0.0660, +0.1026]. Graph expansion passes its latency gate, but comparison and
trend recall do not reach the required +0.05 improvement. Fixed-context answer
generation was not run because the retrieval gates had already failed.

## Follow-up Boundary

The graph is useful infrastructure for graph-aware exploration and future
datasets, but the current benchmark labels semantic relevance rather than
citation-neighborhood relevance. A future attempt should first build a
citation-specific evaluation subset and add query-aware candidate ranking;
it should not tune further on the frozen 200-question set.

## Attribution

Citation metadata is provided by the Semantic Scholar Academic Graph API.
Published results should cite Kinney et al., *The Semantic Scholar Open Data
Platform*, arXiv:2301.10140 (2023).
