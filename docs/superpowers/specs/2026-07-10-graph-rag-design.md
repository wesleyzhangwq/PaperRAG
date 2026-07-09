# Cross-Paper Graph RAG Design

## Status

Approved in the 2026-07-10 design discussion. This document defines the first
Graph RAG increment only; it is not an implementation plan.

## Goal

Improve answers that require relationships across papers, especially comparison
and trend-synthesis questions. The system will keep the existing hybrid
Qdrant retrieval as the semantic evidence retriever, and add Neo4j as a
relationship-expansion layer.

The intended flow is:

```text
question
  -> hybrid retrieval finds local seed papers
  -> Neo4j expands related local papers through verified relationships
  -> hybrid retrieval fetches original chunks from the expanded paper set
  -> existing evidence, synthesis, groundedness, and citation stages run unchanged
```

The graph therefore answers “which related papers should we inspect?” It never
replaces original local chunks as the evidence used to answer or cite.

## Confirmed Product Decisions

- The first use case is cross-paper retrieval, not within-paper navigation.
- The graph contains only verifiable relationships. LLM-inferred claims such
  as “improves” or “replaces” are explicitly out of scope.
- Neo4j is required for the graph projection; MySQL remains the authority for
  `Paper` and `Chunk`, and Qdrant remains the authority for chunk vectors.
- Citation relationships will come from a batch external academic-metadata
  source. The first adapter targets the Semantic Scholar Academic Graph API,
  using existing arXiv IDs and DOI where available.
- The first user-visible surface is a concise existing Thinking Timeline step,
  not a graph browser or a global graph-explorer page.
- Graph retrieval is enabled only for complex cross-paper requests, initially
  comparison and trend-synthesis. It must silently preserve the present hybrid
  retrieval path whenever the graph is unavailable or adds no useful papers.

## Existing-System Constraints

- `backend/app/services/ingest.py` currently stores paper metadata and chunks
  in MySQL, then writes chunk vectors and metadata to Qdrant.
- Imported arXiv metadata has no references field. A citation graph cannot be
  built from the current metadata JSON alone.
- `retrieve_local` is the existing dense retrieval plus candidate BM25 fusion.
  Its original chunks, `paper_id`, and page metadata are the only valid answer
  evidence for this feature.
- The LangGraph graph already separates planning, deterministic routing,
  executor steps, evidence processing, sufficiency, synthesis, groundedness,
  citation gating, and SSE presentation. Graph RAG must fit this boundary as
  one executor tool rather than adding another graph node.
- `ChatFilter.paper_ids` is already supported by the retriever’s metadata
  filter. The second retrieval pass can therefore constrain Qdrant to graph
  candidates without creating a parallel chunk store.

## Architecture and Ownership

```text
arXiv metadata/PDF -> existing ingest -> MySQL Paper/Chunk + Qdrant chunks
                                           |
                                           v
                                  graph sync (offline, idempotent)
                                           |
external citation metadata ----------------+--> Neo4j paper graph

user query -> retrieve_local -> seed papers -> retrieve_graph -> candidate paper IDs
                                                        |
                                                        v
                              constrained retrieve_local -> original local chunks
                                                        |
                                                        v
                                      evidence -> synthesis -> citation gate
```

### MySQL and Qdrant

MySQL remains the source of truth for local corpus membership, paper metadata,
and chunk text. Qdrant remains the source of truth for semantic chunk
similarity. Neither database is written by online graph retrieval.

### Neo4j

Neo4j is a disposable, rebuildable projection for graph traversal. It stores
only minimal bibliographic node fields, relationship provenance, and relation
weights. A broken or empty Neo4j instance cannot prevent MySQL/Qdrant ingest or
chat from succeeding.

### External citation adapter

The adapter accepts canonical paper identifiers and returns cited/citing-paper
metadata. Semantic Scholar documents support `ARXIV:<id>` identifiers as well
as DOI identifiers, and provides paginated citations and references endpoints.
The adapter must be isolated behind an interface so a source change does not
affect graph storage or query code.

Reference documentation:

- <https://www.semanticscholar.org/product/api>

## Neo4j Graph Model

### Nodes

| Label | Key | Purpose |
|---|---|---|
| `Paper` | `graph_key` | A local paper or a minimal external bridge paper. |
| `Author` | `s2_author_id` | Verified author identity. |
| `Category` | arXiv category code | Existing category metadata. |

`Paper.in_corpus` is true only for a local paper with successful ingestion and
retrievable chunks. External papers are permitted only as bridge nodes and
cannot be returned to synthesis as answer sources.

For a local paper, `graph_key` is permanently `arxiv:<paper_id>`. When the
adapter resolves a Semantic Scholar identity, it sets `s2_paper_id` on that
same node. An external record with an arXiv ID matching a local paper merges
into the local node; an external-only record uses `s2:<s2_paper_id>` as its
key. This avoids parallel local/external copies of the same paper.

### Relationships

| Relationship | Direction | Provenance | Query role |
|---|---|---|---|
| `CITES` | citing paper -> cited paper | External citation metadata | Primary relationship. |
| `AUTHORED_BY` | paper -> author | Paper metadata / citation metadata | Secondary discovery signal. |
| `IN_CATEGORY` | paper -> category | arXiv metadata | Weak diversity signal. |

Every `CITES` relationship stores `source`, `source_paper_id`,
`retrieved_at`, and any available external relationship ID. No edge is created
by title similarity, LLM extraction, or inferred semantic similarity.

Constraints and indexes are required for `Paper.graph_key`, local
`Paper.paper_id`, optional `Paper.s2_paper_id`, `Author.s2_author_id`, and
`Category.code`.

## Synchronization Design

### Lifecycle

1. A full `graph_sync --all` job reads every local `Paper` whose ingestion is
   successful and creates the initial projection.
2. The job resolves a paper through the external adapter using arXiv ID first,
   then DOI when an arXiv match is unavailable.
3. It upserts the local paper, minimal external bridge papers, and verified
   relationships in one per-source-paper replacement transaction.
4. New successful ingests are marked `graph_sync_pending`. A later graph-sync
   invocation processes them after the MySQL transaction has committed.

Graph synchronization is deliberately asynchronous relative to PDF ingest.
External metadata latency, source throttling, or a Neo4j outage must not roll
back a valid local paper or vector collection.

### Status and errors

The local paper record receives graph-sync status, last-sync time, and a
sanitized error summary. Valid states are `pending`, `ok`, `unresolved`, and
`failed`.

- `unresolved`: no external-paper identity was found; author/category edges
  may still be projected from local metadata, but no `CITES` edge is invented.
- `failed`: retryable external or Neo4j failure; the next sync retries it.
- `ok`: the source paper’s previous graph edges have been replaced with the
  current verified edge set.

## Online Retrieval Design

### Activation

The planner may request `retrieve_graph` only for comparison or trend-synthesis
intent. The deterministic route node validates that graph retrieval is enabled
and reachable and ensures the action occurs after all planned local retrieval
steps. Simpler questions never pay graph-query latency.

### `retrieve_graph` executor action

Inputs:

- current local retrieval context;
- original or rewritten query;
- configured seed, hop, timeout, and candidate limits.

Algorithm:

1. Select up to four distinct `in_corpus` seed papers from the highest-scored
   local retrieval hits.
2. Run parameterized Cypher from those IDs. Traversal is limited to one or two
   hops and returns only candidate papers with `in_corpus=true`.
3. Prefer paths containing `CITES`; down-weight paths sharing `AUTHORED_BY`
   or `IN_CATEGORY`; apply a fixed hop decay; keep at most twelve candidates.
4. Re-run the existing local retriever using the original query and a
   `paper_ids` filter equal to the graph candidates.
5. Merge only the resulting local chunks into `retrieval_context`, preserving
   the existing document de-duplication behavior.

The action attaches `retrieval_score`, `graph_score`, `graph_paths`, and
`retrieval_source` metadata. Evidence processing treats semantic chunk score
as primary. Graph score can rank otherwise comparable chunks but must never
promote a semantically weak chunk solely because it has a short graph path.

### Fallback contract

`retrieve_graph` returns a warning trace and leaves context unchanged when
Neo4j is disabled, unavailable, times out, has no seed, or finds no additional
local paper. It does not call an LLM, raise a user-visible chat error, or retry
the entire agent run. The behavior then matches today’s dense-plus-BM25 path.

## User Experience and SSE

The executor publishes a normal stable `step:N` event titled “跨论文图谱扩展”.
Its safe detail contains seed paper IDs/titles, relationship-type counts,
maximum hops, graph-candidate count, added chunk count, elapsed time, and a
fallback reason where relevant. It never exposes Cypher, credentials, or
unbounded external-paper lists.

No new SSE event type, persistent user-facing graph data model, or full graph
visualization is part of this increment.

## Configuration and Deployment

Add a Neo4j service and persistent volume to `docker-compose.yml`, plus the
Neo4j Python driver to backend dependencies. The backend must not declare Neo4j
as a startup health dependency: graph connectivity is lazy and optional so base
chat startup still works without it.

New settings:

```text
NEO4J_URI
NEO4J_USER
NEO4J_PASSWORD
NEO4J_DATABASE
GRAPH_RAG_ENABLED=false
GRAPH_SEED_PAPERS=4
GRAPH_MAX_HOPS=2
GRAPH_CANDIDATE_LIMIT=12
GRAPH_QUERY_TIMEOUT_MS=800
SEMANTIC_SCHOLAR_API_KEY
```

The graph feature flag defaults to off. Secrets remain in `.env` and are not
included in logs or SSE events.

## Testing and Evaluation

### Tests

- Unit-test identifier mapping, source record normalization, edge de-duplication,
  relation weighting, hop/candidate limits, and parameterized Cypher inputs.
- Mock the Neo4j driver and external adapter in normal backend tests. No unit
  test calls a real external API or Neo4j service.
- Add optional container-backed integration tests for an idempotent backfill and
  a two-hop traversal fixture.
- Test executor merging, empty graph results, Neo4j timeouts, disabled graph
  mode, route ordering, and stable SSE trace contents.
- Retain existing citation-gate tests: a graph path alone never validates a
  final citation without a local evidence chunk.

### Evaluation gates

Extend the existing 200-question retrieval evaluation with labeled direct-citation,
two-hop-citation, and shared-author cross-paper cases. Compare identical corpus
and Qdrant settings with Graph RAG disabled and enabled.

Graph RAG may be enabled for comparison/trend traffic only when all of these
hold:

1. Combined comparison and trend-synthesis Recall@5 improves by at least 0.05
   absolute.
2. Whole-positive-set NDCG@5 decreases by no more than 0.01 absolute.
3. Fixed-context generation maintains a 1.00 citation-support rate.
4. Warm graph expansion P95 adds no more than 800 ms.

## Delivery Phases

### Phase 1: Graph foundation

Add Neo4j deployment/configuration, graph client/repository, adapter interface,
Semantic Scholar adapter, graph schema, backfill command, status tracking, and
fixture tests. Keep `GRAPH_RAG_ENABLED=false`.

### Phase 2: Retrieval integration

Add `retrieve_graph`, planner and route validation, second-pass constrained
chunk retrieval, provenance-aware evidence ordering, warning fallback, and
Timeline trace tests.

### Phase 3: Evidence-based rollout

Add graph-targeted evaluation cases, run ablations, tune bounded weights and
limits, publish the comparison report, and enable the feature flag only for
comparison/trend-synthesis after the evaluation gates pass.

## Non-Goals

- No LLM-extracted or inferred paper relationships.
- No use of external abstract/snippet content as final answer evidence.
- No graph replacement for Qdrant or MySQL.
- No all-query graph traversal.
- No global graph explorer, graph CRUD UI, or graph visualization page.
- No unrelated refactor of existing LangGraph orchestration or SSE protocol.
