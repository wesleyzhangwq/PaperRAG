# Cross-Paper Graph RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional Neo4j-backed Graph RAG path that expands semantically retrieved local papers through verified cross-paper relationships, then fetches original local chunks as the only answer evidence.

**Architecture:** MySQL remains authoritative for papers and chunks, Qdrant remains authoritative for chunk retrieval, and Neo4j is a rebuildable projection. A separate synchronization command imports verified citation metadata into Neo4j; online `retrieve_graph` uses local Qdrant hits as seeds, traverses at most two graph hops, and constrains a second Qdrant retrieval to returned local paper IDs. The existing evidence, citation, SSE, and fallback contracts remain in force.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy/MySQL 8, Qdrant, LangGraph, Neo4j Python driver, Neo4j 5 Community, `requests`, pytest, Docker Compose.

## Global Constraints

- Before implementation, create an isolated worktree; this checkout already contains user-owned uncommitted changes. Stage only files named in each task.
- Add `neo4j>=5,<6` to `backend/requirements.txt`; do not add an ORM, a background-worker framework, or an LLM graph-extraction dependency.
- `GRAPH_RAG_ENABLED` defaults to `false`; base chat startup, normal ingestion, and all non-graph queries must work with Neo4j absent.
- MySQL `Paper` and `Chunk` plus Qdrant chunks are the only answer evidence. External citation metadata and Neo4j paths must never become synthesis context or citations.
- `CITES` is the primary edge. `AUTHORED_BY` and `IN_CATEGORY` are weaker expansion signals. Do not create edges from title similarity or LLM inference.
- The online graph action is limited to four seed papers, two hops, twelve candidate local papers, and an 800 ms graph query timeout.
- Tests mock Neo4j, Semantic Scholar, Qdrant, and LLM calls. Only the explicitly marked Docker smoke test uses a local Neo4j container.
- Follow the existing pure-node convention: only executor dispatches the graph action; no new LangGraph node or SSE event type is added.

---

## File Structure

| Path | Responsibility |
|---|---|
| `backend/app/core/config.py` | Typed Neo4j, graph-feature, and citation-source settings. |
| `backend/app/models/paper.py` | Graph synchronization status fields on local papers. |
| `backend/app/db/mysql.py` | Idempotent migration of graph status columns for existing MySQL databases. |
| `backend/app/db/neo4j.py` | Lazy Neo4j repository, schema setup, provenance-aware graph upsert, and bounded traversal. |
| `backend/app/services/semantic_scholar.py` | External citation-source adapter and normalized response dataclasses. |
| `backend/app/services/graph_sync.py` | MySQL-to-Neo4j projection orchestration and sync status transitions. |
| `backend/scripts/sync_graph.py` | Explicit `--all` / `--paper-id` operator entry point. |
| `backend/app/tools/retrieve_graph.py` | Tool-facing graph candidate expansion boundary. |
| `backend/app/services/graph_retriever.py` | Seed selection, graph score calculation, and second-pass Qdrant retrieval. |
| `backend/app/agent/nodes/executor.py` | `retrieve_graph` dispatch, provenance metadata, safe warning fallback, and trace detail. |
| `backend/app/agent/nodes/evidence.py` | Preserve `retrieval_score` as the primary evidence-ranking value. |
| `backend/app/agent/nodes/planner.py` | Parse `retrieve_graph` as an executable retrieval action. |
| `backend/app/agent/nodes/route.py` | Deterministically inject, order, or remove graph retrieval by intent and feature flag. |
| `backend/app/agent/prompts/planner.py` | Tell the planner when graph expansion is appropriate. |
| `backend/app/agent/stages.py` | Chinese action label for the stable executor step. |
| `eval/run_rag_eval.py` | `service_graph` ablation runner that calls the same retrieval service as production. |
| `eval/rag_metrics.py` | Graph-targeted comparison/trend summary and graph-step latency percentile. |
| `eval/scripts/build_graph_eval_manifest.py` | Deterministic candidate manifest for direct-citation, two-hop, and shared-author case review. |
| `eval/datasets/questions_v3_graph.jsonl` | Reviewed graph-targeted evaluation questions with expected local paper IDs. |
| `docker-compose.yml`, `.env.example`, `README.md`, `eval/README.md` | Deployment, configuration, and operator documentation. |

## Shared Interfaces

Implement these interfaces before code that consumes them:

```python
# backend/app/services/semantic_scholar.py
@dataclass(frozen=True)
class RemotePaper:
    s2_paper_id: str
    arxiv_id: str | None
    doi: str | None
    title: str
    year: int | None
    authors: tuple[tuple[str, str], ...]

@dataclass(frozen=True)
class CitationSnapshot:
    source: RemotePaper
    references: tuple[RemotePaper, ...]
    citations: tuple[RemotePaper, ...]

class CitationSourceUnavailable(RuntimeError):
    pass

class CitationSourceNotFound(RuntimeError):
    pass

def fetch_citation_snapshot(*, arxiv_id: str, doi: str | None) -> CitationSnapshot:
    pass

# backend/app/db/neo4j.py
@dataclass(frozen=True)
class GraphCandidate:
    paper_id: str
    graph_score: float
    paths: tuple[dict[str, object], ...]

class GraphUnavailable(RuntimeError):
    pass

class Neo4jGraphRepository:
    def ensure_schema(self) -> None:
        pass

    def replace_source_projection(
        self,
        *,
        source_paper_id: str,
        papers: list[dict[str, object]],
        citation_edges: list[dict[str, str]],
        authors: list[dict[str, str]],
        categories: list[str],
    ) -> None:
        pass

    def expand_local_papers(
        self,
        *,
        seed_paper_ids: list[str],
        seed_scores: dict[str, float],
        max_hops: int,
        limit: int,
    ) -> list[GraphCandidate]:
        pass

def get_neo4j_repository() -> Neo4jGraphRepository:
    pass

# backend/app/services/graph_retriever.py
@dataclass(frozen=True)
class GraphRetrievalReport:
    seed_paper_ids: tuple[str, ...]
    candidates: tuple[GraphCandidate, ...]
    added_chunks: int
    fallback_reason: str | None
    graph_elapsed_ms: float

def retrieve_graph_context(
    *,
    query: str,
    existing_context: list[Document],
    top_k: int,
) -> tuple[list[Document], GraphRetrievalReport]:
    pass
```

`pass` above is interface notation only. Each implementation task below replaces it with executable code and tests it before committing.

### Task 1: Add optional graph infrastructure, settings, and local sync state

**Files:**
- Modify: `backend/requirements.txt`
- Modify: `backend/app/core/config.py`
- Modify: `backend/app/models/paper.py`
- Modify: `backend/app/db/mysql.py`
- Modify: `docker-compose.yml`
- Modify: `.env.example`
- Create: `backend/tests/test_graph_settings.py`
- Create: `backend/tests/test_graph_migration.py`

**Interfaces:**
- Produces `Settings.graph_rag_enabled`, `Settings.neo4j_uri`, `Settings.graph_max_hops`, and the other graph settings used by all later tasks.
- Produces `Paper.graph_sync_status`, `Paper.graph_synced_at`, and `Paper.graph_sync_error` for Task 3.

- [ ] **Step 1: Write failing settings and migration tests**

```python
# backend/tests/test_graph_settings.py
from app.core.config import Settings


def test_graph_settings_default_to_a_disabled_bounded_feature() -> None:
    settings = Settings(_env_file=None)

    assert settings.graph_rag_enabled is False
    assert settings.graph_seed_papers == 4
    assert settings.graph_max_hops == 2
    assert settings.graph_candidate_limit == 12
    assert settings.graph_query_timeout_ms == 800
    assert settings.neo4j_uri is None


def test_graph_settings_accept_explicit_connection_values(monkeypatch) -> None:
    monkeypatch.setenv("GRAPH_RAG_ENABLED", "true")
    monkeypatch.setenv("NEO4J_URI", "bolt://neo4j:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "local-graph-password")

    settings = Settings(_env_file=None)

    assert settings.graph_rag_enabled is True
    assert settings.neo4j_uri == "bolt://neo4j:7687"
    assert settings.neo4j_user == "neo4j"
```

```python
# backend/tests/test_graph_migration.py
from unittest.mock import MagicMock, patch

from app.db import mysql


@patch("app.db.mysql.inspect")
def test_migrate_papers_adds_only_missing_graph_columns(mock_inspect) -> None:
    inspector = MagicMock()
    inspector.get_table_names.return_value = ["papers"]
    inspector.get_columns.return_value = [{"name": "paper_id"}]
    mock_inspect.return_value = inspector
    connection = MagicMock()

    with patch.object(mysql.engine, "begin") as begin:
        begin.return_value.__enter__.return_value = connection
        mysql._migrate_papers()

    statements = "\n".join(str(call.args[0]) for call in connection.execute.call_args_list)
    assert "graph_sync_status" in statements
    assert "graph_synced_at" in statements
    assert "graph_sync_error" in statements
    assert "ix_papers_graph_sync_status" in statements
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
cd backend && .venv/bin/python -m pytest tests/test_graph_settings.py tests/test_graph_migration.py -q
```

Expected: FAIL because `Settings` and `Paper` have no graph fields and `_migrate_papers` is undefined.

- [ ] **Step 3: Add the bounded configuration, migration, model, dependency, and Compose service**

Add the following exact settings to `Settings` after the Qdrant section:

```python
    # --- Graph RAG / Neo4j ---
    neo4j_uri: Optional[str] = None
    neo4j_user: str = "neo4j"
    neo4j_password: Optional[str] = None
    neo4j_database: str = "neo4j"
    graph_rag_enabled: bool = False
    graph_seed_papers: int = 4
    graph_max_hops: int = 2
    graph_candidate_limit: int = 12
    graph_query_timeout_ms: int = 800
    semantic_scholar_api_key: Optional[str] = None
```

Add the following fields to `Paper` directly after `ingest_error`:

```python
    graph_sync_status: Mapped[str] = mapped_column(
        String(16), default="pending", index=True, nullable=False
    )
    graph_synced_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    graph_sync_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
```

Add this migration and call it from `init_db()` after `_migrate_chat_history()`:

```python
def _migrate_papers() -> None:
    try:
        inspector = inspect(engine)
        if "papers" not in inspector.get_table_names():
            return
        existing = {column["name"] for column in inspector.get_columns("papers")}
        statements = {
            "graph_sync_status": "ALTER TABLE papers ADD COLUMN graph_sync_status VARCHAR(16) NOT NULL DEFAULT 'pending'",
            "graph_synced_at": "ALTER TABLE papers ADD COLUMN graph_synced_at DATETIME NULL",
            "graph_sync_error": "ALTER TABLE papers ADD COLUMN graph_sync_error TEXT NULL",
        }
        with engine.begin() as connection:
            for column, statement in statements.items():
                if column not in existing:
                    connection.execute(text(statement))
            if "graph_sync_status" not in existing:
                connection.execute(text(
                    "CREATE INDEX ix_papers_graph_sync_status ON papers (graph_sync_status)"
                ))
    except Exception as exc:
        log.warning("papers graph migration skipped: %s", exc)
```

Append `neo4j>=5,<6` under a new `# ===== Graph DB =====` heading in
`backend/requirements.txt`. Add a `neo4j` Compose service with ports `7474`
and `7687`, a `neo4j_data` volume, and
`NEO4J_AUTH=neo4j/${NEO4J_PASSWORD:-local-graph-password}`.
Pass Neo4j and graph settings into the backend service, but do not add Neo4j to
backend `depends_on`.

Add these `.env.example` entries:

```dotenv
# ----- Graph RAG / Neo4j -----
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=local-graph-password
NEO4J_DATABASE=neo4j
GRAPH_RAG_ENABLED=false
GRAPH_SEED_PAPERS=4
GRAPH_MAX_HOPS=2
GRAPH_CANDIDATE_LIMIT=12
GRAPH_QUERY_TIMEOUT_MS=800
SEMANTIC_SCHOLAR_API_KEY=
```

- [ ] **Step 4: Run focused tests and validate Compose syntax**

Run:

```bash
cd backend && .venv/bin/python -m pytest tests/test_graph_settings.py tests/test_graph_migration.py -q
cd .. && docker compose config >/dev/null
```

Expected: both tests PASS and `docker compose config` exits 0 without printing secrets.

- [ ] **Step 5: Commit the foundation**

```bash
git add backend/requirements.txt backend/app/core/config.py backend/app/models/paper.py backend/app/db/mysql.py docker-compose.yml .env.example backend/tests/test_graph_settings.py backend/tests/test_graph_migration.py
git commit -m "feat: add graph rag infrastructure settings"
```

### Task 2: Implement the external citation adapter and lazy Neo4j repository

**Files:**
- Create: `backend/app/services/semantic_scholar.py`
- Create: `backend/app/db/neo4j.py`
- Create: `backend/tests/test_semantic_scholar.py`
- Create: `backend/tests/test_neo4j_repository.py`

**Interfaces:**
- Consumes: Task 1 graph settings.
- Produces: `CitationSnapshot`, `CitationSourceUnavailable`, `GraphCandidate`, `GraphUnavailable`, and `Neo4jGraphRepository` for Tasks 3 and 4.

- [ ] **Step 1: Write failing adapter and repository tests**

```python
# backend/tests/test_semantic_scholar.py
from unittest.mock import MagicMock, patch

from app.services.semantic_scholar import fetch_citation_snapshot


@patch("app.services.semantic_scholar.requests.get")
def test_fetch_snapshot_maps_arxiv_identifiers_and_both_citation_directions(mock_get) -> None:
    mock_get.side_effect = [
        MagicMock(status_code=200, json=lambda: {"paperId": "s2-source", "externalIds": {"ArXiv": "2401.00001"}, "title": "Source", "year": 2024, "authors": []}),
        MagicMock(status_code=200, json=lambda: {"data": [{"citedPaper": {"paperId": "s2-ref", "externalIds": {"ArXiv": "2301.00001"}, "title": "Reference", "year": 2023, "authors": []}}], "next": None}),
        MagicMock(status_code=200, json=lambda: {"data": [{"citingPaper": {"paperId": "s2-citing", "externalIds": {"ArXiv": "2501.00001"}, "title": "Citing", "year": 2025, "authors": []}}], "next": None}),
    ]

    snapshot = fetch_citation_snapshot(arxiv_id="2401.00001", doi=None)

    assert snapshot.source.s2_paper_id == "s2-source"
    assert [paper.arxiv_id for paper in snapshot.references] == ["2301.00001"]
    assert [paper.arxiv_id for paper in snapshot.citations] == ["2501.00001"]
```

```python
# backend/tests/test_neo4j_repository.py
from unittest.mock import MagicMock

from app.db.neo4j import Neo4jGraphRepository


def test_expand_local_papers_returns_only_distinct_local_candidates() -> None:
    driver = MagicMock()
    session = driver.session.return_value.__enter__.return_value
    session.run.return_value = [
        {"paper_id": "B", "seed_paper_id": "A", "hops": 1, "relations": ["CITES"]},
        {"paper_id": "B", "seed_paper_id": "A", "hops": 2, "relations": ["CITES", "IN_CATEGORY"]},
    ]
    repo = Neo4jGraphRepository(driver=driver, database="neo4j", timeout_ms=800)

    candidates = repo.expand_local_papers(
        seed_paper_ids=["A"], seed_scores={"A": 0.9}, max_hops=2, limit=12
    )

    assert [candidate.paper_id for candidate in candidates] == ["B"]
    assert candidates[0].paths[0]["relations"] == ["CITES"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
cd backend && .venv/bin/python -m pytest tests/test_semantic_scholar.py tests/test_neo4j_repository.py -q
```

Expected: FAIL because the adapter and repository modules do not exist.

- [ ] **Step 3: Implement source normalization and a lazy repository**

In `semantic_scholar.py`, use three paginated `requests.get` calls per
source-paper identity: one paper-detail request, one references request, and
one citations request. Use `ARXIV:<arxiv_id>` first and retry once with
`DOI:<doi>` only after an arXiv 404. Include the optional `x-api-key` header
only when configured. Convert every response into this complete immutable
model:

```python
@dataclass(frozen=True)
class RemotePaper:
    s2_paper_id: str
    arxiv_id: str | None
    doi: str | None
    title: str
    year: int | None
    authors: tuple[tuple[str, str], ...]


def _remote_paper(raw: dict) -> RemotePaper:
    external = raw.get("externalIds") or {}
    authors = tuple(
        (str(author.get("authorId") or ""), str(author.get("name") or ""))
        for author in raw.get("authors") or []
        if author.get("authorId") and author.get("name")
    )
    return RemotePaper(
        s2_paper_id=str(raw.get("paperId") or ""),
        arxiv_id=str(external.get("ArXiv") or "") or None,
        doi=str(external.get("DOI") or "") or None,
        title=str(raw.get("title") or ""),
        year=int(raw["year"]) if raw.get("year") is not None else None,
        authors=authors,
    )
```

In `neo4j.py`, construct the driver only in `get_neo4j_repository()` and raise
`GraphUnavailable("Neo4j is not configured")` when URI or password is absent.
Use a 2-hop constant Cypher pattern and a parameterized hop filter:

```cypher
MATCH (seed:Paper {in_corpus: true})
WHERE seed.paper_id IN $seed_paper_ids
MATCH path=(seed)-[*1..2]-(candidate:Paper {in_corpus: true})
WHERE candidate.paper_id <> seed.paper_id
  AND length(path) <= $max_hops
  AND all(rel IN relationships(path) WHERE type(rel) IN $relationship_types)
RETURN DISTINCT candidate.paper_id AS paper_id,
       seed.paper_id AS seed_paper_id,
       length(path) AS hops,
       [rel IN relationships(path) | type(rel)] AS relations
LIMIT $raw_limit
```

Calculate candidate score in Python as
`seed_score * relation_weight * (0.65 ** (hops - 1))`, with weights `1.0` for
`CITES`, `0.45` for `AUTHORED_BY`, and `0.20` for `IN_CATEGORY`. Preserve at
most three highest-scoring path summaries per candidate and return at most the
requested candidate limit.

- [ ] **Step 4: Run tests and the Neo4j-only smoke check**

Run:

```bash
cd backend && .venv/bin/python -m pytest tests/test_semantic_scholar.py tests/test_neo4j_repository.py -q
cd .. && docker compose up -d neo4j && docker compose exec neo4j cypher-shell -u neo4j -p "$NEO4J_PASSWORD" "RETURN 1 AS ok"
```

Expected: focused tests PASS and Cypher output contains `1`.

- [ ] **Step 5: Commit the source and repository boundary**

```bash
git add backend/app/services/semantic_scholar.py backend/app/db/neo4j.py backend/tests/test_semantic_scholar.py backend/tests/test_neo4j_repository.py
git commit -m "feat: add citation graph repository"
```

### Task 3: Build an idempotent graph synchronization command

**Files:**
- Create: `backend/app/services/graph_sync.py`
- Create: `backend/scripts/sync_graph.py`
- Modify: `backend/app/services/ingest.py`
- Create: `backend/tests/test_graph_sync.py`

**Interfaces:**
- Consumes: `Paper` graph status from Task 1 and the adapter/repository from Task 2.
- Produces: `sync_paper(db, paper) -> str` and `run_graph_sync(paper_ids: list[str] | None) -> dict` for operators.

- [ ] **Step 1: Write failing synchronization tests**

```python
# backend/tests/test_graph_sync.py
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.services.graph_sync import sync_paper


@patch("app.services.graph_sync.fetch_citation_snapshot")
@patch("app.services.graph_sync.get_neo4j_repository")
def test_sync_paper_projects_local_and_external_nodes_then_marks_ok(mock_repo, mock_fetch) -> None:
    paper = SimpleNamespace(
        paper_id="2401.00001", doi=None, title="Local", year=2024,
        authors=["Ada"], categories=["cs.CL"], ingest_status="ok",
        num_chunks=3, graph_sync_status="pending", graph_sync_error=None,
    )
    mock_fetch.return_value = SimpleNamespace(
        source=SimpleNamespace(s2_paper_id="s2-a", arxiv_id="2401.00001", doi=None, title="Local", year=2024, authors=(("author-a", "Ada"),)),
        references=(SimpleNamespace(s2_paper_id="s2-b", arxiv_id=None, doi=None, title="Bridge", year=2020, authors=()),),
        citations=(),
    )
    db = MagicMock()

    status = sync_paper(db, paper, local_papers={"2401.00001": paper})

    assert status == "ok"
    assert paper.graph_sync_status == "ok"
    assert paper.graph_sync_error is None
    mock_repo.return_value.replace_source_projection.assert_called_once()


@patch("app.services.graph_sync.fetch_citation_snapshot", side_effect=RuntimeError("rate limited"))
def test_sync_paper_marks_retryable_failure_without_raising(mock_fetch) -> None:
    paper = SimpleNamespace(paper_id="2401.00001", doi=None, ingest_status="ok", num_chunks=1)
    db = MagicMock()

    status = sync_paper(db, paper, local_papers={"2401.00001": paper})

    assert status == "failed"
    assert paper.graph_sync_status == "failed"
    assert "rate limited" in paper.graph_sync_error
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
cd backend && .venv/bin/python -m pytest tests/test_graph_sync.py -q
```

Expected: FAIL because `graph_sync.py` and `sync_paper` do not exist.

- [ ] **Step 3: Implement projection mapping and the CLI**

Implement `sync_paper` with these status transitions:

```python
def _mark(paper: Paper, status: str, error: str | None = None) -> str:
    paper.graph_sync_status = status
    paper.graph_sync_error = error
    paper.graph_synced_at = datetime.utcnow() if status == "ok" else None
    return status


def sync_paper(db: Session, paper: Paper, *, local_papers: dict[str, Paper]) -> str:
    if paper.ingest_status != "ok" or paper.num_chunks < 1:
        return _mark(paper, "pending")
    try:
        snapshot = fetch_citation_snapshot(arxiv_id=paper.paper_id, doi=paper.doi)
        payload = build_projection_payload(paper, snapshot, local_papers)
        get_neo4j_repository().replace_source_projection(
            source_paper_id=paper.paper_id,
            papers=payload["papers"],
            citation_edges=payload["citation_edges"],
            authors=payload["authors"],
            categories=payload["categories"],
        )
    except CitationSourceNotFound:
        return _mark(paper, "unresolved")
    except Exception as exc:
        return _mark(paper, "failed", f"{type(exc).__name__}: {exc}")
    return _mark(paper, "ok")
```

`build_projection_payload` must assign `graph_key="arxiv:<paper_id>"` for
any reference/citation whose arXiv ID belongs to `local_papers`, otherwise
`graph_key="s2:<s2_paper_id>"`. It must write both directions discovered by
the source: `source -> reference` and `citing -> source`. Each relationship
has `source="semantic_scholar"` and `source_paper_id=<local source paper id>`.

In `Neo4jGraphRepository.replace_source_projection`, first delete only
relationships tagged with the current `source_paper_id`, then `MERGE` nodes by
`graph_key`, set `in_corpus`, and create `CITES`, `AUTHORED_BY`, and
`IN_CATEGORY` edges. Never delete a node or an edge owned by another local
source paper.

Implement this CLI argument contract:

```python
parser.add_argument("--all", action="store_true")
parser.add_argument("--paper-id", action="append", default=[])
```

`--all` processes every successful local paper; one or more `--paper-id`
values process exactly those local IDs; neither option exits with code 2. The
script prints one JSON stats object containing `ok`, `unresolved`, `failed`,
and `total`.

After a successful `_ingest_one`, set `paper.graph_sync_status = "pending"`,
clear `graph_sync_error`, and leave actual synchronization to the explicit CLI.

- [ ] **Step 4: Run unit tests and an idempotence smoke check**

Run:

```bash
cd backend && .venv/bin/python -m pytest tests/test_graph_sync.py -q
cd .. && docker compose exec backend python scripts/sync_graph.py --all
docker compose exec backend python scripts/sync_graph.py --all
docker compose exec neo4j cypher-shell -u neo4j -p "$NEO4J_PASSWORD" "MATCH (p:Paper) RETURN count(p) AS papers"
```

Expected: unit tests PASS; both sync invocations output JSON stats; the second
run does not increase node count or create duplicate provenance edges.

- [ ] **Step 5: Commit graph synchronization**

```bash
git add backend/app/services/graph_sync.py backend/scripts/sync_graph.py backend/app/services/ingest.py backend/tests/test_graph_sync.py
git commit -m "feat: sync local papers into citation graph"
```

### Task 4: Add provenance-preserving graph expansion and constrained chunk retrieval

**Files:**
- Create: `backend/app/tools/retrieve_graph.py`
- Create: `backend/app/services/graph_retriever.py`
- Modify: `backend/app/agent/nodes/executor.py`
- Modify: `backend/app/agent/nodes/evidence.py`
- Create: `backend/tests/agent/test_graph_retrieval.py`

**Interfaces:**
- Consumes: `Neo4jGraphRepository.expand_local_papers` from Task 2 and existing `retrieve(query, flt, top_k)`.
- Produces: executor action `retrieve_graph` and document metadata keys `retrieval_score`, `graph_score`, `graph_paths`, and `retrieval_source` for Task 5 and existing evidence processing.

- [ ] **Step 1: Write failing graph-retrieval executor tests**

```python
# backend/tests/agent/test_graph_retrieval.py
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from app.agent.nodes.executor import executor_node
from app.agent.state import StepSpec


def _state() -> dict:
    return {
        "messages": [HumanMessage(content="compare A and B")],
        "intent": {"type": "comparison", "complexity": "high"},
        "plan": [StepSpec(action="retrieve_graph", params={"query": "compare A and B", "top_k": 3}, reason="cross-paper expansion")],
        "plan_step_index": 0,
        "retrieval_context": [Document(page_content="seed", metadata={"paper_id": "A", "retrieval_score": 0.9, "retrieval_source": "local"})],
        "step_traces": [],
        "reflection_count": 0,
    }


@patch("app.agent.nodes.executor.retrieve_graph_context")
def test_executor_graph_action_adds_only_second_pass_local_chunks(mock_retrieve_graph) -> None:
    expanded = Document(page_content="local evidence", metadata={"paper_id": "B", "retrieval_score": 0.8, "graph_score": 0.9, "retrieval_source": "graph_local"})
    mock_retrieve_graph.return_value = ([expanded], MagicMock(seed_paper_ids=("A",), candidates=(MagicMock(paper_id="B"),), added_chunks=1, fallback_reason=None, graph_elapsed_ms=12.5))

    result = executor_node(_state(), db=MagicMock())

    assert [doc.metadata["paper_id"] for doc in result["retrieval_context"]] == ["A", "B"]
    assert result["step_traces"][-1]["detail"]["added"] == 1


@patch("app.agent.nodes.executor.retrieve_graph_context")
def test_executor_graph_action_warns_and_keeps_context_on_fallback(mock_retrieve_graph) -> None:
    state = _state()
    mock_retrieve_graph.return_value = (state["retrieval_context"], MagicMock(seed_paper_ids=("A",), candidates=(), added_chunks=0, fallback_reason="neo4j_unavailable", graph_elapsed_ms=0.0))

    result = executor_node(state, db=MagicMock())

    assert len(result["retrieval_context"]) == 1
    assert result["step_traces"][-1]["detail"]["fallback_reason"] == "neo4j_unavailable"


def test_evidence_prefers_retrieval_score_over_legacy_score() -> None:
    from app.agent.nodes.evidence import _score

    doc = Document(page_content="evidence", metadata={"score": 0.1, "retrieval_score": 0.8})

    assert _score(doc) == 0.8
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
cd backend && .venv/bin/python -m pytest tests/agent/test_graph_retrieval.py -q
```

Expected: FAIL because `retrieve_graph_context` and the `retrieve_graph` executor branch do not exist.

- [ ] **Step 3: Implement the graph retrieval service and executor branch**

Use a non-LangChain wrapper in `backend/app/tools/retrieve_graph.py`:

```python
def retrieve_graph_candidates(
    *,
    seed_paper_ids: list[str],
    seed_scores: dict[str, float],
    max_hops: int,
    limit: int,
) -> list[GraphCandidate]:
    return get_neo4j_repository().expand_local_papers(
        seed_paper_ids=seed_paper_ids,
        seed_scores=seed_scores,
        max_hops=max_hops,
        limit=limit,
    )
```

Implement `retrieve_graph_context` with these exact rules:

```python
seed_docs = [
    doc for doc in existing_context
    if (doc.metadata or {}).get("retrieval_source") in {"local", "graph_local"}
    and (doc.metadata or {}).get("paper_id")
]
seed_by_paper = first_highest_scored_document_per_paper(seed_docs)
seed_by_paper = dict(
    sorted(seed_by_paper.items(), key=lambda item: item[1], reverse=True)[:settings.graph_seed_papers]
)
candidates = retrieve_graph_candidates(
    seed_paper_ids=list(seed_by_paper),
    seed_scores=seed_by_paper,
    max_hops=min(2, max(1, settings.graph_max_hops)),
    limit=settings.graph_candidate_limit,
)
docs_scores = retrieve(
    query,
    flt=ChatFilter(paper_ids=[candidate.paper_id for candidate in candidates]),
    top_k=top_k,
)
```

Add this helper to `executor.py` and pass all local retrieval results through it
before `_append_unique_documents`:

```python
def _with_retrieval_metadata(
    doc: Document,
    score: float,
    *,
    source: str,
    graph_score: float | None = None,
    graph_paths: list[dict[str, object]] | None = None,
) -> Document:
    metadata = {
        **(doc.metadata or {}),
        "retrieval_score": float(score),
        "retrieval_source": source,
    }
    if graph_score is not None:
        metadata["graph_score"] = float(graph_score)
    if graph_paths:
        metadata["graph_paths"] = graph_paths
    return Document(page_content=doc.page_content, metadata=metadata)
```

Return the original context unchanged with `fallback_reason` set to one of
`graph_disabled`, `no_local_seeds`, `neo4j_unavailable`, or `no_new_local_candidates`
when a graph operation cannot add evidence. Decorate every initial local result
with `retrieval_score=<float score>` and `retrieval_source="local"`; decorate
second-pass chunks with `retrieval_score`, `graph_score`, `graph_paths`, and
`retrieval_source="graph_local"`. Create new `Document` instances instead of
mutating metadata returned by the retriever cache.

Add this executor branch before `retrieve_arxiv`:

```python
elif action == "retrieve_graph":
    graph_context, report = retrieve_graph_context(
        query=str(params.get("query") or fallback_query),
        existing_context=new_context,
        top_k=int(params.get("top_k") or get_settings().retrieval_k),
    )
    new_context, added_count = _append_unique_documents(new_context, graph_context)
    output_summary = (
        f"graph: {len(report.candidates)} candidate papers, {added_count} chunks added"
        if report.fallback_reason is None
        else f"graph fallback: {report.fallback_reason}"
    )
    output_detail = {
        "seed_paper_ids": list(report.seed_paper_ids),
        "candidate_paper_ids": [candidate.paper_id for candidate in report.candidates],
        "added": added_count,
        "fallback_reason": report.fallback_reason,
    }
```

Update `_score` in `evidence.py` to read `retrieval_score` before legacy
`score`, preserving the existing order for unscored arXiv/web documents:

```python
raw = (doc.metadata or {}).get("retrieval_score", (doc.metadata or {}).get("score"))
```

Make the final executor stage event a warning for graph fallback as well as
errors by computing its status this way:

```python
stage_status = "warning" if (
    output_detail.get("error")
    or output_detail.get("fallback_reason")
    or output_summary.startswith("unknown")
) else "done"
```

- [ ] **Step 4: Run focused graph and regression tests**

Run:

```bash
cd backend && .venv/bin/python -m pytest tests/agent/test_graph_retrieval.py tests/agent/test_executor.py tests/agent/test_pipeline_nodes.py -q
```

Expected: PASS; graph fallback leaves context untouched, while success adds only
Qdrant-returned local chunks carrying provenance metadata.

- [ ] **Step 5: Commit online graph retrieval**

```bash
git add backend/app/tools/retrieve_graph.py backend/app/services/graph_retriever.py backend/app/agent/nodes/executor.py backend/app/agent/nodes/evidence.py backend/tests/agent/test_graph_retrieval.py
git commit -m "feat: expand local retrieval through citation graph"
```

### Task 5: Route graph retrieval deterministically and expose its stable Timeline step

**Files:**
- Modify: `backend/app/agent/nodes/planner.py`
- Modify: `backend/app/agent/nodes/route.py`
- Modify: `backend/app/agent/prompts/planner.py`
- Modify: `backend/app/agent/stages.py`
- Create: `backend/tests/agent/test_graph_route.py`
- Modify: `backend/tests/agent/test_intent_planner.py`

**Interfaces:**
- Consumes: `retrieve_graph` executor action from Task 4 and graph settings from Task 1.
- Produces: a plan where graph retrieval occurs once after the final local retrieval step and emits the existing stable `step:N` event.

- [ ] **Step 1: Write failing route and plan-parsing tests**

```python
# backend/tests/agent/test_graph_route.py
from unittest.mock import patch

from app.agent.nodes.route import route_node
from app.agent.state import StepSpec


@patch("app.agent.nodes.route.get_settings")
def test_route_injects_graph_after_final_local_retrieval_for_comparison(mock_settings) -> None:
    mock_settings.return_value.graph_rag_enabled = True
    mock_settings.return_value.retrieval_k = 8
    mock_settings.return_value.tavily_api_key = None
    state = {
        "intent": {"type": "comparison", "complexity": "high"},
        "plan": [
            StepSpec(action="query_rewrite", params={"original_query": "compare A B"}, reason="split"),
            StepSpec(action="retrieve_local", params={"query": "A", "top_k": 8}, reason="seed A"),
            StepSpec(action="retrieve_local", params={"query": "B", "top_k": 8}, reason="seed B"),
        ],
        "step_traces": [],
    }

    result = route_node(state, query="compare A and B")

    assert [step["action"] for step in result["plan"]] == ["query_rewrite", "retrieve_local", "retrieve_local", "retrieve_graph"]
    assert result["plan"][-1]["params"] == {"query": "compare A and B", "top_k": 8}


@patch("app.agent.nodes.route.get_settings")
def test_route_removes_graph_action_when_feature_is_disabled(mock_settings) -> None:
    mock_settings.return_value.graph_rag_enabled = False
    mock_settings.return_value.tavily_api_key = None
    state = {
        "intent": {"type": "trend_synthesis", "complexity": "high"},
        "plan": [StepSpec(action="retrieve_graph", params={"query": "trend"}, reason="graph")],
        "step_traces": [],
    }

    result = route_node(state, query="trend")

    assert "retrieve_graph" not in [step["action"] for step in result["plan"]]
    assert result["plan"][0]["action"] == "retrieve_local"
```

Add a planner parsing test whose mocked JSON contains `retrieve_graph` and
asserts `_parse_plan` retains it with a normalized non-empty `query`.

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
cd backend && .venv/bin/python -m pytest tests/agent/test_graph_route.py tests/agent/test_intent_planner.py -q
```

Expected: FAIL because graph actions are not recognized or deterministically ordered.

- [ ] **Step 3: Implement planner, route, and label policy**

Add `retrieve_graph` to `EXECUTABLE_ACTIONS` and `QUERY_ACTIONS`. Add this
planner prompt line:

```text
- retrieve_graph: 基于已命中的本地论文沿引用、作者和类别关系扩展（只适合跨论文比较或趋势问题；必须在 retrieve_local 之后）
```

In `route.py`, define these constants and normalize the plan after policy 1:

```python
_GRAPH_ELIGIBLE_INTENTS = {"comparison", "trend_synthesis"}
_BASE_RETRIEVAL_ACTIONS = {"retrieve_local", "retrieve_arxiv", "search_web"}
_RETRIEVAL_ACTIONS = _BASE_RETRIEVAL_ACTIONS | {"retrieve_graph"}


def _normalize_graph_step(plan: list[StepSpec], *, query: str, enabled: bool, intent_type: str, top_k: int) -> tuple[list[StepSpec], list[str]]:
    without_graph = [step for step in plan if step["action"] != "retrieve_graph"]
    if not enabled or intent_type not in _GRAPH_ELIGIBLE_INTENTS:
        return without_graph, ["dropped_graph_retrieval"] if len(without_graph) != len(plan) else []
    local_indexes = [index for index, step in enumerate(without_graph) if step["action"] == "retrieve_local"]
    if not local_indexes:
        return without_graph, []
    insert_at = local_indexes[-1] + 1
    graph_step = StepSpec(
        action="retrieve_graph",
        params={"query": query, "top_k": top_k},
        reason="路由策略：跨论文问题在本地命中后扩展可验证关系",
    )
    return without_graph[:insert_at] + [graph_step] + without_graph[insert_at:], ["injected_graph_retrieval"]
```

Call `_normalize_graph_step` with `settings.graph_rag_enabled`,
`(state.get("intent") or {}).get("type", "")`, and `settings.retrieval_k`.
For policy 1, calculate `retrieval_steps` from `_BASE_RETRIEVAL_ACTIONS`, not
`_RETRIEVAL_ACTIONS`; this guarantees a graph-only LLM plan receives a local
seed step before disabled graph actions are removed. Do not probe Neo4j in
route; executor owns its lazy connection and warning fallback. Add
`retrieve_graph: "跨论文图谱扩展"` to `ACTION_LABELS`, and add a `"图谱关系"`
source label in route detail when the action is present.

- [ ] **Step 4: Run agent, routing, and streaming regression tests**

Run:

```bash
cd backend && .venv/bin/python -m pytest tests/agent/test_graph_route.py tests/agent/test_intent_planner.py tests/agent/test_graph_retrieval.py tests/agent/test_streaming_events.py -q
```

Expected: PASS; eligible plans contain exactly one graph step after local seeds,
disabled/simple plans contain none, and executor continues to emit stable `step:N` events.

- [ ] **Step 5: Commit deterministic graph routing**

```bash
git add backend/app/agent/nodes/planner.py backend/app/agent/nodes/route.py backend/app/agent/prompts/planner.py backend/app/agent/stages.py backend/tests/agent/test_graph_route.py backend/tests/agent/test_intent_planner.py
git commit -m "feat: route cross-paper questions through graph retrieval"
```

### Task 6: Add graph-aware evaluation, reviewed case selection, and operator documentation

**Files:**
- Modify: `eval/run_rag_eval.py`
- Modify: `eval/rag_metrics.py`
- Create: `eval/scripts/build_graph_eval_manifest.py`
- Create: `eval/datasets/questions_v3_graph.jsonl`
- Create: `eval/tests/test_graph_rag_eval.py`
- Modify: `eval/README.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: `retrieve_graph_context` from Task 4 and existing `summarize_retrieval_cases` breakdowns.
- Produces: `--retriever service_graph`, a graph-case manifest, reviewed graph cases, and documented enablement commands.

- [ ] **Step 1: Write failing graph-evaluation tests**

```python
# eval/tests/test_graph_rag_eval.py
from unittest.mock import patch

from eval.run_rag_eval import run_pure_rag_eval
from eval.rag_metrics import summarize_graph_targeted_cases


@patch("app.services.graph_retriever.retrieve_graph_context")
@patch("app.services.retriever.retrieve")
def test_service_graph_runner_records_graph_retrieval_results(mock_retrieve, mock_graph) -> None:
    mock_retrieve.return_value = []
    mock_graph.return_value = ([], None)

    rows = run_pure_rag_eval(
        questions=[{
            "qid": "g001", "query": "compare graph methods",
            "expected_paper_ids": ["A"], "expected_mode": "answer",
            "difficulty": "hard", "type": "comparison",
        }],
        k_values=[1, 5], context_k=3, retrieval_top_k=5,
        generate=False, retriever_name="service_graph",
    )

    assert rows[0]["qid"] == "g001"
    assert "retrieved_chunks" in rows[0]


def test_graph_targeted_summary_uses_comparison_and_trend_rows_only() -> None:
    summary = summarize_graph_targeted_cases(
        [
            {"type": "comparison", "has_expected": True, "recall_at_5": 1.0, "ndcg_at_5": 1.0, "latency_s": 0.2, "graph_latency_s": 0.1},
            {"type": "trend_synthesis", "has_expected": True, "recall_at_5": 0.5, "ndcg_at_5": 0.5, "latency_s": 0.3, "graph_latency_s": 0.2},
            {"type": "fact_extract", "has_expected": True, "recall_at_5": 0.0, "ndcg_at_5": 0.0, "latency_s": 0.1, "graph_latency_s": None},
        ],
        k_values=[1, 5],
    )

    assert summary["count"] == 2
    assert summary["recall_at_5"] == 0.75
    assert summary["graph_latency_p95"] == 0.2
```

```python
# eval/tests/test_graph_rag_eval.py
from eval.scripts.build_graph_eval_manifest import classify_path


def test_classify_path_labels_direct_two_hop_and_shared_author() -> None:
    assert classify_path(["CITES"]) == "direct_citation"
    assert classify_path(["CITES", "CITES"]) == "two_hop_citation"
    assert classify_path(["AUTHORED_BY", "AUTHORED_BY"]) == "shared_author"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
backend/.venv/bin/python -m pytest eval/tests/test_graph_rag_eval.py -q
```

Expected: FAIL because `service_graph`, the graph manifest helper, and its path classifier do not exist.

- [ ] **Step 3: Implement shared production/evaluation retrieval and reviewed-case selection**

In `run_pure_rag_eval`, add `service_graph` to the allowed retrievers. It must
first call the existing `retrieve(query, top_k=...)`, convert those documents to
the same provenance-decorated local context used by the executor, then call
`retrieve_graph_context`. Convert only its returned local documents into the
existing `build_retrieved_chunks` format. Store graph report fields in each row
only when a non-null report is returned.

Add this metric helper to `eval/rag_metrics.py` and include its result as
`summary["graph_targeted"]` in `run_rag_eval.py`:

```python
def summarize_graph_targeted_cases(cases: list[dict], k_values: list[int]) -> dict:
    targeted = [
        row for row in cases
        if row.get("type") in {"comparison", "trend_synthesis"}
    ]
    summary = _summarize_bucket(targeted, k_values)
    graph_latencies = [
        float(row["graph_latency_s"])
        for row in targeted
        if row.get("graph_latency_s") is not None
    ]
    summary["graph_latency_p95"] = _round(_percentile(graph_latencies, 0.95), 4)
    return summary
```

When `service_graph` receives a `GraphRetrievalReport`, write
`row["graph_latency_s"] = round(report.graph_elapsed_ms / 1000, 4)`. Extend
`_comparison_rows` with `graph_targeted_recall_at_5` and
`graph_targeted_graph_latency_p95` from nested `graph_targeted` summaries so
the JSON comparison directly evaluates the approved gates.

Implement `build_graph_eval_manifest.py` with `classify_path` exactly as tested
above. Its `main()` connects through `get_neo4j_repository()` and writes a JSON
manifest containing three deterministic groups:

```cypher
MATCH (a:Paper {in_corpus: true})-[r:CITES]->(b:Paper {in_corpus: true})
RETURN a.paper_id AS source, b.paper_id AS target, [type(r)] AS relations
ORDER BY source, target
LIMIT 10
```

```cypher
MATCH path=(a:Paper {in_corpus: true})-[:CITES*2]-(b:Paper {in_corpus: true})
WHERE a.paper_id < b.paper_id
RETURN a.paper_id AS source, b.paper_id AS target, [rel IN relationships(path) | type(rel)] AS relations
ORDER BY source, target
LIMIT 10
```

```cypher
MATCH (a:Paper {in_corpus: true})-[:AUTHORED_BY]->(author:Author)<-[:AUTHORED_BY]-(b:Paper {in_corpus: true})
WHERE a.paper_id < b.paper_id
RETURN a.paper_id AS source, b.paper_id AS target, ["AUTHORED_BY", "AUTHORED_BY"] AS relations
ORDER BY source, target
LIMIT 10
```

After running the manifest builder against the synchronized corpus, create one
reviewed Chinese question per manifest row in `questions_v3_graph.jsonl`. Each
record contains `qid`, `query`, `expected_paper_ids`, `expected_mode`,
`reference_answer`, `difficulty`, `type`, `tags`, and `graph_case`. Retain only
records whose reference answer can be supported by local chunks from every
expected paper; discard rows that need external-only evidence.

Document these exact benchmark commands in `eval/README.md`:

```bash
backend/.venv/bin/python eval/run_rag_eval.py --dataset eval/datasets/questions_v3_200.jsonl --retriever service --run-id graph-baseline --context-k 5 --k-values 1 3 5 10
GRAPH_RAG_ENABLED=true backend/.venv/bin/python eval/run_rag_eval.py --dataset eval/datasets/questions_v3_200.jsonl --retriever service_graph --run-id graph-candidate --context-k 5 --k-values 1 3 5 10 --compare-summary eval/results/rag/graph-baseline/summary.json
backend/.venv/bin/python eval/scripts/build_graph_eval_manifest.py --output eval/results/graph-case-manifest.json
GRAPH_RAG_ENABLED=true backend/.venv/bin/python eval/run_rag_eval.py --dataset eval/datasets/questions_v3_graph.jsonl --retriever service_graph --run-id graph-cases --context-k 5 --k-values 1 3 5 10 --generate
```

Update `README.md` storage and architecture sections to name Neo4j as an
optional relationship projection, describe the graph-sync command, and state
that graph expansion is limited to complex cross-paper questions.

- [ ] **Step 4: Run unit tests and the documented baseline/candidate comparison**

Run:

```bash
backend/.venv/bin/python -m pytest eval/tests/test_graph_rag_eval.py eval/tests/test_run_rag_eval.py -q
backend/.venv/bin/python eval/run_rag_eval.py --dataset eval/datasets/questions_v3_200.jsonl --retriever service --run-id graph-baseline --context-k 5 --k-values 1 3 5 10
GRAPH_RAG_ENABLED=true backend/.venv/bin/python eval/run_rag_eval.py --dataset eval/datasets/questions_v3_200.jsonl --retriever service_graph --run-id graph-candidate --context-k 5 --k-values 1 3 5 10 --compare-summary eval/results/rag/graph-baseline/summary.json
```

Expected: unit tests PASS and both runs write `summary.json`, `manifest.json`,
`per_question.jsonl`, and `report.md`. Enable the feature flag only if the
approved gates are satisfied: comparison/trend Recall@5 improves by at least
0.05 absolute, whole-positive NDCG@5 declines by at most 0.01 absolute,
fixed-context citation support is 1.00, and warm graph expansion P95 is at
most 0.800 seconds.

- [ ] **Step 5: Commit evaluation and documentation**

```bash
git add eval/run_rag_eval.py eval/scripts/build_graph_eval_manifest.py eval/datasets/questions_v3_graph.jsonl eval/tests/test_graph_rag_eval.py eval/README.md README.md
git commit -m "feat: evaluate cross-paper graph rag"
```

## Final Verification

- [ ] Run the complete backend suite:

```bash
cd backend && .venv/bin/python -m pytest tests -q
```

- [ ] Run the pure-RAG evaluation test suite:

```bash
cd .. && backend/.venv/bin/python -m pytest eval/tests -q
```

- [ ] Verify Docker health and disabled fallback:

```bash
docker compose up -d mysql qdrant neo4j backend
GRAPH_RAG_ENABLED=false docker compose exec backend python -c "from app.core.config import get_settings; assert get_settings().graph_rag_enabled is False"
docker compose exec neo4j cypher-shell -u neo4j -p "$NEO4J_PASSWORD" "MATCH (p:Paper) RETURN count(p) AS papers"
```

- [ ] Verify the final diff contains only intended Graph RAG files and no secret:

```bash
git diff --check HEAD~6..HEAD
git log --oneline -6
rg -n "(NEO4J_PASSWORD=.{1,}|SEMANTIC_SCHOLAR_API_KEY=.{1,})" --glob '!*.example' --glob '!docs/superpowers/**' .
```

Expected: whitespace check passes, six focused commits are visible, and the
secret scan returns no tracked credential value.
