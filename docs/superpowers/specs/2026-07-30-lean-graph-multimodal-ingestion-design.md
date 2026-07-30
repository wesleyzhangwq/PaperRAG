# Lean Graph and Heterogeneous Ingestion Design

Date: 2026-07-30

## Goals

1. Reduce orchestration complexity without removing safety or observability.
2. Give arXiv and local files one provenance-preserving ingestion path.
3. Make uploaded evidence citable through the same groundedness and citation
   safety chain as arXiv evidence.

## Graph boundary rule

A LangGraph node is justified when the runtime can branch, loop, retry, or
persist/checkpoint at that boundary. A user-visible stage does not need to be a
separate graph node when it always executes directly beside another stage.

The control graph therefore has eight nodes:

1. `guard`
2. `analyze` (`intent` + `complexity_router`)
3. `plan` (initial planner or re-planner + deterministic route policy)
4. `executor`
5. `evidence_gate` (filter/rerank/compress + sufficiency)
6. `synthesis`
7. `groundedness`
8. `finalize` (citation gate + presentation)

The former 13 node responsibilities remain intact, and the existing stable
stage IDs plus `StepTrace` records remain separate. This keeps the frontend
timeline and failure telemetry precise while making the checkpoint graph
explainable.

`fast_local`, `full_agentic`, and `fast_escalated` behavior is unchanged:

- `full_agentic` remains the safe default and runs the planner.
- `fast_local` is only available in explicit `auto` mode and starts with one
  deterministic local-retrieval plan.
- failed fast-path sufficiency escalates once to the full planner and records
  `fast_escalated`.

The router's dev50/frozen200 acceptance evaluation is still required before
claiming latency or quality gains or changing the default.

## Unified ingestion state machine

```text
queued -> saved -> parsing -> normalizing -> chunking
       -> indexing -> persisting -> completed
       \-------------------------------------> failed
```

Every upload job exposes `stage`, `progress`, `error_code`, warnings, media
type, and content hash. arXiv jobs enter the same parsing/indexing stages after
metadata fetch and PDF download.

## Supported formats and modalities

| Input | Extraction |
| --- | --- |
| PDF | native text, tables, scanned-page OCR, embedded-image OCR |
| DOCX | heading-aware text sections, tables, embedded-image OCR |
| PPTX | slide text, tables, picture OCR |
| HTML | active-content removal, body text, tables |
| Markdown/TXT | decoded text |
| CSV/XLSX | table blocks with sheet locators |
| PNG/JPEG/WebP/TIFF | local Tesseract OCR |

Every normalized block and persisted chunk has:

- `modality`: `text`, `table`, or `image_ocr`
- `source_locator`: page, slide, sheet, section, table, or image identifiers
- source kind, media type, original filename, and SHA-256 content hash

Office ZIP expansion and file size are bounded before parsing. OCR is local and
configurable; failures become job warnings unless the file has no other usable
content.

## Persistence boundary

The service parses and chunks before modifying corpus rows. It upserts the new
deterministic vector IDs before replacing MySQL chunks, validates that the
embedding provider returns one vector per text, and never suppresses stale
Qdrant deletion failures. This prevents an embedding outage from deleting the
last committed relational chunk set.

MySQL and Qdrant do not provide a shared distributed transaction. The design is
therefore failure-aware rather than falsely claiming cross-store atomicity.

## Citation compatibility

- arXiv evidence: `[arxiv:1706.03762]` (existing form retained)
- uploaded evidence: `[source:local-<sha-prefix>]`

Both forms are parsed by the deterministic precheck, groundedness stage,
citation gate, Markdown pills, and source-card UI. Local sources deliberately
do not receive fabricated arXiv links.

## Verification

- Full backend suite: 160 tests passed.
- Frontend: TypeScript/Vite production build passed.
- Frontend citation/timeline/thinking tests: 3 passed.
- Chat UX static checks passed.

Frozen routing latency/quality evaluation and live provider/database E2E remain
separate operational acceptance steps; neither is claimed here.
