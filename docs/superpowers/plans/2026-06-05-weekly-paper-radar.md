# Weekly Paper Radar Implementation Plan

## Goal

Add an application-level scheduled PaperRAG job that monitors a narrow research direction, selects the most relevant 10 recent papers each week, writes an explainable digest, ingests available PDFs, and removes PDFs after ingest.

## Direction

Default vertical direction: Agentic RAG for Scientific AI.

Scope includes retrieval-augmented generation, dense retrieval, reranking, LLM agents, tool use, reasoning, reflection, evaluation, hallucination, factuality, citation attribution, and scientific question answering.

## Implementation

1. Add a backend `weekly_radar` service with pure ranking/reporting helpers and a network orchestration function.
2. Fetch recent arXiv candidates for `cs.CL`, `cs.AI`, `cs.IR`, and `cs.LG` over a configurable time window.
3. Score papers using an explainable weighted score:
   - topic relevance: 40
   - citation/impact signal: 25
   - category match: 15
   - recency: 10
   - novelty/benchmark/framework signal: 10
4. Prefer arXiv candidates because the current source resolver and citation UI are arXiv-id based.
5. Write weekly JSON and Markdown reports under `data/weekly_paper_radar/`.
6. Add a CLI and shell runner as the stable task entrypoint.
7. Configure a Codex app automation for Monday 09:00 local time.

## Acceptance

1. Unit tests cover scoring, dedupe, report writing, ingest-record conversion, and dry-run behavior.
2. A dry-run can fetch/rank/write a report without downloading PDFs or calling ingest.
3. Codex app automation contains one PaperRAG weekly radar entry pointing at the shell runner.
4. Generated reports and weekly PDFs are not committed.
