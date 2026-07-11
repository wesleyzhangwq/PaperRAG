# 501-Paper Pure RAG Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a paper-disjoint 50-question development benchmark and 200-question frozen pure-RAG benchmark against the repaired 501-paper corpus, then update the factual resume source and one-page master with validated results.

**Architecture:** Export bounded representative evidence from MySQL, split the paper corpus deterministically before question generation, and preserve evidence chunk IDs in each positive question. Tune only on development data, run lexical and dense baselines plus one frozen hybrid candidate on the test data, and audit paired per-question deltas before updating resume claims.

**Tech Stack:** Python 3.12, SQLAlchemy/MySQL, Qdrant, bge-m3, rank-bm25, pytest, HTML/CSS print verification.

## Global Constraints

- MySQL 501 papers and 54,467 chunks are the corpus source of truth.
- Development and frozen-test expected paper IDs are disjoint.
- The 200-question test distribution is 60 / 40 / 30 / 30 / 20 / 20.
- The 50-question development distribution is 15 / 10 / 8 / 7 / 5 / 5.
- Agentic RAG and Graph RAG are excluded from this evaluation.
- No frozen-test retuning is allowed.
- Resume claims must cite persisted manifests and summaries.

---

### Task 1: MySQL Evidence Corpus Export

**Files:**
- Create: `eval/scripts/export_mysql_corpus.py`
- Create: `eval/tests/test_export_mysql_corpus.py`

**Interfaces:**
- Produces: `select_representative_chunks(chunks, limit) -> list[dict]`
- Produces: `build_paper_record(paper, chunks, evidence_limit) -> dict`
- Produces: JSON corpus consumed by Task 2 and by lexical retrieval.

- [ ] Write failing tests proving deterministic selection, bounded evidence text, secret redaction, and chunk ownership.
- [ ] Run `PYTHONPATH=. /Users/wesz_station/Projects/PaperRAG/backend/.venv/bin/python -m pytest eval/tests/test_export_mysql_corpus.py -q` and confirm failure because the module is absent.
- [ ] Implement the exporter using SQLAlchemy `Paper` and `Chunk` rows, stable ordering, representative section signals, and atomic JSON output.
- [ ] Run the focused tests and then the full `eval/tests` suite.
- [ ] Export `eval/datasets/mysql_papers_501_20260711.json` and verify 501 records, 54,467 total chunks, unique paper IDs, and non-empty evidence for every paper.

### Task 2: Paper-Disjoint Dataset Generation

**Files:**
- Modify: `eval/scripts/gen_questions.py`
- Modify: `eval/tests/test_gen_questions.py`
- Create: `eval/datasets/questions_501_dev_50.jsonl`
- Create: `eval/datasets/questions_501_test_200.jsonl`
- Create: `eval/datasets/questions_501_manifest.json`

**Interfaces:**
- Produces: `split_papers_by_category(papers, dev_size, seed) -> tuple[list[dict], list[dict]]`
- Produces: positive rows containing `evidence_chunk_ids` and `generation_status="llm"`.
- Consumes: Task 1 corpus JSON.

- [ ] Add failing tests for deterministic category-stratified paper splitting, paper disjointness, evidence chunk validation, arbitrary expected distributions, and fallback rejection.
- [ ] Run the focused tests and confirm failures for the missing behavior.
- [ ] Update prompts to use labeled representative chunks and require evidence chunk IDs; add CLI arguments for split, seed, and per-type counts.
- [ ] Run focused and full eval tests.
- [ ] Generate the 50-question development set and 200-question frozen test set with the configured LLM.
- [ ] Run deterministic dataset validation and write a manifest containing corpus hash, split IDs, generation model, seed, and quality counts.

### Task 3: Development-Set Baselines and Tuning

**Files:**
- Modify only if tests expose a runner defect: `eval/run_rag_eval.py`, `eval/tests/test_run_rag_eval.py`
- Create through runner output: `eval/results/rag/rag-501-dev-*`

**Interfaces:**
- Consumes: development JSONL and Task 1 lexical corpus.
- Produces: comparable `summary.json`, `per_question.jsonl`, `manifest.json`, and `report.md` runs.

- [ ] Run the lexical BM25 floor on all 50 development questions.
- [ ] Run a dense-only `bge-m3` baseline with hybrid retrieval disabled.
- [ ] Sweep hybrid alpha, retrieval depth, oversampling, and raw/paper-dedup/MMR context strategies with one variable family at a time.
- [ ] Select one candidate by NDCG@5 then Recall@5 under latency and context-precision guardrails.
- [ ] Persist the selected settings before reading frozen-test results.

### Task 4: Frozen 200-Question Evaluation and Paired Audit

**Files:**
- Create: `eval/scripts/compare_rag_runs.py`
- Create: `eval/tests/test_compare_rag_runs.py`
- Create through runner output: `eval/results/rag/rag-501-test-*`

**Interfaces:**
- Produces: `paired_bootstrap_delta(baseline, candidate, metric, samples, seed) -> dict`
- Produces: final comparison JSON/Markdown with paired 95% confidence intervals.

- [ ] Write failing tests for row alignment, duplicate/missing qid rejection, deterministic bootstrap intervals, and metric delta direction.
- [ ] Implement paired comparison and run focused/full eval tests.
- [ ] Run lexical floor, dense-only baseline, and the frozen tuned candidate exactly once on the 200-question test.
- [ ] Generate paired deltas, per-type tables, latency guardrails, and a data-quality report.
- [ ] Verify all 200 rows completed and all manifests identify the 501-paper active collection.

### Task 5: Documentation and Resume Evidence

**Files:**
- Modify: `eval/README.md`
- Modify: `/Users/wesz_station/Projects/resume/facts.md`
- Modify: `/Users/wesz_station/Projects/resume/master/resume-master.html`

**Interfaces:**
- Consumes: final manifests, summaries, paired comparison, and quality report.
- Produces: current factual evidence and a one-page master resume.

- [ ] Replace current-corpus references to 9,704 chunks with the verified 501-paper / 54,467-chunk corpus and new frozen-test metrics.
- [ ] Move old 100-paper and 30-question Agentic metrics to a historical, non-resume section.
- [ ] Update the PaperRAG resume bullets without claiming Graph-RAG gains.
- [ ] Run PaperRAG tests, dataset validators, resume print checks, and rendered visual inspection.
- [ ] Record exact result paths and remaining Graph-RAG dependency on Semantic Scholar/Neo4j.

