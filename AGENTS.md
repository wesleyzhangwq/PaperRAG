# Cite Scope — Architecture & Design Philosophy

## What This Project Is

Cite Scope is an **Agentic RAG** (Retrieval-Augmented Generation) system for academic paper Q&A.
It imports arXiv metadata and PDFs into a vector store, then uses an LLM-driven adaptive planning agent
to retrieve, evaluate, and synthesize cited answers — with self-reflection for quality control.

The system is designed for portfolio/interview presentation: it prioritizes
**reasoning depth**, **architectural completeness**, and **explainability** over raw latency.

---

## Architecture Overview

The graph mirrors the enterprise agentic-RAG pipeline with 13 nodes. A
deterministic complexity router sits between intent classification and
planning. `full_agentic` is the safe default and always retains the planner;
`auto` is an explicit opt-in that may give a low-risk query one deterministic
local retrieval step instead. Both branches rejoin before retrieval and keep
the complete evidence and citation-safety chain:

```
User Query + Chat History
        │
        ▼
      guard
   ├─ blocked → presentation → END
   └─ ok
        │
        ▼
      intent
        │
        ▼
 complexity_router
   ├─ full_agentic (default, forced, or any veto) → planner ┐
   └─ fast_local (auto only; one retrieve_local plan) ──────┤
                                                            ▼
                                                          route
                                                            │
                                                            ▼
                                                     executor (loop)
                                                            │ plan exhausted
                                                            ▼
                                                         evidence
                                                            │
                                                            ▼
                                                       sufficiency
   ├─ sufficient / degraded ────────────────────────────────┤
   ├─ first fast-path insufficiency → planner once          │
   │  (execution_path becomes fast_escalated) → route       │
   └─ other insufficiency + budget → re_planner → executor  │
                                                            ▼
                                                     synthesis (streaming)
                                                            │
                                                            ▼
                                                      groundedness
   ├─ re_generate ───────────────────────────────→ synthesis
   ├─ re_retrieve ───────────────────────────────→ re_planner
   └─ pass / budget exhausted
        │
        ▼
 citation_gate → presentation → END
```

### Tech Stack

| Layer     | Technology                                       |
|-----------|--------------------------------------------------|
| Agent     | LangGraph StateGraph (13 nodes + conditional edges) |
| LLM       | MiniMax M2.7 via OpenAI-compatible API           |
| Embedding | SiliconFlow BAAI/bge-m3 via OpenAI-compatible API |
| Vector DB | Qdrant (hybrid: dense + BM25 sparse fusion)      |
| SQL DB    | MySQL 8 (paper metadata, chat history, conversations) |
| Backend   | FastAPI + SQLAlchemy + Pydantic                  |
| Frontend  | Vue 3 + Tailwind CSS + Pinia                     |
| Streaming | SSE via LangGraph `astream_events` + custom event adapter |

---

## Design Principles

1. **Conservative routing before adaptive planning.** The default
   `AGENT_ROUTING_MODE=full_agentic` always preserves the LLM planner. The
   opt-in `auto` mode may use one deterministic local retrieval step only for
   high-confidence, low-risk queries; uncertainty and policy vetoes retain the
   planner. A fast-path run still passes through sufficiency, synthesis,
   groundedness, and the citation gate, and may be marked `fast_escalated`.

2. **Self-verification.** Every answer passes through a 3-dimension reflection check (citation
   faithfulness, completeness, logical consistency) before reaching the user. Failed checks
   trigger re-retrieval or re-generation, up to a configurable budget.

3. **Transparent reasoning.** Every agent step emits a `StepTrace` with timing and summaries.
   The frontend's `AgentTimeline` component renders these in real time so users see *how* the
   answer was built, not just the answer itself.

4. **Tools as interfaces, not entangled code.** Each of the 6 executor tools
   (`retrieve_local`, `retrieve_arxiv`, `search_web`, `query_rewrite`,
   `get_paper_detail`, `get_paper_chunks`) is a standalone module under
   `backend/app/tools/`. Evidence sufficiency is a graph stage, not an executor
   tool. The executor dispatches by action name; adding a tool means adding a
   file and a dispatch case.

5. **Separation of concerns in graph nodes.** Nodes return partial state updates and only emit
   LangGraph custom events for live UI traces. HTTP and SSE protocol mapping stay in
   `backend/app/routers/chat.py` and `backend/app/agent/streaming.py`.

---

## Backend Structure

```
backend/app/
├── agent/                     # LangGraph agent core
│   ├── graph.py               # StateGraph definition + compile + run_agent_sync()
│   ├── state.py               # AgentState, StepSpec, StepTrace, ReflectionResult
│   ├── streaming.py           # custom-event forwarder + final_state folding
│   ├── stages.py              # stage() context manager, STAGE_TITLES/ACTION_LABELS, emit_plan
│   ├── nodes/                 # One file per graph node
│   │   ├── guard.py           # Content safety & validity checks (deterministic)
│   │   ├── intent.py          # Query classification (LLM)
│   │   ├── complexity_router.py # Fast/full policy + bounded audit decision
│   │   ├── planner.py         # Retrieval-only plan + re_planner (LLM)
│   │   ├── route.py           # Source-routing policy (deterministic)
│   │   ├── executor.py        # Tool dispatch loop (no LLM, calls tools)
│   │   ├── evidence.py        # Filter/rerank/compress context (deterministic)
│   │   ├── sufficiency.py     # Evidence sufficiency gate + conditional edge (LLM)
│   │   ├── synthesis.py       # Answer generation with streaming (LLM)
│   │   ├── groundedness.py    # Citation precheck + 3-dim verification (LLM)
│   │   ├── citation_gate.py   # Citation resolution + hallucination strip (DB)
│   │   └── presentation.py    # UI-friendly payload (confidence, labels)
│   └── prompts/               # System prompts for each LLM-calling node
│
├── tools/                     # 6 standalone executor tools + graph-stage helpers
│   ├── retrieve_local.py      # Qdrant vector + BM25 hybrid search
│   ├── retrieve_arxiv.py      # arXiv API real-time search
│   ├── search_web.py          # Tavily web search (graceful fallback)
│   ├── query_rewrite.py       # LLM sub-question decomposition
│   ├── evaluate_docs.py       # LLM evidence sufficiency helper used by the graph stage
│   ├── paper_detail.py        # MySQL paper metadata lookup
│   └── paper_chunks.py        # Chunk retrieval by paper_id
│
├── routers/                   # FastAPI endpoints
│   ├── chat.py                # POST /chat (sync) + POST /chat/stream (SSE)
│   ├── conversations.py       # CRUD for conversation sessions
│   ├── papers.py              # GET /papers (list/search)
│   ├── upload.py              # POST /upload/arxiv (arXiv import + auto-ingest)
│   └── ingest.py              # POST /ingest (admin: re-process PDFs)
│
├── services/
│   ├── retriever.py           # Core retrieval engine (vector + BM25 + cache)
│   └── ingest.py              # PDF → chunks → Qdrant ingestion pipeline
│
├── db/
│   ├── mysql.py               # SQLAlchemy engine + session
│   └── qdrant.py              # Qdrant client + embedding provider
│
├── models/                    # SQLAlchemy ORM
│   ├── paper.py               # Paper + Chunk tables
│   ├── chat_history.py        # Chat messages with sources/thinking JSON
│   └── conversation.py        # Conversation sessions
│
├── schemas/chat.py            # Pydantic request/response models
├── core/config.py             # All settings from .env
└── main.py                    # FastAPI app entry
```

## Frontend Structure

```
frontend/src/
├── composables/
│   ├── useSSE.ts              # SSE connection + async generator
│   └── useChat.ts             # Send message → handle all SSE events and timeline upserts
│
├── components/
│   ├── chat/
│   │   ├── AgentTimeline.vue  # Stable-id stage/tool timeline with live status
│   │   ├── AssistantBubble.vue # Markdown + CitationPopover + source list
│   │   ├── UserBubble.vue
│   │   ├── MessageList.vue
│   │   └── InputArea.vue
│   └── citation/
│       └── CitationPopover.vue # Hover popover: paper title, authors, year, arXiv link
│
├── stores/
│   ├── chat.ts                # Messages + session state (Pinia)
│   └── conversations.ts       # Conversation list
│
├── layouts/ChatLayout.vue     # Sidebar + main area
├── views/ChatView.vue
├── types/index.ts             # All TypeScript interfaces
├── utils/markdown.ts          # markdown-it + citation pill rendering
└── styles/base.css            # CSS variables (Claude-inspired warm palette)
```

---

## SSE Streaming Protocol (v2 — stable-id stage events)

Every pipeline node self-reports lifecycle `stage` events with a STABLE id;
the frontend upserts a timeline by id — no index reconstruction.

```
POST /chat/stream → text/event-stream

event: conversation   → { conversation_id }
event: stage          → { id, stage, status, title, summary?, detail?, duration_ms? }
                        # id: "guard"|"intent"|"complexity"|"plan"|"route"|"step:N"|
                        #     "evidence"|"sufficiency"|"synthesis"|"groundedness"|
                        #     "citation"
                        # status: start | done | warning | failed | skipped
event: plan           → { revision, steps: [{ id: "step:N", action, title, reason }] }
                        # re-published on every plan change (planner/route/re_planner)
event: answer_start   → { attempt, reset }            # re-generation resets the bubble
event: token          → { t: "partial text" }         # real-time synthesis tokens
event: sources        → { sources: [{ paper_id, title, ... }] }
event: presentation   → { confidence, steps, retrieval_summary, source_cards,
                          execution_path, complexity_decision }
event: elapsed        → { ms }                        # heartbeat
event: done           → { steps_count, reflections, execution_path }
event: error          → { message, type }
```

Frontend (`useSSE.ts`) parses per the SSE spec (multi-line data, CRLF-tolerant,
blank-line dispatch). Tokens render via rAF batching — zero artificial delay,
at most one DOM update per frame. The timeline reducer lives in
`frontend/src/utils/timeline.ts`; the UI is `AgentTimeline.vue`.

---

## Self-Reflection (Reflexion Pattern)

Three verification dimensions, each a boolean check:

| Dimension           | What it checks                                           |
|---------------------|----------------------------------------------------------|
| Citation faithfulness | Every `[arxiv:ID]` in the answer exists in retrieved context |
| Completeness        | The answer addresses all aspects of the user's question   |
| Logical consistency | No internal contradictions; coherent reasoning chain      |

Fix strategies on failure:
- **`re_retrieve`** → re_planner generates supplementary retrieval steps → executor runs them → re-synthesize
- **`re_generate`** → context is sufficient but answer has issues → re-run synthesis with issues as constraints
- **Budget exhausted** (`reflection_count >= 2`) → force output current best answer

---

## Key Configuration (.env)

```bash
# Agent behavior
AGENT_MAX_PLAN_STEPS=7          # Max steps the planner can generate
AGENT_MAX_REFLECTIONS=2         # Max reflection retries before force-output
AGENT_ROUTING_MODE=full_agentic # Safe default; auto requires frozen-eval acceptance

# Retrieval tuning
RETRIEVAL_K=12                  # Initial retrieval count
HYBRID_ALPHA=0.3                # Vector vs BM25 weight (0=pure BM25, 1=pure vector)
HYBRID_OVERSAMPLE=2.5           # Oversample factor for hybrid fusion

# External tools
TAVILY_API_KEY=                 # Web search (optional, graceful fallback)
ARXIV_MAX_RESULTS=5             # arXiv results per query

# Security
ADMIN_API_KEY=                  # Protects /ingest endpoint; open if empty
```

The complexity router has unit and graph-integration coverage, but its
`dev50`/`frozen200` latency and quality evaluation has **not** been run in this
worktree. Do not enable `auto` by default or claim a latency/quality improvement
until those frozen-evaluation acceptance checks pass.

---

## Conventions for Contributors

- **Pure node functions.** Agent nodes receive `(state, **kwargs)` and return a partial state dict.
  No side effects beyond the returned state (except `emit()` for SSE tokens in synthesis).
- **One tool = one file** under `backend/app/tools/`. Each tool is independently testable.
- **Tests mock LLM and DB**, never call real APIs. Use `patch("module._get_llm", ...)` pattern.
  Synthesis mocks need `mock_llm.stream.return_value` (not `.invoke`).
- **Frontend event handling** lives in `useChat.ts`. All SSE event types are handled in the
  switch statement there. Adding a new event type means: add to `SSEEvent` union type in
  `useSSE.ts`, handle in `useChat.ts`, update relevant composable/store.
