# PaperRAG — Architecture & Design Philosophy

## What This Project Is

PaperRAG is an **Agentic RAG** (Retrieval-Augmented Generation) system for academic paper Q&A.
It ingests arXiv PDFs into a vector store, then uses an LLM-driven adaptive planning agent
to retrieve, evaluate, and synthesize cited answers — with self-reflection for quality control.

The system is designed for portfolio/interview presentation: it prioritizes
**reasoning depth**, **architectural completeness**, and **explainability** over raw latency.

---

## Architecture Overview

```
User Query + Chat History
        │
        ▼
┌──────────────┐
│    intent     │  LLM classifies: type (simple/complex/comparison), entities, complexity
└──────┬───────┘
       ▼
┌──────────────┐
│   planner     │  LLM generates a structured execution plan: List[StepSpec]
└──────┬───────┘
       ▼
┌──────────────┐ ◄──────────────────────────────┐
│   executor    │  Dispatches plan steps to 7    │
│   (loop)      │  tools sequentially            │
└──────┬───────┘                                 │
       ▼                                         │
┌──────────────┐                                 │
│  synthesis    │  Generates cited answer from    │
│  (streaming)  │  accumulated context            │
└──────┬───────┘                                 │
       ▼                                         │
┌──────────────┐      fail + re_retrieve         │
│  reflection   │ ──────────────────────► ┌──────────────┐
│  (3 dims)     │                         │  re_planner   │
└──────┬───────┘      fail + re_generate  └──────┬───────┘
       │           ──────────► synthesis          │
       │ pass                                     │
       ▼
┌──────────────┐
│ final_answer  │  Extract [arxiv:ID] → Source objects from DB
└──────┬───────┘
       ▼
┌──────────────┐
│ presentation  │  Build productized UI payload (confidence, step labels, source cards)
└──────────────┘
       ▼
      END
```

### Tech Stack

| Layer     | Technology                                       |
|-----------|--------------------------------------------------|
| Agent     | LangGraph StateGraph (7 nodes + conditional edges) |
| LLM       | MiniMax M2.7 via OpenAI-compatible API           |
| Embedding | Alibaba text-embedding-v4 (DashScope)            |
| Vector DB | Qdrant (hybrid: dense + BM25 sparse fusion)      |
| SQL DB    | MySQL 8 (paper metadata, chat history, conversations) |
| Backend   | FastAPI + SQLAlchemy + Pydantic                  |
| Frontend  | Vue 3 + Tailwind CSS + Pinia                     |
| Streaming | SSE (Server-Sent Events) via thread + Queue      |

---

## Design Principles

1. **Adaptive over fixed.** The planner generates per-query execution plans instead of a hardcoded
   pipeline. A simple factual question gets 3 steps; a complex comparison gets 7+.

2. **Self-verification.** Every answer passes through a 3-dimension reflection check (citation
   faithfulness, completeness, logical consistency) before reaching the user. Failed checks
   trigger re-retrieval or re-generation, up to a configurable budget.

3. **Transparent reasoning.** Every agent step emits a `StepTrace` with timing and summaries.
   The frontend's `ThinkingCard` component renders these in real time so users see *how* the
   answer was built, not just the answer itself.

4. **Tools as interfaces, not entangled code.** Each of the 7 tools (`retrieve_local`,
   `retrieve_arxiv`, `search_web`, `query_rewrite`, `evaluate_docs`, `get_paper_detail`,
   `get_paper_chunks`) is a standalone module under `backend/app/tools/`. The executor
   dispatches by action name; adding a tool means adding a file and a dispatch case.

5. **Separation of concerns in graph nodes.** Nodes are pure functions
   `(state, **deps) → partial state update`. They don't know about HTTP, SSE, or threads.
   The streaming infrastructure (`streaming.py` + `chat.py` worker thread) wraps the graph
   externally.

---

## Backend Structure

```
backend/app/
├── agent/                     # LangGraph agent core
│   ├── graph.py               # StateGraph definition + compile + run_agent_sync()
│   ├── state.py               # AgentState, StepSpec, StepTrace, ReflectionResult
│   ├── streaming.py           # ContextVar-based Queue for real-time SSE push
│   ├── nodes/                 # One file per graph node
│   │   ├── intent.py          # Query classification (LLM)
│   │   ├── planner.py         # Plan generation + re_planner (LLM)
│   │   ├── executor.py        # Tool dispatch loop (no LLM, calls tools)
│   │   ├── synthesis.py       # Answer generation with streaming (LLM)
│   │   ├── reflection.py      # 3-dimension self-verification (LLM)
│   │   ├── final_answer.py    # Citation → Source object resolution (DB)
│   │   └── presentation.py    # UI-friendly payload (confidence, labels)
│   └── prompts/               # System prompts for each LLM-calling node
│
├── tools/                     # 7 standalone tools
│   ├── retrieve_local.py      # Qdrant vector + BM25 hybrid search
│   ├── retrieve_arxiv.py      # arXiv API real-time search
│   ├── search_web.py          # Tavily web search (graceful fallback)
│   ├── query_rewrite.py       # LLM sub-question decomposition
│   ├── evaluate_docs.py       # LLM document sufficiency check
│   ├── paper_detail.py        # MySQL paper metadata lookup
│   └── paper_chunks.py        # Chunk retrieval by paper_id
│
├── routers/                   # FastAPI endpoints
│   ├── chat.py                # POST /chat (sync) + POST /chat/stream (SSE)
│   ├── conversations.py       # CRUD for conversation sessions
│   ├── papers.py              # GET /papers (list/search)
│   ├── upload.py              # POST /upload (PDF upload + auto-ingest)
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
│   ├── useChat.ts             # Send message → handle all SSE events
│   └── useThinking.ts         # Agent step state machine (pending → running → done)
│
├── components/
│   ├── chat/
│   │   ├── ThinkingCard.vue   # Collapsible agent reasoning display
│   │   ├── StepIndicator.vue  # Per-step status (○ pending, ◐ running, ● done ✓/✗)
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

## SSE Streaming Protocol

```
POST /chat/stream → text/event-stream

event: conversation   → { conversation_id }
event: intent         → { type, entities, complexity }
event: plan           → { steps: [...], total_steps }
event: step_start     → { index, action, reason }
event: step_done      → { node, action, input_summary, output_summary, duration_ms }
event: token          → { t: "partial text" }        # real-time synthesis tokens
event: reflection     → { passed, citation_ok, completeness_ok, logic_ok, issues }
event: re_plan        → { new_steps: [...] }
event: sources        → { sources: [{ paper_id, title, ... }] }
event: presentation   → { confidence, steps, retrieval_summary, source_cards }
event: elapsed        → { ms }                        # heartbeat
event: done           → { steps_count, reflections }
event: error          → { message, type }
```

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
