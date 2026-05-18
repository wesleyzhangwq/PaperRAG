# PaperRAG v2: Agentic RAG Full Redesign

## Overview

Redesign PaperRAG from a fixed pipeline + simple ReAct agent into a full Agentic RAG system with adaptive planning, multi-hop retrieval, self-reflection, and transparent reasoning.

**Goal:** Portfolio/interview-oriented. Prioritize reasoning depth, architectural completeness, and explainability over latency optimization.

**Approach:** Adaptive Planner Agent (方案 B) — LLM-driven planning determines which steps to execute based on query complexity, with a Reflexion loop for self-verification.

## Architecture: Adaptive Planner Agent

### Core Flow

```
User Query + History
       │
       ▼
┌──────────────┐
│ intent_node  │  Analyze intent: type(simple/complex/comparison), entities, complexity
└──────┬───────┘
       ▼
┌──────────────┐
│ planner_node │  Generate structured execution plan based on intent
└──────┬───────┘
       ▼
┌──────────────────┐
│ executor_node    │  Execute plan steps sequentially:
│  (loops over     │    query_rewrite, retrieve_local, retrieve_arxiv,
│   plan steps)    │    search_web, evaluate_docs, reasoning_synthesis
└──────┬───────────┘
       ▼
┌──────────────────┐         ┌─────────────┐
│ self_reflection  │──fail──▶│ re_planner  │
│                  │         └──────┬──────┘
└──────┬───────────┘                │
       │ pass                       ▼
       ▼                    back to executor_node
┌──────────────┐
│ final_answer │  Format + build citations
└──────┬───────┘
       ▼
      END
```

### State Schema

```python
class StepSpec(TypedDict):
    action: str          # "query_rewrite" | "retrieve_local" | "retrieve_arxiv" | "search_web" | "evaluate_docs" | "reasoning_synthesis"
    params: dict         # Tool-specific parameters
    reason: str          # Why this step is needed (explainability)

class StepTrace(TypedDict):
    node: str            # Node that executed
    action: str          # Action performed
    input_summary: str   # Human-readable input summary (for frontend)
    output_summary: str  # Human-readable output summary
    duration_ms: float   # Execution time

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: Optional[dict]              # {type, entities, complexity}
    plan: list[StepSpec]                # Current execution plan
    plan_step_index: int                # Current step being executed
    retrieval_context: list[Document]   # Accumulated retrieval results
    step_traces: list[StepTrace]        # Execution trace (streamed to frontend via SSE)
    reflection_count: int               # Number of reflections performed
    final_answer: Optional[str]         # Final generated answer
```

### LangGraph Nodes

| Node | Responsibility | LLM Call? |
|------|----------------|-----------|
| `intent_node` | Classify query type, extract entities, assess complexity | Yes |
| `planner_node` | Generate `List[StepSpec]` execution plan | Yes |
| `executor_node` | Dispatch current plan step to corresponding tool | No (dispatches) |
| `reasoning_synthesis` | Generate cited answer from accumulated context | Yes (streaming) |
| `self_reflection` | 3-dimension verification (citation, completeness, logic) | Yes |
| `re_planner` | Generate supplementary plan from reflection feedback | Yes |
| `final_answer` | Format output, build Source objects from citations | No |

### Conditional Edges

```python
# After executor: check if more steps remain
"executor_node" → if plan_step_index < len(plan): "executor_node"
                  else: "reasoning_synthesis"

# After reflection: pass or fail
"self_reflection" → if pass: "final_answer"
                    elif reflection_count < MAX_REFLECTIONS: "re_planner"
                    else: "final_answer"  # force output with disclaimer

# After re_planner: back to execution
"re_planner" → "executor_node"
```

## Tools (7)

### Retrieval Tools

| Tool | Input | Output | Backend |
|------|-------|--------|---------|
| `retrieve_local` | `query: str, top_k: int=8, filter?: {category, year_min, year_max}` | `List[ChunkResult]` with scores | Qdrant vector search + BM25 hybrid rerank |
| `retrieve_arxiv` | `query: str, max_results: int=5` | `List[ArxivPaper]` (title, abstract, id, year) | arXiv API (real-time, metadata only) |
| `search_web` | `query: str, max_results: int=3` | `List[WebResult]` (title, snippet, url) | Tavily API |

### Knowledge Tools

| Tool | Input | Output | Backend |
|------|-------|--------|---------|
| `get_paper_detail` | `paper_id: str` | Full metadata string | MySQL query |
| `get_paper_chunks` | `paper_id: str, max_chunks: int=10` | Top chunks for that paper | MySQL + Qdrant |

### Reasoning Tools

| Tool | Input | Output | Backend |
|------|-------|--------|---------|
| `query_rewrite` | `original_query: str, intent: dict` | `List[str]` (1-3 rewritten sub-queries) | LLM call |
| `evaluate_docs` | `query: str, context: List[Document]` | `{sufficient: bool, reason: str, missing_aspects: List[str]}` | LLM call |

### Tool Design Notes

- `retrieve_arxiv` is lightweight: metadata + abstract only, no PDF download.
- `query_rewrite` supports sub-question decomposition for complex queries.
- `evaluate_docs` is the key differentiator of Agentic RAG: the agent decides whether to continue retrieving. `missing_aspects` feeds directly into re-planning.
- `search_web` supplements background knowledge beyond papers (concepts, recent news).

## Self-Reflection (Reflexion Pattern)

### Three Dimensions

1. **Citation Faithfulness**: Every `[arxiv:ID]` in the answer must exist in the retrieved context. No ungrounded claims.
2. **Completeness**: Does the answer fully address the user's question? Any missing aspects?
3. **Logical Consistency**: No internal contradictions. Coherent reasoning chain.

### Reflection Output Schema

```python
class ReflectionResult(TypedDict):
    passed: bool
    citation_ok: bool
    completeness_ok: bool
    logic_ok: bool
    issues: list[str]                    # Specific problems found
    fix_strategy: Optional[str]          # "re_retrieve" | "re_generate" | None
```

### Fix Strategies

- `re_retrieve`: Reflection found missing aspects → re_planner generates supplementary retrieval steps → executor runs them → re-synthesize.
- `re_generate`: Reflection found logic/citation issues but context is sufficient → re-run synthesis with `issues` as additional constraints.
- `reflection_count >= 2`: Force output current best answer + append disclaimer about limitations.

## SSE Streaming Protocol

### Endpoint

```
POST /chat/stream
Content-Type: application/json
Accept: text/event-stream
```

### Event Types

```
event: intent
data: {"type": "comparison", "complexity": "complex", "entities": ["BERT", "GPT"]}

event: plan
data: {"steps": [{"action": "query_rewrite", "reason": "..."}, ...], "total_steps": 5}

event: step_start
data: {"index": 0, "action": "query_rewrite", "reason": "分解为子问题"}

event: step_done
data: {"index": 0, "action": "query_rewrite", "output_summary": "生成 2 个子查询", "duration_ms": 820}

event: reflection
data: {"pass": true, "citation_ok": true, "completeness_ok": true, "logic_ok": true}

event: re_plan
data: {"reason": "缺少 GPT-4 相关资料", "new_steps": [...]}

event: token
data: {"t": "基于"}

event: sources
data: {"sources": [{paper_id, title, authors, year, ...}]}

event: done
data: {"total_ms": 4200, "steps_count": 5, "reflections": 1}

event: error
data: {"message": "..."}
```

### Design Notes

- `step_start` / `step_done` are paired; frontend shows loading → complete transition.
- `token` events only fire during `reasoning_synthesis` (streaming final answer).
- `reflection` with `pass=false` is followed by `re_plan` event.
- Synchronous `/chat` endpoint also runs agent internally, returns `ChatResponse` directly.

## Frontend: Claude-Style Redesign

### Design Language

- Warm white background (`#FAF9F7`), not pure black text (`#1A1A1A`)
- Borderless cards with subtle shadow + spacing for hierarchy
- Rounded but restrained (`border-radius: 12px`)
- System font stack (SF Pro / Inter)
- Minimal animations: only opacity + translateY transitions

### Tech Stack

| Current | New | Reason |
|---------|-----|--------|
| Naive UI (dark) | Tailwind CSS + Headless UI | Fine-grained control for Claude aesthetic |
| Dark theme | Light primary (dark optional) | Claude's design language is warm-white |
| Three-column | Conversation flow + collapsible sidebar | Modern, focuses on dialogue |
| marked.js | markdown-it + custom citation plugin | Need custom citation rendering |
| Vue 3 + Vite | Vue 3 + Vite (keep) | No reason to change |
| Pinia | Pinia (keep) | Works well |
| Axios | Axios (sync only) + native fetch (SSE) | SSE needs ReadableStream |

### Layout

```
┌──────────────────────────────────────────────────────────────┐
│  Header: PaperRAG logo + New Chat + Settings                  │
├──────────────────────────────────────────────────────────────┤
│  ┌── Sidebar (collapsible) ──┐  ┌── Main Chat Area ───────┐ │
│  │  Recent Conversations      │  │                          │ │
│  │  Paper Library (browse)    │  │  [Message flow]          │ │
│  │                            │  │  User bubble             │ │
│  │                            │  │  ThinkingCard            │ │
│  │                            │  │  Assistant bubble        │ │
│  │                            │  │                          │ │
│  │                            │  │  ┌────────────────────┐  │ │
│  │                            │  │  │ Input Area         │  │ │
│  └────────────────────────────┘  │  └────────────────────┘  │ │
│                                  └──────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

### ThinkingCard Component

The core UX innovation. Shows agent reasoning transparently, similar to Claude's thinking display:

```
┌──────────────────────────────────────────────────────────┐
│  ⚡ Agent 正在思考...                          [展开 ▾]   │
│─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │
│  ● 意图分析                              820ms  ✓       │
│    识别为对比类问题，涉及 BERT 和 GPT                     │
│                                                           │
│  ● 查询改写                              650ms  ✓       │
│    → "BERT masked language model pretraining"            │
│    → "GPT autoregressive pretraining strategy"           │
│                                                           │
│  ● 本地检索 (子查询 1)                    340ms  ✓       │
│    找到 6 个相关片段，最高相似度 0.92                     │
│                                                           │
│  ◐ 充分性评估                             ...   ⟳       │
│    判断检索结果是否足够回答问题                           │
└──────────────────────────────────────────────────────────┘
```

Step status indicators:
- `◐ ... ⟳` — Running (subtle spin animation)
- `● ... ✓` — Complete
- `● ... ✗` — Failed / needs supplement

### Citation Rendering

Inline `[arxiv:2301.xxxxx]` → numbered pill badge `[¹]`. Hover shows popover:

```
┌──────────────────────────────────────────┐
│  Attention Is All You Need               │
│  Vaswani et al. · 2017 · cs.CL          │
│  Score: 0.92 · Page 3                    │
│  ─────────────────────────────────       │
│  "The Transformer architecture relies    │
│  entirely on attention mechanisms..."    │
│  [View on arXiv ↗]                       │
└──────────────────────────────────────────┘
```

### CSS Variables

```css
:root {
  --bg-primary: #FAF9F7;
  --bg-secondary: #F5F4F0;
  --bg-card: #FFFFFF;
  --text-primary: #1A1A1A;
  --text-secondary: #6B6560;
  --text-tertiary: #9B9590;
  --accent: #D97706;
  --accent-light: #FEF3C7;
  --border: #E8E5E0;
  --shadow: 0 1px 3px rgba(0,0,0,0.04);
  --radius: 12px;
  --radius-sm: 8px;
  --font-sans: -apple-system, 'Inter', sans-serif;
  --font-mono: 'JetBrains Mono', monospace;
}
```

## Backend File Structure

```
backend/
├── app/
│   ├── main.py                      # FastAPI app entry
│   ├── core/
│   │   ├── config.py                # Settings (extended)
│   │   ├── context.py               # Request context
│   │   └── observability.py         # JSON structured logging
│   │
│   ├── agent/                       # Agent core (NEW)
│   │   ├── __init__.py
│   │   ├── graph.py                 # LangGraph StateGraph definition + compile
│   │   ├── state.py                 # AgentState, StepSpec, StepTrace types
│   │   ├── nodes/
│   │   │   ├── intent.py            # intent_node
│   │   │   ├── planner.py           # planner_node + re_planner
│   │   │   ├── executor.py          # executor_node (dispatches to tools)
│   │   │   ├── synthesis.py         # reasoning_synthesis (streaming)
│   │   │   ├── reflection.py        # self_reflection
│   │   │   └── final_answer.py      # Format output + citation building
│   │   └── prompts/
│   │       ├── intent.py
│   │       ├── planner.py
│   │       ├── synthesis.py
│   │       └── reflection.py
│   │
│   ├── tools/                       # Tool layer (NEW)
│   │   ├── __init__.py
│   │   ├── retrieve_local.py        # Vector + BM25 hybrid search
│   │   ├── retrieve_arxiv.py        # arXiv API real-time search
│   │   ├── search_web.py            # Tavily/SerpAPI
│   │   ├── paper_detail.py          # Paper metadata query
│   │   ├── paper_chunks.py          # Paper chunks retrieval
│   │   ├── query_rewrite.py         # LLM query rewriting
│   │   └── evaluate_docs.py         # LLM sufficiency evaluation
│   │
│   ├── services/
│   │   ├── retriever.py             # Core retrieval engine (called by tools)
│   │   └── ingest.py                # PDF ingestion pipeline (kept)
│   │
│   ├── routers/
│   │   ├── chat.py                  # /chat + /chat/stream
│   │   ├── papers.py                # /papers
│   │   ├── ingest.py                # /ingest
│   │   └── upload.py                # /upload
│   │
│   ├── db/
│   │   ├── mysql.py                 # SQLAlchemy engine
│   │   └── qdrant.py                # Qdrant client + embedding
│   │
│   ├── models/
│   │   ├── paper.py                 # Paper + Chunk ORM
│   │   └── chat_history.py          # Chat history table
│   │
│   ├── schemas/
│   │   └── chat.py                  # Request/Response + SSE event types
│   │
│   └── middleware/
│       └── request_context.py       # x-request-id
│
├── scripts/
│   ├── download_arxiv.py            # Kept
│   ├── ingest.py                    # Kept
│   └── rebuild_vectors.py           # Kept
│
├── requirements.txt                  # Updated deps
└── Dockerfile
```

## Frontend File Structure

```
frontend/src/
├── main.ts
├── App.vue
├── styles/
│   ├── base.css                     # CSS reset + variables
│   └── tailwind.css
├── layouts/
│   └── ChatLayout.vue
├── views/
│   └── ChatView.vue
├── components/
│   ├── chat/
│   │   ├── MessageList.vue
│   │   ├── UserBubble.vue
│   │   ├── AssistantBubble.vue
│   │   ├── ThinkingCard.vue         # Core component
│   │   ├── StepIndicator.vue
│   │   └── InputArea.vue
│   ├── citation/
│   │   ├── CitationPill.vue
│   │   └── CitationPopover.vue
│   ├── sidebar/
│   │   ├── ConversationList.vue
│   │   └── PaperBrowser.vue
│   └── common/
│       ├── IconButton.vue
│       └── LoadingDots.vue
├── composables/
│   ├── useSSE.ts                    # SSE connection + event parsing
│   ├── useChat.ts                   # Chat logic (send, receive, state)
│   └── useThinking.ts              # Agent step tracking
├── stores/
│   ├── chat.ts                      # Messages + sessions
│   └── papers.ts                    # Paper list
├── api/
│   └── client.ts                    # Axios (sync only)
├── types/
│   └── index.ts                     # All TypeScript types
└── utils/
    ├── markdown.ts                  # markdown-it + citation plugin
    └── colors.ts                    # Claude palette constants
```

## Configuration

### New `.env` Keys

```bash
# Agent
AGENT_MAX_PLAN_STEPS=7              # Max steps planner can generate
AGENT_MAX_REFLECTIONS=2             # Max reflection loops before force-output

# Tools - External APIs
TAVILY_API_KEY=                     # Web search
ARXIV_MAX_RESULTS=5                 # arXiv per-query limit

# Models (optional: use separate models for different nodes)
PLANNER_MODEL=                      # Defaults to LLM_MODEL if empty
REFLECTION_MODEL=                   # Defaults to LLM_MODEL if empty

# Existing (kept)
LLM_API_KEY=
LLM_API_BASE=
LLM_MODEL=
EMBEDDING_API_KEY=
EMBEDDING_API_BASE=
EMBEDDING_MODEL=
MYSQL_URL=
QDRANT_URL=
QDRANT_COLLECTION=

# Retrieval (kept)
RETRIEVAL_K=16
HYBRID_ALPHA=0.72
HYBRID_OVERSAMPLE=2.5

# Cache (kept)
CACHE_RETRIEVAL_ENABLED=true
CACHE_RETRIEVAL_TTL_SEC=180
CACHE_EMBEDDING_ENABLED=true
```

## Migration Strategy

**Full rewrite.** No incremental migration.

### What Is Preserved

- Docker Compose (MySQL + Qdrant containers)
- Database schema (Paper, Chunk, ChatHistory tables)
- `.env` configuration pattern
- `scripts/` (download_arxiv, ingest, rebuild_vectors)
- Existing ingested paper data in MySQL + Qdrant
- `services/retriever.py` core logic (adapted into new tool)
- `services/ingest.py` pipeline
- `db/` layer (mysql.py, qdrant.py)

### What Is Deleted and Rewritten

- `services/generator.py` → replaced by `agent/nodes/synthesis.py`
- `services/agent.py` → replaced by `agent/graph.py`
- `services/tools.py` → replaced by `tools/` module
- `routers/chat.py` → rewritten for new SSE protocol
- Entire `frontend/src/` → full rewrite in Claude style

### Dependencies Update

New Python packages:
- `langgraph` (already used, keep)
- `tavily-python` (web search)
- `arxiv` (already used, keep)

New Frontend packages:
- `tailwindcss` + `@headlessui/vue` (replaces Naive UI)
- `markdown-it` (replaces marked)
- Remove: `naive-ui`

## Evaluation Compatibility

The synchronous `/chat` endpoint returns the same `ChatResponse` schema:

```python
class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    used_chunks: int
    # New fields (backwards-compatible additions):
    step_traces: Optional[list[StepTrace]] = None
    reflection_result: Optional[dict] = None
```

The eval framework (`eval/run_eval.py`) continues to call `/chat` synchronously. New metrics can leverage `step_traces` for deeper analysis.
