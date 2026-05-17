# PaperRAG Agent Upgrade Design

Goal: upgrade PaperRAG from a fixed RAG pipeline to a tool-using agent with streaming, multi-turn memory, and production-grade evaluation. Target resume position: Agent / LLM Application Engineer.

Implementation order: multi-turn conversation -> streaming SSE -> LangGraph agent + tools -> LLM-as-Judge evaluation + metrics upgrade.

## 1. Multi-Turn Conversation + Session Memory

### Data Layer

New MySQL table `chat_history`:

| Column | Type | Notes |
|--------|------|-------|
| id | BIGINT AUTO_INCREMENT PK | |
| session_id | VARCHAR(64) INDEX | frontend-generated UUID |
| role | ENUM('user', 'assistant') | |
| content | TEXT | |
| created_at | DATETIME DEFAULT NOW() | |

### Service Layer

`generator.py` `run_chat()` changes:
- Before building prompt, load latest N messages from `chat_history` where `session_id` matches (N=10, configurable via `CHAT_HISTORY_WINDOW` in Settings).
- Insert history messages into `ChatPromptTemplate` between system and user messages as `("user", msg)` / `("assistant", msg)` pairs.
- After LLM responds, write both the user query and assistant answer to `chat_history`.

### Frontend

- `chat.ts` store generates `sessionId = crypto.randomUUID()` on init.
- Every `chat()` call passes `session_id`.
- New "New Conversation" button resets `sessionId` and clears message list.

### Out of Scope

- Token-level history truncation (use message count limit for now).
- Cross-session long-term memory.

## 2. Streaming SSE

### Backend

New route `POST /chat/stream` returning `StreamingResponse(media_type="text/event-stream")`.

Three SSE event types:
- `event: token` / `data: {"t": "<token>"}` -- one per token.
- `event: sources` / `data: {"sources": [...]}` -- sent once after LLM finishes, carries the citation list.
- `event: done` / `data: {}` -- marks stream end.

`generator.py` adds `run_chat_stream()` using `llm.stream()` instead of `llm.invoke()`, yielding tokens. After agent integration (Section 3), the stream endpoint runs the LangGraph agent with streaming callbacks, emitting tool-call progress as intermediate `event: step` events in addition to tokens.

The existing synchronous `POST /chat` endpoint remains unchanged (eval framework and API consumers depend on it).

### Frontend

- `api/client.ts` adds `chatStream()` using native `fetch` + `ReadableStream` to parse SSE.
- `chat.ts` store's `sendMessage` switches to the stream endpoint, appending tokens incrementally to the last message's `content`.
- On receiving `sources` event, updates `currentSources`.
- Loading state transitions: idle -> loading -> streaming (on first token) -> idle (on done).

### Out of Scope

- WebSocket (SSE is sufficient for unidirectional streaming).
- Token counting / rate limiting.

## 3. LangGraph Agent + Tools

### Agent State Machine

```
START -> agent_think -> [tool_call?]
                         |-- yes -> execute_tool -> agent_think (loop)
                         |-- no  -> final_answer -> END
```

- `agent_think`: LLM analyzes current state (query + history + tool results), decides next action.
- `execute_tool`: runs the selected tool, writes result back to state.
- `final_answer`: LLM generates cited answer.
- `max_steps=5` to prevent infinite loops.

### Tools (4)

| Tool | Input | Behavior | Maps To |
|------|-------|----------|---------|
| `search_papers` | `query: str, top_k: int=8` | Hybrid vector+BM25 retrieval, returns chunk list with scores | `retriever.retrieve()` |
| `filter_papers` | `query: str, category?: str, year_min?: int, year_max?: int` | Run a filtered search: calls `retrieve()` with metadata constraints applied, returns chunk list | `retriever.retrieve()` with `ChatFilter` |
| `get_paper_detail` | `paper_id: str` | Full paper metadata (title, abstract, authors, year, categories) | `db.query(Paper)` |
| `compare_papers` | `paper_ids: list[str]` | Side-by-side comparison of multiple papers' abstracts and key info | New, composite query |

### Architecture Changes

- New file `services/agent.py`: defines LangGraph `StateGraph`, tool definitions, agent state schema.
- `generator.py`: keeps `run_chat()` as non-agent fallback (for eval), adds `run_agent_chat()` that runs the LangGraph graph.
- `routers/chat.py`: defaults to agent path; supports `?mode=pipeline` query param to fall back to the original pipeline.
- `retriever.py`: unchanged, called internally by `search_papers` tool.

### LLM Configuration

- Agent system prompt defines role + available tools + behavior constraints (must cite, no fabrication).
- Tools bound to LLM via `bind_tools()`.

### Out of Scope

- Parallel tool calling (sequential is more controllable).
- Multi-agent communication.
- Explicit planning/reflection nodes (ReAct includes implicit reasoning).

## 4. LLM-as-Judge Evaluation + Metrics Upgrade

### Judge Dimensions (3, each scored 1-5)

| Dimension | Meaning | Input |
|-----------|---------|-------|
| Faithfulness | Is the answer faithful to the retrieved context? No fabrication? | answer + context chunks |
| Answer Relevance | Does the answer address the user's question? | answer + query |
| Answer Correctness | Is the answer factually correct given the context? | answer + query + context chunks |

### Implementation

- New file `eval/judge.py`: wraps judge LLM call.
- Judge prompt requires structured JSON output: `{"faithfulness": 4, "relevance": 5, "correctness": 3, "reasoning": "..."}`.
- Judge uses an independent LLM instance (configurable separately to avoid self-evaluation bias).

### Integration with Eval Framework

- `run_eval.py` adds `--judge` flag; when enabled, each question additionally calls the judge.
- Without `--judge`, behavior is unchanged for fast iteration.

### Configuration

New `.env` keys (optional, defaults to main LLM config):
- `JUDGE_LLM_MODEL`
- `JUDGE_LLM_API_BASE`
- `JUDGE_LLM_API_KEY`

### Metrics Upgrade

#### Retrieval Metrics (no Judge dependency)

| Metric | Description | Status |
|--------|-------------|--------|
| NDCG@5 | Ranking-aware retrieval quality | New |
| Precision@5 | Proportion of relevant docs in top-5 | New |
| Recall@5 | Proportion of relevant docs retrieved | Keep |
| MRR | Mean Reciprocal Rank | Keep |
| Context Precision | Proportion of retrieved chunks actually cited by LLM | New |

#### Generation Metrics (LLM-as-Judge)

| Metric | Description |
|--------|-------------|
| Faithfulness | Judge score avg |
| Answer Relevance | Judge score avg |
| Answer Correctness | Judge score avg (replaces heuristic) |

#### Kept As-Is

| Metric | Description |
|--------|-------------|
| Latency P90 | End-to-end latency |
| Tokens/Request | Cost tracking |

#### Removed

| Metric | Reason |
|--------|--------|
| `hit_at_5` | Subsumed by NDCG@5 and Precision@5 |
| `insufficient_ratio` | Debug-only, move to log output |
| Heuristic `answer_correctness` | Replaced by Judge-based correctness |

### Output Format

`summary.csv` columns:
```
run_id, timestamp, dataset, ndcg_5, precision_5, recall_5, mrr, context_precision,
faithfulness_avg, relevance_avg, correctness_avg, latency_p90, tokens_per_request
```
