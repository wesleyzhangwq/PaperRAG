# PaperRAG Agent Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade PaperRAG from a fixed RAG pipeline to a tool-using LangGraph agent with streaming SSE, multi-turn conversation memory, and LLM-as-Judge evaluation with standard RAGAS-aligned metrics.

**Architecture:** Four incremental phases, each producing a working commit. Phase 1 adds session-based chat history to MySQL. Phase 2 adds SSE streaming via a new `/chat/stream` endpoint. Phase 3 replaces the fixed pipeline with a LangGraph ReAct agent that autonomously selects tools. Phase 4 upgrades the eval framework with LLM judge scoring and standard retrieval metrics (NDCG@5, Precision@5, Context Precision).

**Tech Stack:** Python 3.11+, FastAPI, LangGraph, LangChain, SQLAlchemy 2.0, MySQL 8, Qdrant, Vue 3 + Pinia + Naive UI, pytest

---

## File Map

### Phase 1: Multi-Turn Conversation
| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `backend/app/models/chat_history.py` | ChatHistory ORM model |
| Modify | `backend/app/db/mysql.py:20-23` | Import ChatHistory model for table creation |
| Modify | `backend/app/core/config.py:44-51` | Add `chat_history_window` setting |
| Modify | `backend/app/schemas/chat.py:16-21` | session_id already exists; no change needed |
| Modify | `backend/app/services/generator.py:48-51,134-191` | Load history, inject into prompt, persist after response |
| Modify | `frontend/src/stores/chat.ts` | Add sessionId, pass to API, add newConversation action |
| Modify | `frontend/src/api/client.ts:48-51` | session_id already passed; no change needed |
| Modify | `frontend/src/components/ChatWindow.vue:70-79` | Add "New Conversation" button |

### Phase 2: Streaming SSE
| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `backend/app/services/generator.py` | Add `run_chat_stream()` generator function |
| Modify | `backend/app/routers/chat.py` | Add `POST /chat/stream` SSE endpoint |
| Modify | `frontend/src/api/client.ts` | Add `chatStream()` using fetch + ReadableStream |
| Modify | `frontend/src/stores/chat.ts` | Switch `ask()` to use stream, add streaming state |
| Modify | `frontend/src/components/ChatWindow.vue` | Show streaming indicator |

### Phase 3: LangGraph Agent + Tools
| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `backend/app/services/agent.py` | LangGraph StateGraph, tool definitions, agent state |
| Create | `backend/app/services/tools.py` | Tool implementations (search, filter, detail, compare) |
| Modify | `backend/app/routers/chat.py` | Add `mode` param, route to agent or pipeline |
| Modify | `backend/app/services/generator.py` | Add `run_agent_chat()` + `run_agent_chat_stream()` |
| Modify | `backend/requirements.txt` | Add `langgraph` dependency |

### Phase 4: LLM-as-Judge + Metrics Upgrade
| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `eval/judge.py` | Judge LLM wrapper, structured scoring |
| Create | `eval/metrics.py` | NDCG@5, Precision@5, Context Precision calculators |
| Modify | `eval/run_eval.py` | Integrate judge + new metrics, update CSV schema |
| Modify | `backend/app/core/config.py` | Add judge LLM config keys |
| Modify | `.env.example` | Add judge config entries |

---

## Task 1: ChatHistory ORM Model

**Files:**
- Create: `backend/app/models/chat_history.py`
- Modify: `backend/app/db/mysql.py:20-23`
- Modify: `backend/app/core/config.py:44-51`

- [ ] **Step 1: Create ChatHistory model**

```python
# backend/app/models/chat_history.py
"""SQLAlchemy ORM model: ChatHistory."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Enum, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.mysql import Base


class ChatHistory(Base):
    __tablename__ = "chat_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    role: Mapped[str] = mapped_column(Enum("user", "assistant", name="chat_role"), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
```

- [ ] **Step 2: Register ChatHistory in init_db**

In `backend/app/db/mysql.py`, change line 22:

```python
def init_db() -> None:
    """Import models and create all tables."""
    from app.models import paper  # noqa: F401 register tables
    from app.models import chat_history  # noqa: F401 register tables
    Base.metadata.create_all(bind=engine)
```

- [ ] **Step 3: Add chat_history_window to Settings**

In `backend/app/core/config.py`, add after `final_context_k: int = 3` (line 51):

```python
    # --- Chat history ---
    chat_history_window: int = 10
```

- [ ] **Step 4: Add CHAT_HISTORY_WINDOW to .env.example**

Add after `FINAL_CONTEXT_K=3`:

```
# Chat history (number of recent messages to include as context)
CHAT_HISTORY_WINDOW=10
```

- [ ] **Step 5: Commit**

```bash
git add backend/app/models/chat_history.py backend/app/db/mysql.py backend/app/core/config.py .env.example
git commit -m "feat: add ChatHistory ORM model and config"
```

---

## Task 2: Inject History into Generator

**Files:**
- Modify: `backend/app/services/generator.py:1-191`

- [ ] **Step 1: Add history loading helper**

Add these imports at the top of `backend/app/services/generator.py`:

```python
from app.models.chat_history import ChatHistory
```

Add this function after the `_prompt` definition (after line 51):

```python
def _load_history(db: Session, session_id: str, window: int) -> list[tuple[str, str]]:
    rows = (
        db.query(ChatHistory)
        .filter(ChatHistory.session_id == session_id)
        .order_by(ChatHistory.created_at.desc())
        .limit(window)
        .all()
    )
    return [(r.role, r.content) for r in reversed(rows)]
```

- [ ] **Step 2: Modify run_chat to use history**

Replace the `run_chat` function (lines 134-191) with:

```python
def run_chat(db: Session, req: ChatRequest) -> ChatResponse:
    flt: Optional[ChatFilter] = req.filter
    top_k = req.top_k or settings.retrieval_k
    final_k = req.final_k or settings.final_context_k
    rid = request_id_ctx.get()
    session_id = req.session_id

    docs_scores = retrieve(req.query, flt=flt, top_k=top_k)
    if not docs_scores:
        answer = "参考资料不足以回答该问题（未检索到相关论文片段）。"
        _save_turn(db, session_id, req.query, answer)
        return ChatResponse(answer=answer, sources=[], used_chunks=0)

    trimmed = docs_scores[:final_k]
    trimmed_ids = {(d.metadata or {}).get("paper_id") for d, _ in trimmed}
    context = _format_context([d for d, _ in trimmed])

    history = _load_history(db, session_id, settings.chat_history_window)
    messages: list[tuple[str, str]] = [("system", SYSTEM_PROMPT)]
    for role, content in history:
        messages.append((role, content))
    messages.append(("user", USER_TEMPLATE))
    prompt = ChatPromptTemplate.from_messages(messages)

    llm = _get_llm()
    chain = prompt | llm | StrOutputParser()
    t_llm = time.perf_counter()
    try:
        answer = chain.invoke({"query": req.query, "context": context})
    except Exception:
        log.exception(
            "chat_llm_failed",
            extra={
                "event": "rag.chat",
                "request_id": rid,
                "phase": "llm_error",
                "error_kind": "llm",
                "top_k": top_k,
                "final_k": final_k,
            },
        )
        return ChatResponse(
            answer="模型调用暂时失败，请稍后重试。",
            sources=[],
            used_chunks=len(trimmed),
        )

    log.info(
        "chat_llm_ok",
        extra={
            "event": "rag.chat",
            "request_id": rid,
            "phase": "after_llm",
            "ms": round((time.perf_counter() - t_llm) * 1000, 2),
            "top_k": top_k,
            "final_k": final_k,
        },
    )

    cited_ids = _extract_cited_ids(answer)
    sources = _build_sources(db, docs_scores, cited_ids, trimmed_ids)

    _save_turn(db, session_id, req.query, answer)

    return ChatResponse(answer=answer, sources=sources, used_chunks=len(trimmed))
```

- [ ] **Step 3: Add _save_turn helper**

Add after `_load_history`:

```python
def _save_turn(db: Session, session_id: str, query: str, answer: str) -> None:
    db.add(ChatHistory(session_id=session_id, role="user", content=query))
    db.add(ChatHistory(session_id=session_id, role="assistant", content=answer))
    db.commit()
```

- [ ] **Step 4: Commit**

```bash
git add backend/app/services/generator.py
git commit -m "feat: inject chat history into LLM prompt for multi-turn conversation"
```

---

## Task 3: Frontend Multi-Turn Support

**Files:**
- Modify: `frontend/src/stores/chat.ts`
- Modify: `frontend/src/components/ChatWindow.vue:70-79`

- [ ] **Step 1: Add sessionId to chat store**

Replace the full contents of `frontend/src/stores/chat.ts`:

```typescript
import { defineStore } from 'pinia'
import { ref } from 'vue'
import { chat, type ChatFilter, type Source } from '../api/client'

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  used_chunks?: number
  created_at: number
}

export const useChatStore = defineStore('chat', () => {
  const messages = ref<ChatMessage[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)
  const currentSources = ref<Source[]>([])
  const sessionId = ref(crypto.randomUUID())

  async function ask(query: string, filter?: ChatFilter) {
    if (!query.trim()) return
    error.value = null
    messages.value.push({
      id: crypto.randomUUID(),
      role: 'user',
      content: query,
      created_at: Date.now(),
    })
    loading.value = true
    try {
      const resp = await chat(query, filter, sessionId.value)
      messages.value.push({
        id: crypto.randomUUID(),
        role: 'assistant',
        content: resp.answer,
        sources: resp.sources,
        used_chunks: resp.used_chunks,
        created_at: Date.now(),
      })
      currentSources.value = resp.sources
    } catch (e: any) {
      error.value = e?.response?.data?.detail ?? e?.message ?? 'unknown error'
      messages.value.push({
        id: crypto.randomUUID(),
        role: 'assistant',
        content: `请求失败：${error.value}`,
        created_at: Date.now(),
      })
    } finally {
      loading.value = false
    }
  }

  function newConversation() {
    messages.value = []
    currentSources.value = []
    error.value = null
    sessionId.value = crypto.randomUUID()
  }

  return { messages, loading, error, currentSources, sessionId, ask, newConversation }
})
```

- [ ] **Step 2: Add "New Conversation" button to ChatWindow**

In `frontend/src/components/ChatWindow.vue`, replace the actions div (lines 72-79):

```html
      <div class="actions">
        <NUpload :custom-request="handleUpload" accept=".pdf" :show-file-list="false">
          <NUploadTrigger abstract>
            <NButton size="small" secondary>上传 PDF</NButton>
          </NUploadTrigger>
        </NUpload>
        <NButton size="small" quaternary @click="chatStore.newConversation" :disabled="!chatStore.messages.length">新对话</NButton>
      </div>
```

- [ ] **Step 3: Remove the old `clear` call**

In `frontend/src/components/ChatWindow.vue`, the old "清空" button called `chatStore.clear`. The `clear` function in the store is now replaced by `newConversation`. Remove the line referencing `year_max` in the `send()` function (line 35) if it still exists:

```typescript
  const filter = {
    category: papersStore.filters.category ?? undefined,
    year_min: papersStore.filters.year_min ?? undefined,
  }
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/stores/chat.ts frontend/src/components/ChatWindow.vue
git commit -m "feat: frontend multi-turn conversation with session management"
```

---

## Task 4: Streaming SSE Backend

**Files:**
- Modify: `backend/app/services/generator.py`
- Modify: `backend/app/routers/chat.py`

- [ ] **Step 1: Add run_chat_stream generator to generator.py**

Add these imports at the top of `backend/app/services/generator.py`:

```python
from collections.abc import Generator
```

Add this function at the end of the file:

```python
def run_chat_stream(
    db: Session, req: ChatRequest
) -> Generator[dict, None, None]:
    flt: Optional[ChatFilter] = req.filter
    top_k = req.top_k or settings.retrieval_k
    final_k = req.final_k or settings.final_context_k
    rid = request_id_ctx.get()
    session_id = req.session_id

    docs_scores = retrieve(req.query, flt=flt, top_k=top_k)
    if not docs_scores:
        answer = "参考资料不足以回答该问题（未检索到相关论文片段）。"
        _save_turn(db, session_id, req.query, answer)
        yield {"event": "token", "data": {"t": answer}}
        yield {"event": "sources", "data": {"sources": []}}
        yield {"event": "done", "data": {}}
        return

    trimmed = docs_scores[:final_k]
    trimmed_ids = {(d.metadata or {}).get("paper_id") for d, _ in trimmed}
    context = _format_context([d for d, _ in trimmed])

    history = _load_history(db, session_id, settings.chat_history_window)
    messages: list[tuple[str, str]] = [("system", SYSTEM_PROMPT)]
    for role, content in history:
        messages.append((role, content))
    messages.append(("user", USER_TEMPLATE))
    prompt = ChatPromptTemplate.from_messages(messages)

    llm = _get_llm()
    chain = prompt | llm

    full_answer = ""
    t_llm = time.perf_counter()
    try:
        for chunk in chain.stream({"query": req.query, "context": context}):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            if token:
                full_answer += token
                yield {"event": "token", "data": {"t": token}}
    except Exception:
        log.exception(
            "chat_stream_failed",
            extra={
                "event": "rag.chat",
                "request_id": rid,
                "phase": "llm_error",
                "error_kind": "llm",
                "top_k": top_k,
                "final_k": final_k,
            },
        )
        yield {"event": "token", "data": {"t": "模型调用暂时失败，请稍后重试。"}}
        yield {"event": "sources", "data": {"sources": []}}
        yield {"event": "done", "data": {}}
        return

    log.info(
        "chat_stream_ok",
        extra={
            "event": "rag.chat",
            "request_id": rid,
            "phase": "after_llm",
            "ms": round((time.perf_counter() - t_llm) * 1000, 2),
            "top_k": top_k,
            "final_k": final_k,
        },
    )

    cited_ids = _extract_cited_ids(full_answer)
    sources = _build_sources(db, docs_scores, cited_ids, trimmed_ids)

    _save_turn(db, session_id, req.query, full_answer)

    sources_data = [s.model_dump() for s in sources]
    yield {"event": "sources", "data": {"sources": sources_data}}
    yield {"event": "done", "data": {}}
```

- [ ] **Step 2: Add SSE endpoint to chat router**

Replace `backend/app/routers/chat.py` with:

```python
from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db.mysql import get_db
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.generator import run_chat, run_chat_stream

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    return run_chat(db, req)


@router.post("/stream")
def chat_stream(req: ChatRequest, db: Session = Depends(get_db)):
    def event_generator():
        for msg in run_chat_stream(db, req):
            event = msg["event"]
            data = json.dumps(msg["data"], ensure_ascii=False)
            yield f"event: {event}\ndata: {data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

- [ ] **Step 3: Commit**

```bash
git add backend/app/services/generator.py backend/app/routers/chat.py
git commit -m "feat: add streaming SSE endpoint POST /chat/stream"
```

---

## Task 5: Frontend Streaming

**Files:**
- Modify: `frontend/src/api/client.ts`
- Modify: `frontend/src/stores/chat.ts`
- Modify: `frontend/src/components/ChatWindow.vue`

- [ ] **Step 1: Add chatStream to api/client.ts**

Add at the end of `frontend/src/api/client.ts`, before the closing:

```typescript
export async function chatStream(
  query: string,
  filter?: ChatFilter,
  session_id = 'default',
  onToken?: (token: string) => void,
  onSources?: (sources: Source[]) => void,
  onDone?: () => void,
  onError?: (err: Error) => void,
): Promise<void> {
  const resp = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, filter, session_id }),
  })

  if (!resp.ok || !resp.body) {
    onError?.(new Error(`HTTP ${resp.status}`))
    return
  }

  const reader = resp.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''

    let currentEvent = ''
    for (const line of lines) {
      if (line.startsWith('event: ')) {
        currentEvent = line.slice(7)
      } else if (line.startsWith('data: ')) {
        const data = JSON.parse(line.slice(6))
        if (currentEvent === 'token') onToken?.(data.t)
        else if (currentEvent === 'sources') onSources?.(data.sources)
        else if (currentEvent === 'done') onDone?.()
        currentEvent = ''
      }
    }
  }
}
```

- [ ] **Step 2: Switch chat store to use streaming**

Replace the `ask` function in `frontend/src/stores/chat.ts` with:

```typescript
  const streaming = ref(false)

  async function ask(query: string, filter?: ChatFilter) {
    if (!query.trim()) return
    error.value = null
    messages.value.push({
      id: crypto.randomUUID(),
      role: 'user',
      content: query,
      created_at: Date.now(),
    })
    loading.value = true

    const assistantId = crypto.randomUUID()
    messages.value.push({
      id: assistantId,
      role: 'assistant',
      content: '',
      created_at: Date.now(),
    })
    const assistantMsg = messages.value[messages.value.length - 1]

    try {
      await chatStream(
        query,
        filter,
        sessionId.value,
        (token) => {
          if (!streaming.value) {
            streaming.value = true
            loading.value = false
          }
          assistantMsg.content += token
        },
        (sources) => {
          assistantMsg.sources = sources
          currentSources.value = sources
        },
        () => {
          streaming.value = false
        },
        (err) => {
          error.value = err.message
          assistantMsg.content = `请求失败：${err.message}`
          streaming.value = false
          loading.value = false
        },
      )
    } catch (e: any) {
      error.value = e?.message ?? 'unknown error'
      assistantMsg.content = `请求失败：${error.value}`
    } finally {
      loading.value = false
      streaming.value = false
    }
  }
```

Update the import at the top:

```typescript
import { chatStream, type ChatFilter, type Source } from '../api/client'
```

Add `streaming` to the return statement:

```typescript
  return { messages, loading, streaming, error, currentSources, sessionId, ask, newConversation }
```

- [ ] **Step 3: Update ChatWindow to show streaming indicator**

In `frontend/src/components/ChatWindow.vue`, change the NSpin line (line 71):

```html
      <NSpin v-if="chatStore.loading || chatStore.streaming" size="small" />
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/api/client.ts frontend/src/stores/chat.ts frontend/src/components/ChatWindow.vue
git commit -m "feat: frontend streaming SSE with incremental token rendering"
```

---

## Task 6: Add LangGraph Dependency

**Files:**
- Modify: `backend/requirements.txt`

- [ ] **Step 1: Add langgraph to requirements**

Add after the LangChain stack section in `backend/requirements.txt`:

```
# ===== Agent orchestration =====
langgraph>=0.2,<0.4
```

- [ ] **Step 2: Commit**

```bash
git add backend/requirements.txt
git commit -m "deps: add langgraph for agent orchestration"
```

---

## Task 7: Agent Tools

**Files:**
- Create: `backend/app/services/tools.py`

- [ ] **Step 1: Create tool implementations**

```python
# backend/app/services/tools.py
"""Agent tools for PaperRAG — each wraps an existing service function."""
from __future__ import annotations

from langchain_core.tools import tool
from sqlalchemy.orm import Session

from app.models.paper import Paper
from app.schemas.chat import ChatFilter
from app.services.retriever import retrieve


def _format_chunks(docs_scores: list) -> str:
    parts = []
    for d, score in docs_scores:
        md = d.metadata or {}
        parts.append(
            f"[arxiv:{md.get('paper_id','?')} | score={score:.3f} | "
            f"page={md.get('page_num','?')}]\n{d.page_content[:500]}"
        )
    return "\n\n---\n\n".join(parts) if parts else "No results found."


@tool
def search_papers(query: str, top_k: int = 8) -> str:
    """Search academic papers using hybrid vector + BM25 retrieval.
    Returns ranked chunks with paper IDs, scores, and text snippets.
    Use this when the user asks a question that requires finding relevant papers."""
    docs_scores = retrieve(query, top_k=top_k)
    return _format_chunks(docs_scores)


@tool
def filter_papers(query: str, category: str = "", year_min: int = 0, year_max: int = 0) -> str:
    """Search papers with metadata filters applied.
    Use this when the user wants results from a specific category (e.g. 'cs.CL')
    or year range. Pass category as arXiv category string, year_min/year_max as integers (0 means no limit)."""
    flt = ChatFilter(
        category=category or None,
        year_min=year_min if year_min > 0 else None,
        year_max=year_max if year_max > 0 else None,
    )
    docs_scores = retrieve(query, flt=flt, top_k=8)
    return _format_chunks(docs_scores)


def get_paper_detail(db: Session, paper_id: str) -> str:
    """Get full metadata for a specific paper by its arxiv paper_id.
    Returns title, authors, year, categories, abstract."""
    paper = db.query(Paper).filter(Paper.paper_id == paper_id).one_or_none()
    if paper is None:
        return f"Paper {paper_id} not found in database."
    authors = ", ".join(paper.authors or [])
    categories = ", ".join(paper.categories or [])
    return (
        f"paper_id: {paper.paper_id}\n"
        f"title: {paper.title}\n"
        f"authors: {authors}\n"
        f"year: {paper.year}\n"
        f"categories: {categories}\n"
        f"abstract: {paper.abstract or 'N/A'}"
    )


def compare_papers(db: Session, paper_ids: list[str]) -> str:
    """Compare multiple papers side by side.
    Returns a formatted comparison of titles, authors, years, and abstracts."""
    parts = []
    for pid in paper_ids[:5]:
        paper = db.query(Paper).filter(Paper.paper_id == pid).one_or_none()
        if paper is None:
            parts.append(f"## {pid}\nNot found in database.")
            continue
        authors = ", ".join((paper.authors or [])[:5])
        parts.append(
            f"## {paper.paper_id}\n"
            f"title: {paper.title}\n"
            f"authors: {authors}\n"
            f"year: {paper.year}\n"
            f"category: {paper.primary_category}\n"
            f"abstract: {(paper.abstract or 'N/A')[:600]}"
        )
    return "\n\n---\n\n".join(parts) if parts else "No papers found."
```

- [ ] **Step 2: Commit**

```bash
git add backend/app/services/tools.py
git commit -m "feat: add agent tool implementations (search, filter, detail, compare)"
```

---

## Task 8: LangGraph Agent

**Files:**
- Create: `backend/app/services/agent.py`

- [ ] **Step 1: Create agent module with StateGraph**

```python
# backend/app/services/agent.py
"""LangGraph ReAct agent for PaperRAG."""
from __future__ import annotations

import logging
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.services.tools import (
    compare_papers,
    filter_papers,
    get_paper_detail,
    search_papers,
)

log = logging.getLogger("app.services.agent")

AGENT_SYSTEM_PROMPT = """你是一个严谨的学术论文问答助手，可以使用工具来搜索和分析论文。

你有以下工具可用：
1. search_papers: 搜索相关论文（使用向量+BM25混合检索）
2. filter_papers: 按类别或年份范围过滤搜索论文
3. get_paper_detail: 获取某篇论文的完整元数据（标题、摘要、作者等）
4. compare_papers: 并排对比多篇论文

硬性规则：
1. 必须先使用工具搜索论文，再基于搜索结果回答。禁止凭空回答。
2. 每条论据/结论末尾必须追加 [arxiv:PAPER_ID] 引用。
3. 禁止编造 paper_id；只能使用工具返回的 paper_id。
4. 若工具返回结果不足以回答，如实说"参考资料不足以回答该问题"。
5. 输出使用简洁的 Markdown，中文作答（除非用户用英文提问）。"""

MAX_STEPS = 5


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def _get_agent_llm() -> ChatOpenAI:
    s = get_settings()
    if not s.llm_api_key:
        raise RuntimeError("Missing LLM_API_KEY in .env")
    return ChatOpenAI(
        model=s.llm_model,
        base_url=s.llm_api_base,
        api_key=s.llm_api_key,
        temperature=0.2,
        max_retries=max(0, s.llm_max_retries),
    )


def build_tools(db: Session) -> list:
    @tool
    def get_paper_detail_tool(paper_id: str) -> str:
        """Get full metadata for a specific paper by its arxiv paper_id.
        Returns title, authors, year, categories, abstract."""
        return get_paper_detail(db, paper_id)

    @tool
    def compare_papers_tool(paper_ids: list[str]) -> str:
        """Compare multiple papers side by side.
        Returns a formatted comparison of titles, authors, years, and abstracts."""
        return compare_papers(db, paper_ids)

    return [search_papers, filter_papers, get_paper_detail_tool, compare_papers_tool]


def build_graph(db: Session) -> StateGraph:
    tools = build_tools(db)
    tool_map = {t.name: t for t in tools}
    llm = _get_agent_llm().bind_tools(tools)

    def agent_think(state: AgentState) -> AgentState:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    def execute_tool(state: AgentState) -> AgentState:
        last = state["messages"][-1]
        results = []
        for call in last.tool_calls:
            fn = tool_map.get(call["name"])
            if fn is None:
                results.append(ToolMessage(
                    content=f"Unknown tool: {call['name']}",
                    tool_call_id=call["id"],
                ))
                continue
            try:
                output = fn.invoke(call["args"])
            except Exception as e:
                output = f"Tool error: {type(e).__name__}: {e}"
            results.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
        return {"messages": results}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "execute_tool"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent_think", agent_think)
    graph.add_node("execute_tool", execute_tool)
    graph.set_entry_point("agent_think")
    graph.add_conditional_edges("agent_think", should_continue)
    graph.add_edge("execute_tool", "agent_think")

    return graph.compile()


def run_agent(db: Session, query: str, history: list[tuple[str, str]]) -> str:
    graph = build_graph(db)

    messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)]
    for role, content in history:
        if role == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=query))

    config = {"recursion_limit": MAX_STEPS * 2 + 1}
    result = graph.invoke({"messages": messages}, config=config)

    final = result["messages"][-1]
    return final.content if hasattr(final, "content") else str(final)
```

- [ ] **Step 2: Commit**

```bash
git add backend/app/services/agent.py
git commit -m "feat: add LangGraph ReAct agent with state graph"
```

---

## Task 9: Integrate Agent into Generator and Router

**Files:**
- Modify: `backend/app/services/generator.py`
- Modify: `backend/app/routers/chat.py`

- [ ] **Step 1: Add run_agent_chat to generator.py**

Add this import at the top of `backend/app/services/generator.py`:

```python
from app.services.agent import run_agent
```

Add this function at the end of the file (after `run_chat_stream`):

```python
def run_agent_chat(db: Session, req: ChatRequest) -> ChatResponse:
    rid = request_id_ctx.get()
    session_id = req.session_id
    top_k = req.top_k or settings.retrieval_k

    history = _load_history(db, session_id, settings.chat_history_window)

    t0 = time.perf_counter()
    try:
        answer = run_agent(db, req.query, history)
    except Exception:
        log.exception(
            "agent_chat_failed",
            extra={
                "event": "rag.agent",
                "request_id": rid,
                "phase": "agent_error",
                "error_kind": "agent",
            },
        )
        return ChatResponse(
            answer="Agent 调用暂时失败，请稍后重试。",
            sources=[],
            used_chunks=0,
        )

    log.info(
        "agent_chat_ok",
        extra={
            "event": "rag.agent",
            "request_id": rid,
            "phase": "after_agent",
            "ms": round((time.perf_counter() - t0) * 1000, 2),
        },
    )

    cited_ids = _extract_cited_ids(answer)
    sources: list[Source] = []
    for pid in cited_ids:
        paper = db.query(Paper).filter(Paper.paper_id == pid).one_or_none()
        if paper:
            sources.append(Source(
                paper_id=pid,
                title=paper.title or "",
                authors=paper.authors or [],
                year=paper.year,
                primary_category=paper.primary_category,
                doi=paper.doi,
                arxiv_url=f"https://arxiv.org/abs/{pid}",
            ))

    _save_turn(db, session_id, req.query, answer)

    return ChatResponse(answer=answer, sources=sources, used_chunks=0)
```

- [ ] **Step 2: Update chat router to support mode param**

Replace `backend/app/routers/chat.py`:

```python
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db.mysql import get_db
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.generator import run_agent_chat, run_chat, run_chat_stream

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    db: Session = Depends(get_db),
    mode: str = Query("agent", regex="^(agent|pipeline)$"),
) -> ChatResponse:
    if mode == "pipeline":
        return run_chat(db, req)
    return run_agent_chat(db, req)


@router.post("/stream")
def chat_stream(req: ChatRequest, db: Session = Depends(get_db)):
    def event_generator():
        for msg in run_chat_stream(db, req):
            event = msg["event"]
            data = json.dumps(msg["data"], ensure_ascii=False)
            yield f"event: {event}\ndata: {data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

- [ ] **Step 3: Commit**

```bash
git add backend/app/services/generator.py backend/app/routers/chat.py
git commit -m "feat: integrate LangGraph agent into chat endpoint with mode switching"
```

---

## Task 10: Eval Metrics Module

**Files:**
- Create: `eval/metrics.py`

- [ ] **Step 1: Create metrics module with NDCG@5, Precision@5, Context Precision**

```python
# eval/metrics.py
"""Standard RAG retrieval metrics: NDCG@k, Precision@k, Context Precision."""
from __future__ import annotations

import math


def ndcg_at_k(pred_pids: list[str], expected_pids: set[str], k: int = 5) -> float:
    if not expected_pids:
        return 0.0
    pred = pred_pids[:k]
    dcg = 0.0
    for i, pid in enumerate(pred):
        if pid in expected_pids:
            dcg += 1.0 / math.log2(i + 2)
    ideal_hits = min(len(expected_pids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def precision_at_k(pred_pids: list[str], expected_pids: set[str], k: int = 5) -> float:
    if not expected_pids:
        return 0.0
    pred = pred_pids[:k]
    if not pred:
        return 0.0
    hits = sum(1 for pid in pred if pid in expected_pids)
    return hits / len(pred)


def context_precision(cited_pids: list[str], retrieved_pids: list[str]) -> float:
    if not retrieved_pids:
        return 0.0
    cited_set = set(cited_pids)
    hits = sum(1 for pid in retrieved_pids if pid in cited_set)
    return hits / len(retrieved_pids)
```

- [ ] **Step 2: Commit**

```bash
git add eval/metrics.py
git commit -m "feat: add NDCG@5, Precision@5, Context Precision metric functions"
```

---

## Task 11: LLM-as-Judge

**Files:**
- Create: `eval/judge.py`
- Modify: `backend/app/core/config.py`
- Modify: `.env.example`

- [ ] **Step 1: Add judge config to Settings**

In `backend/app/core/config.py`, add after `llm_max_retries: int = 2` (line 71):

```python
    # --- Judge LLM (for eval, defaults to main LLM) ---
    judge_llm_model: Optional[str] = None
    judge_llm_api_base: Optional[str] = None
    judge_llm_api_key: Optional[str] = None
```

- [ ] **Step 2: Add judge config to .env.example**

Add after the LLM_MAX_RETRIES line:

```
# Judge LLM for evaluation (optional, defaults to main LLM config)
JUDGE_LLM_MODEL=
JUDGE_LLM_API_BASE=
JUDGE_LLM_API_KEY=
```

- [ ] **Step 3: Create judge.py**

```python
# eval/judge.py
"""LLM-as-Judge: score RAG answers on faithfulness, relevance, correctness."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from langchain_openai import ChatOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.core.config import get_settings

JUDGE_PROMPT = """You are an impartial judge evaluating a RAG (Retrieval-Augmented Generation) system's answer quality.

Given:
- **User Query**: {query}
- **Retrieved Context**: {context}
- **System Answer**: {answer}

Score the answer on these three dimensions (1-5 each):

1. **Faithfulness**: Is the answer strictly based on the provided context? Does it avoid fabricating information not present in the context? (5 = perfectly faithful, 1 = mostly fabricated)

2. **Relevance**: Does the answer directly address the user's question? Is it on-topic and useful? (5 = perfectly relevant, 1 = completely off-topic)

3. **Correctness**: Given the context, is the answer factually correct? Are the claims supported by the retrieved chunks? (5 = all claims correct, 1 = mostly incorrect)

Respond with ONLY a JSON object, no other text:
{{"faithfulness": <int 1-5>, "relevance": <int 1-5>, "correctness": <int 1-5>, "reasoning": "<brief explanation>"}}"""


@dataclass
class JudgeScores:
    faithfulness: int
    relevance: int
    correctness: int
    reasoning: str


def _get_judge_llm() -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        model=s.judge_llm_model or s.llm_model,
        base_url=s.judge_llm_api_base or s.llm_api_base,
        api_key=s.judge_llm_api_key or s.llm_api_key,
        temperature=0.0,
        max_retries=2,
    )


def judge_answer(query: str, context: str, answer: str) -> JudgeScores:
    llm = _get_judge_llm()
    prompt = JUDGE_PROMPT.format(query=query, context=context, answer=answer)
    response = llm.invoke(prompt)
    text = response.content.strip()

    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]

    data = json.loads(text)
    return JudgeScores(
        faithfulness=max(1, min(5, int(data.get("faithfulness", 1)))),
        relevance=max(1, min(5, int(data.get("relevance", 1)))),
        correctness=max(1, min(5, int(data.get("correctness", 1)))),
        reasoning=str(data.get("reasoning", "")),
    )
```

- [ ] **Step 4: Commit**

```bash
git add eval/judge.py backend/app/core/config.py .env.example
git commit -m "feat: add LLM-as-Judge evaluation module with configurable judge LLM"
```

---

## Task 12: Upgrade run_eval.py

**Files:**
- Modify: `eval/run_eval.py`

- [ ] **Step 1: Rewrite run_eval.py with new metrics and judge integration**

Replace the entire contents of `eval/run_eval.py`:

```python
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from starlette.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))

from backend.app.main import app  # noqa: E402
from eval.metrics import context_precision, ndcg_at_k, precision_at_k  # noqa: E402

try:
    import tiktoken
except Exception:
    tiktoken = None

CITATION_RE = re.compile(r"\[arxiv:([0-9]{4}\.[0-9]{4,6})\]")


def load_questions(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No questions found in {path}")
    return rows


def first_relevant_rank(pred_pids: list[str], expected_pids: set[str]) -> int | None:
    if not expected_pids:
        return None
    for idx, pid in enumerate(pred_pids, start=1):
        if pid in expected_pids:
            return idx
    return None


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = max(0, int(len(sorted_vals) * p) - 1)
    return float(sorted_vals[idx])


def estimate_tokens(text: str) -> int:
    text = text or ""
    if not text:
        return 0
    if tiktoken is not None:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


def run_eval(questions: list[dict], top_k: int, final_k: int, use_judge: bool = False) -> dict:
    latencies: list[float] = []
    rr_values: list[float] = []
    recall_values: list[float] = []
    ndcg_values: list[float] = []
    precision_values: list[float] = []
    ctx_precision_values: list[float] = []
    tokens_per_req: list[int] = []

    judge_faithfulness: list[int] = []
    judge_relevance: list[int] = []
    judge_correctness: list[int] = []

    judge_fn = None
    if use_judge:
        from eval.judge import judge_answer
        judge_fn = judge_answer

    with TestClient(app) as client:
        for item in questions:
            query = item["query"]
            expected_pids = set(item.get("expected_paper_ids") or [])

            t0 = time.time()
            resp = client.post(
                "/chat",
                json={"query": query, "top_k": top_k, "final_k": final_k},
                params={"mode": "pipeline"},
            )
            latency = time.time() - t0
            latencies.append(latency)

            if resp.status_code != 200:
                rr_values.append(0.0)
                recall_values.append(0.0)
                ndcg_values.append(0.0)
                precision_values.append(0.0)
                tokens_per_req.append(estimate_tokens(query))
                continue

            body = resp.json()
            answer = body.get("answer", "")
            sources = body.get("sources") or []
            pred_pids = [s.get("paper_id") for s in sources if s.get("paper_id")]
            tokens_per_req.append(estimate_tokens(query) + estimate_tokens(answer))

            cited_pids = CITATION_RE.findall(answer)

            if expected_pids:
                rank = first_relevant_rank(pred_pids[:5], expected_pids)
                ndcg_values.append(ndcg_at_k(pred_pids, expected_pids, k=5))
                precision_values.append(precision_at_k(pred_pids, expected_pids, k=5))
                top5_hits = len(set(pred_pids[:5]) & expected_pids)
                recall_values.append(top5_hits / max(1, len(expected_pids)))
                rr_values.append(1.0 / rank if rank is not None else 0.0)

            ctx_precision_values.append(context_precision(cited_pids, pred_pids[:5]))

            if judge_fn is not None:
                context_text = "\n---\n".join(
                    s.get("snippet", "") for s in sources if s.get("snippet")
                )
                try:
                    scores = judge_fn(query, context_text, answer)
                    judge_faithfulness.append(scores.faithfulness)
                    judge_relevance.append(scores.relevance)
                    judge_correctness.append(scores.correctness)
                except Exception as e:
                    print(f"Judge error for query '{query[:50]}': {e}", file=sys.stderr)

    retrieval_labeled = sum(1 for q in questions if (q.get("expected_paper_ids") or []))
    retrieval_den = max(1, retrieval_labeled)

    result = {
        "ndcg_5": round(sum(ndcg_values) / retrieval_den, 4) if ndcg_values else 0.0,
        "precision_5": round(sum(precision_values) / retrieval_den, 4) if precision_values else 0.0,
        "recall_5": round(sum(recall_values) / retrieval_den, 4) if recall_values else 0.0,
        "mrr": round(sum(rr_values) / retrieval_den, 4) if rr_values else 0.0,
        "context_precision": round(statistics.mean(ctx_precision_values), 4) if ctx_precision_values else 0.0,
        "latency_p90": round(percentile(latencies, 0.9), 3),
        "tokens_per_request": round(statistics.mean(tokens_per_req), 2) if tokens_per_req else 0.0,
    }

    if judge_faithfulness:
        result["faithfulness_avg"] = round(statistics.mean(judge_faithfulness), 2)
        result["relevance_avg"] = round(statistics.mean(judge_relevance), 2)
        result["correctness_avg"] = round(statistics.mean(judge_correctness), 2)

    return result


def append_summary(path: Path, row: dict) -> None:
    header = [
        "run_id", "timestamp", "dataset",
        "ndcg_5", "precision_5", "recall_5", "mrr", "context_precision",
        "faithfulness_avg", "relevance_avg", "correctness_avg",
        "latency_p90", "tokens_per_request", "notes",
    ]
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PaperRAG eval and append summary CSV.")
    parser.add_argument("--dataset", type=str, default=str(PROJECT_ROOT / "eval/datasets/questions_v1.jsonl"))
    parser.add_argument("--summary-csv", type=str, default=str(PROJECT_ROOT / "eval/results/summary.csv"))
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--final-k", type=int, default=3)
    parser.add_argument("--judge", action="store_true", help="Enable LLM-as-Judge scoring")
    parser.add_argument("--notes", type=str, default="")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    summary_path = Path(args.summary_csv)
    questions = load_questions(dataset_path)

    metrics = run_eval(
        questions=questions,
        top_k=args.top_k,
        final_k=args.final_k,
        use_judge=args.judge,
    )
    run_id = args.run_id or f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    row = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_path.name,
        **metrics,
        "notes": args.notes,
    }
    append_summary(summary_path, row)

    print(json.dumps(row, ensure_ascii=False, indent=2))
    print(f"Appended summary to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Commit**

```bash
git add eval/run_eval.py
git commit -m "feat: upgrade eval with NDCG@5, Precision@5, Context Precision, LLM-as-Judge"
```

---

## Task 13: Final Verification

- [ ] **Step 1: Check Python imports resolve**

```bash
cd backend && python -c "
from app.models.chat_history import ChatHistory
from app.services.agent import build_graph, run_agent
from app.services.tools import search_papers, filter_papers
from app.services.generator import run_chat, run_chat_stream, run_agent_chat
print('All imports OK')
"
```

- [ ] **Step 2: Check frontend builds**

```bash
cd frontend && npm run build
```

Expected: Build succeeds with no errors.

- [ ] **Step 3: Check eval imports**

```bash
python -c "
from eval.metrics import ndcg_at_k, precision_at_k, context_precision
from eval.judge import JudgeScores
print('Eval imports OK')
"
```

- [ ] **Step 4: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix: resolve any import/build issues from agent upgrade"
```
