# Corpus Overview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a read-only corpus overview card to the empty chat state so users can see what the current Cite Scope corpus contains before asking a question.

**Architecture:** The backend exposes a cheap SQL-only `GET /papers/overview` endpoint that aggregates paper and chunk metadata. The frontend loads the overview in `ChatView`, renders a `CorpusOverviewCard` only when the active conversation has no messages, and lets suggested-question chips fill the input box.

**Tech Stack:** FastAPI, SQLAlchemy, Pydantic, Vue 3, Pinia, TypeScript, Tailwind CSS.

---

### Task 1: Backend Overview Endpoint

**Files:**
- Modify: `backend/app/schemas/chat.py`
- Modify: `backend/app/routers/papers.py`
- Test: `backend/tests/test_papers_overview.py`

- [x] **Step 1: Write failing backend tests**

Create `backend/tests/test_papers_overview.py` with tests that:

```python
def test_papers_overview_returns_corpus_shape():
    resp = client.get("/papers/overview")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_papers"] == 3
    assert data["total_chunks"] == 18
    assert data["year_min"] == 2017
    assert data["year_max"] == 2024
    assert data["topic_buckets"][0]["key"] == "rag_ir_memory"
    assert data["topic_buckets"][0]["representative_papers"][0]["paper_id"] == "2005.11401"

def test_papers_overview_handles_empty_corpus():
    resp = client.get("/papers/overview")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_papers"] == 0
    assert data["topic_buckets"] == []
    assert data["suggested_questions"]
```

- [x] **Step 2: Run tests and verify they fail**

Run:

```bash
source backend/.venv/bin/activate
PYTHONPATH=backend pytest backend/tests/test_papers_overview.py -q
```

Expected: FAIL because `/papers/overview` is not implemented.

- [x] **Step 3: Add schemas and endpoint implementation**

Add Pydantic models:

```python
class CorpusRepresentativePaper(BaseModel):
    paper_id: str
    title: str
    year: Optional[int] = None
    primary_category: str
    arxiv_url: str

class CorpusTopicBucket(BaseModel):
    key: str
    label: str
    description: str
    paper_count: int
    chunk_count: int
    representative_papers: list[CorpusRepresentativePaper] = []

class CorpusOverviewResponse(BaseModel):
    total_papers: int
    total_chunks: int
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    topic_buckets: list[CorpusTopicBucket] = []
    suggested_questions: list[str] = []
    generated_at: datetime
```

Add `GET /papers/overview` before `GET /papers/{paper_id}`.

- [x] **Step 4: Run backend overview tests and full backend tests**

Run:

```bash
PYTHONPATH=backend pytest backend/tests/test_papers_overview.py -q
PYTHONPATH=backend pytest backend/tests -q
```

Expected: all tests pass.

### Task 2: Frontend API and Types

**Files:**
- Modify: `frontend/src/types/index.ts`
- Modify: `frontend/src/api/papers.ts`

- [x] **Step 1: Add TypeScript models**

Add interfaces matching the backend response:

```ts
export interface CorpusOverviewResponse {
  total_papers: number
  total_chunks: number
  year_min: number | null
  year_max: number | null
  topic_buckets: CorpusTopicBucket[]
  suggested_questions: string[]
  generated_at: string
}
```

- [x] **Step 2: Add API function**

Add:

```ts
export async function getCorpusOverview(): Promise<CorpusOverviewResponse> {
  const { data } = await api.get<CorpusOverviewResponse>('/papers/overview')
  return data
}
```

- [x] **Step 3: Run frontend typecheck/build**

Run:

```bash
npm --prefix frontend run build
```

Expected: build passes after all frontend tasks are complete.

### Task 3: Corpus Overview Card

**Files:**
- Create: `frontend/src/components/chat/CorpusOverviewCard.vue`
- Modify: `frontend/src/components/chat/MessageList.vue`

- [x] **Step 1: Create component**

Create a card that accepts:

```ts
overview: CorpusOverviewResponse | null
loading: boolean
error: string
```

It emits:

```ts
ask: [question: string]
```

- [x] **Step 2: Render card only in empty chat**

Update `MessageList.vue`:

```vue
<CorpusOverviewCard
  v-if="messages.length === 0"
  :overview="overview"
  :loading="overviewLoading"
  :error="overviewError"
  @ask="$emit('ask', $event)"
/>
```

- [x] **Step 3: Preserve normal message rendering**

Confirm non-empty messages still render `UserBubble`, `ThinkingCard`, and `AnswerCard`.

### Task 4: ChatView and Input Draft Integration

**Files:**
- Modify: `frontend/src/views/ChatView.vue`
- Modify: `frontend/src/components/chat/InputArea.vue`

- [x] **Step 1: Load overview in ChatView**

Use `onMounted(loadOverview)` and pass overview state into `MessageList`.

- [x] **Step 2: Add input draft support**

`InputArea` receives:

```ts
draft: string
```

It watches `draft`, fills the textarea, and focuses it when changed.

- [x] **Step 3: Wire suggested questions**

In `ChatView`, clicking a suggested question sets `draftQuestion`.

### Task 5: Verification

**Files:**
- No new files unless fixes are needed.

- [x] **Step 1: Backend tests**

Run:

```bash
source backend/.venv/bin/activate
PYTHONPATH=backend pytest backend/tests -q
```

- [x] **Step 2: Frontend build**

Run:

```bash
npm --prefix frontend run build
```

- [x] **Step 3: Runtime smoke**

Restart backend and frontend, then verify:

```bash
curl -sS http://localhost:8000/papers/overview | python3 -m json.tool
curl -I -sS http://localhost:5173/ | head
```

- [x] **Step 4: Browser check**

Open `http://localhost:5173/`, create/select an empty conversation, and verify:

- overview card is visible
- metrics show corpus size
- topic buckets render
- suggested question click fills input
- hiding after send is covered by the `messages.length === 0` render condition; no live LLM send was triggered during smoke verification

**Verification evidence captured during implementation:**

- `PYTHONPATH=backend pytest backend/tests -q` -> `60 passed`
- `npm --prefix frontend run build` -> `vue-tsc --noEmit && vite build` passed
- `GET /papers/overview` against the local DB returned `500` papers, `54372` chunks, `2012-2026`, and 7 visible topic buckets
- Playwright snapshot confirmed overview metrics, topic buckets, representative arXiv links, and suggested question input fill
