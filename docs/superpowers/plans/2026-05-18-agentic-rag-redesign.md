# PaperRAG v2: Agentic RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite PaperRAG as a full Agentic RAG system with adaptive planning, multi-hop retrieval, self-reflection, and a Claude-style frontend with transparent reasoning display.

**Architecture:** LangGraph StateGraph with 7 nodes (intent → planner → executor → synthesis → reflection → re_planner → final_answer), 7 tools, SSE streaming protocol, Vue 3 + Tailwind frontend.

**Tech Stack:** Python 3.11 / FastAPI / LangGraph / Qdrant / MySQL / Vue 3 / Vite / Tailwind CSS / markdown-it

---

## Phase 1: Backend Foundation

### Task 1: Clean Backend Scaffold + Config

**Files:**
- Modify: `backend/app/core/config.py`
- Modify: `backend/requirements.txt`
- Delete: `backend/app/services/generator.py`
- Delete: `backend/app/services/agent.py`
- Delete: `backend/app/services/tools.py`
- Create: `backend/app/agent/__init__.py`
- Create: `backend/app/agent/state.py`
- Create: `backend/app/tools/__init__.py`

- [ ] **Step 1: Delete old agent/generator code**

```bash
rm backend/app/services/generator.py
rm backend/app/services/agent.py
rm backend/app/services/tools.py
```

- [ ] **Step 2: Add new dependencies to requirements.txt**

Append to `backend/requirements.txt`:

```
# ===== Web search =====
tavily-python>=0.5,<1.0
```

- [ ] **Step 3: Extend config.py with agent settings**

Add these fields to the `Settings` class in `backend/app/core/config.py` after the existing `llm_max_retries` field:

```python
    # --- Agent ---
    agent_max_plan_steps: int = 7
    agent_max_reflections: int = 2

    # --- Tools: external APIs ---
    tavily_api_key: Optional[str] = None
    arxiv_max_results: int = 5

    # --- Optional separate models for agent nodes ---
    planner_model: Optional[str] = None
    reflection_model: Optional[str] = None
```

- [ ] **Step 4: Create agent state types**

Create `backend/app/agent/__init__.py`:

```python
```

Create `backend/app/agent/state.py`:

```python
"""Agent state schema and supporting types."""
from __future__ import annotations

from typing import Annotated, Optional, TypedDict

from langchain_core.documents import Document
from langgraph.graph.message import add_messages


class StepSpec(TypedDict):
    action: str
    params: dict
    reason: str


class StepTrace(TypedDict):
    node: str
    action: str
    input_summary: str
    output_summary: str
    duration_ms: float


class ReflectionResult(TypedDict):
    passed: bool
    citation_ok: bool
    completeness_ok: bool
    logic_ok: bool
    issues: list[str]
    fix_strategy: Optional[str]


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: Optional[dict]
    plan: list[StepSpec]
    plan_step_index: int
    retrieval_context: list[Document]
    step_traces: list[StepTrace]
    reflection_count: int
    final_answer: Optional[str]
```

- [ ] **Step 5: Create tools package init**

Create `backend/app/tools/__init__.py`:

```python
```

- [ ] **Step 6: Install new deps and verify import**

```bash
cd backend && pip install -r requirements.txt
python -c "from app.agent.state import AgentState, StepSpec, StepTrace, ReflectionResult; print('OK')"
```

Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat: scaffold agent module, state types, and config extensions"
```

---

### Task 2: Tools — retrieve_local

**Files:**
- Create: `backend/app/tools/retrieve_local.py`
- Create: `backend/tests/tools/test_retrieve_local.py`

- [ ] **Step 1: Write test**

Create `backend/tests/__init__.py` and `backend/tests/tools/__init__.py` (empty).

Create `backend/tests/tools/test_retrieve_local.py`:

```python
"""Test retrieve_local tool."""
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from app.tools.retrieve_local import retrieve_local_tool


def test_retrieve_local_returns_formatted_chunks():
    mock_docs = [
        (Document(page_content="Attention is all you need", metadata={"paper_id": "1706.03762", "title": "Attention", "page_num": 1}), 0.92),
        (Document(page_content="BERT uses masked LM", metadata={"paper_id": "1810.04805", "title": "BERT", "page_num": 3}), 0.85),
    ]
    with patch("app.tools.retrieve_local.retrieve", return_value=mock_docs):
        result = retrieve_local_tool.invoke({"query": "attention mechanism", "top_k": 8})

    assert "1706.03762" in result
    assert "0.92" in result or "0.920" in result
    assert "Attention is all you need" in result


def test_retrieve_local_with_filter():
    mock_docs = [
        (Document(page_content="Some NLP content", metadata={"paper_id": "2301.00001", "title": "NLP Paper", "page_num": 2}), 0.88),
    ]
    with patch("app.tools.retrieve_local.retrieve", return_value=mock_docs) as mock_retrieve:
        result = retrieve_local_tool.invoke({"query": "NLP", "top_k": 5, "category": "cs.CL", "year_min": 2023})

    call_args = mock_retrieve.call_args
    assert call_args[1]["flt"] is not None
    assert call_args[1]["flt"].category == "cs.CL"


def test_retrieve_local_empty_results():
    with patch("app.tools.retrieve_local.retrieve", return_value=[]):
        result = retrieve_local_tool.invoke({"query": "nonexistent topic"})

    assert "No results" in result or "no results" in result.lower()
```

- [ ] **Step 2: Run test, verify failure**

```bash
cd backend && python -m pytest tests/tools/test_retrieve_local.py -v
```

Expected: FAIL (module not found)

- [ ] **Step 3: Implement retrieve_local tool**

Create `backend/app/tools/retrieve_local.py`:

```python
"""Tool: local vector + BM25 hybrid retrieval."""
from __future__ import annotations

from langchain_core.tools import tool

from app.schemas.chat import ChatFilter
from app.services.retriever import retrieve


def _format_chunks(docs_scores: list) -> str:
    if not docs_scores:
        return "No results found."
    parts = []
    for d, score in docs_scores:
        md = d.metadata or {}
        parts.append(
            f"[arxiv:{md.get('paper_id', '?')} | {md.get('title', '')[:100]} | "
            f"score={score:.3f} | page={md.get('page_num', '?')}]\n"
            f"{d.page_content[:500]}"
        )
    return "\n\n---\n\n".join(parts)


@tool
def retrieve_local_tool(
    query: str,
    top_k: int = 8,
    category: str = "",
    year_min: int = 0,
    year_max: int = 0,
) -> str:
    """Search local paper database using hybrid vector + BM25 retrieval.
    Returns ranked chunks with paper IDs, relevance scores, and text snippets.
    Use when you need to find relevant academic paper content."""
    flt = None
    if category or year_min or year_max:
        flt = ChatFilter(
            category=category or None,
            year_min=year_min if year_min > 0 else None,
            year_max=year_max if year_max > 0 else None,
        )
    docs_scores = retrieve(query, flt=flt, top_k=top_k)
    return _format_chunks(docs_scores)
```

- [ ] **Step 4: Run tests, verify pass**

```bash
cd backend && python -m pytest tests/tools/test_retrieve_local.py -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add backend/app/tools/retrieve_local.py backend/tests/tools/
git commit -m "feat: implement retrieve_local tool with hybrid search"
```

---

### Task 3: Tools — retrieve_arxiv

**Files:**
- Create: `backend/app/tools/retrieve_arxiv.py`
- Create: `backend/tests/tools/test_retrieve_arxiv.py`

- [ ] **Step 1: Write test**

Create `backend/tests/tools/test_retrieve_arxiv.py`:

```python
"""Test retrieve_arxiv tool."""
from unittest.mock import patch, MagicMock

from app.tools.retrieve_arxiv import retrieve_arxiv_tool


def _mock_arxiv_result(title, summary, entry_id, published_year):
    r = MagicMock()
    r.title = title
    r.summary = summary
    r.entry_id = entry_id
    r.published = MagicMock()
    r.published.year = published_year
    r.primary_category = "cs.CL"
    r.authors = [MagicMock(name="Author A")]
    return r


def test_retrieve_arxiv_returns_formatted():
    mock_results = [
        _mock_arxiv_result("Paper A", "Abstract A about transformers", "http://arxiv.org/abs/2401.00001", 2024),
    ]
    with patch("app.tools.retrieve_arxiv.arxiv") as mock_arxiv:
        mock_client = MagicMock()
        mock_arxiv.Client.return_value = mock_client
        mock_client.results.return_value = iter(mock_results)
        mock_arxiv.Search.return_value = MagicMock()

        result = retrieve_arxiv_tool.invoke({"query": "transformers", "max_results": 5})

    assert "Paper A" in result
    assert "2401.00001" in result
    assert "Abstract A" in result


def test_retrieve_arxiv_empty():
    with patch("app.tools.retrieve_arxiv.arxiv") as mock_arxiv:
        mock_client = MagicMock()
        mock_arxiv.Client.return_value = mock_client
        mock_client.results.return_value = iter([])
        mock_arxiv.Search.return_value = MagicMock()

        result = retrieve_arxiv_tool.invoke({"query": "nonexistent", "max_results": 3})

    assert "No results" in result or "no papers" in result.lower()
```

- [ ] **Step 2: Run test, verify failure**

```bash
cd backend && python -m pytest tests/tools/test_retrieve_arxiv.py -v
```

- [ ] **Step 3: Implement**

Create `backend/app/tools/retrieve_arxiv.py`:

```python
"""Tool: real-time arXiv API search."""
from __future__ import annotations

import arxiv
from langchain_core.tools import tool

from app.core.config import get_settings


@tool
def retrieve_arxiv_tool(query: str, max_results: int = 5) -> str:
    """Search arXiv for recent papers matching the query.
    Returns paper titles, abstracts, IDs, and categories.
    Use when local database may not have the latest papers or when you need broader coverage."""
    settings = get_settings()
    max_results = min(max_results, settings.arxiv_max_results)

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    client = arxiv.Client()
    results = list(client.results(search))

    if not results:
        return "No papers found on arXiv for this query."

    parts = []
    for r in results:
        paper_id = r.entry_id.split("/abs/")[-1] if "/abs/" in r.entry_id else r.entry_id
        authors = ", ".join(str(a) for a in (r.authors or [])[:3])
        parts.append(
            f"[arxiv:{paper_id} | {r.primary_category} | {r.published.year}]\n"
            f"title: {r.title}\n"
            f"authors: {authors}\n"
            f"abstract: {r.summary[:400]}"
        )
    return "\n\n---\n\n".join(parts)
```

- [ ] **Step 4: Run tests, verify pass**

```bash
cd backend && python -m pytest tests/tools/test_retrieve_arxiv.py -v
```

- [ ] **Step 5: Commit**

```bash
git add backend/app/tools/retrieve_arxiv.py backend/tests/tools/test_retrieve_arxiv.py
git commit -m "feat: implement retrieve_arxiv tool for real-time paper search"
```

---

### Task 4: Tools — search_web

**Files:**
- Create: `backend/app/tools/search_web.py`
- Create: `backend/tests/tools/test_search_web.py`

- [ ] **Step 1: Write test**

Create `backend/tests/tools/test_search_web.py`:

```python
"""Test search_web tool."""
from unittest.mock import patch, MagicMock

from app.tools.search_web import search_web_tool


def test_search_web_returns_formatted():
    mock_response = MagicMock()
    mock_response.results = [
        {"title": "What is Attention?", "url": "https://example.com/attention", "content": "Attention mechanisms allow models to focus..."},
    ]
    with patch("app.tools.search_web.TavilyClient") as MockClient:
        MockClient.return_value.search.return_value = mock_response
        result = search_web_tool.invoke({"query": "attention mechanism explained", "max_results": 3})

    assert "What is Attention?" in result
    assert "example.com" in result


def test_search_web_no_api_key():
    with patch("app.tools.search_web.get_settings") as mock_settings:
        mock_settings.return_value.tavily_api_key = None
        result = search_web_tool.invoke({"query": "test"})

    assert "not configured" in result.lower() or "unavailable" in result.lower()
```

- [ ] **Step 2: Run test, verify failure**

```bash
cd backend && python -m pytest tests/tools/test_search_web.py -v
```

- [ ] **Step 3: Implement**

Create `backend/app/tools/search_web.py`:

```python
"""Tool: web search via Tavily API."""
from __future__ import annotations

from langchain_core.tools import tool

from app.core.config import get_settings


@tool
def search_web_tool(query: str, max_results: int = 3) -> str:
    """Search the web for background knowledge, explanations, or recent news.
    Use when the question requires context beyond academic papers (e.g., concept definitions, current events)."""
    settings = get_settings()
    if not settings.tavily_api_key:
        return "Web search is not configured (TAVILY_API_KEY missing)."

    from tavily import TavilyClient

    client = TavilyClient(api_key=settings.tavily_api_key)
    response = client.search(query=query, max_results=max_results)

    results = response.results if hasattr(response, "results") else response.get("results", [])
    if not results:
        return "No web results found."

    parts = []
    for r in results:
        title = r.get("title", r["title"]) if isinstance(r, dict) else r.title
        url = r.get("url", "") if isinstance(r, dict) else r.url
        content = r.get("content", "") if isinstance(r, dict) else r.content
        parts.append(f"[{title}]({url})\n{content[:300]}")
    return "\n\n---\n\n".join(parts)
```

- [ ] **Step 4: Run tests, verify pass**

```bash
cd backend && python -m pytest tests/tools/test_search_web.py -v
```

- [ ] **Step 5: Commit**

```bash
git add backend/app/tools/search_web.py backend/tests/tools/test_search_web.py
git commit -m "feat: implement search_web tool with Tavily API"
```

---

### Task 5: Tools — paper_detail + paper_chunks

**Files:**
- Create: `backend/app/tools/paper_detail.py`
- Create: `backend/app/tools/paper_chunks.py`
- Create: `backend/tests/tools/test_knowledge_tools.py`

- [ ] **Step 1: Write tests**

Create `backend/tests/tools/test_knowledge_tools.py`:

```python
"""Test knowledge tools (paper_detail, paper_chunks)."""
from unittest.mock import patch, MagicMock

from app.tools.paper_detail import get_paper_detail
from app.tools.paper_chunks import get_paper_chunks


def _mock_paper():
    p = MagicMock()
    p.paper_id = "2301.00001"
    p.title = "Test Paper"
    p.authors = ["Author A", "Author B"]
    p.year = 2023
    p.primary_category = "cs.CL"
    p.categories = ["cs.CL", "cs.AI"]
    p.abstract = "This paper presents a novel approach."
    p.doi = None
    return p


def test_get_paper_detail_found():
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.one_or_none.return_value = _mock_paper()

    result = get_paper_detail(mock_db, "2301.00001")

    assert "2301.00001" in result
    assert "Test Paper" in result
    assert "Author A" in result


def test_get_paper_detail_not_found():
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.one_or_none.return_value = None

    result = get_paper_detail(mock_db, "9999.99999")

    assert "not found" in result.lower()


def _mock_chunk(text, page_num):
    c = MagicMock()
    c.chunk_text = text
    c.page_num = page_num
    c.chunk_index = 0
    return c


def test_get_paper_chunks_found():
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = [
        _mock_chunk("First chunk content", 1),
        _mock_chunk("Second chunk content", 2),
    ]

    result = get_paper_chunks(mock_db, "2301.00001", max_chunks=10)

    assert "First chunk" in result
    assert "Second chunk" in result


def test_get_paper_chunks_empty():
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []

    result = get_paper_chunks(mock_db, "9999.99999")

    assert "no chunks" in result.lower() or "not found" in result.lower()
```

- [ ] **Step 2: Run test, verify failure**

```bash
cd backend && python -m pytest tests/tools/test_knowledge_tools.py -v
```

- [ ] **Step 3: Implement paper_detail**

Create `backend/app/tools/paper_detail.py`:

```python
"""Tool: get full paper metadata from MySQL."""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.models.paper import Paper


def get_paper_detail(db: Session, paper_id: str) -> str:
    """Get full metadata for a paper. Returns title, authors, year, categories, abstract."""
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
```

- [ ] **Step 4: Implement paper_chunks**

Create `backend/app/tools/paper_chunks.py`:

```python
"""Tool: get chunks for a specific paper."""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.models.paper import Chunk


def get_paper_chunks(db: Session, paper_id: str, max_chunks: int = 10) -> str:
    """Get text chunks for a specific paper, ordered by chunk_index."""
    chunks = (
        db.query(Chunk)
        .filter(Chunk.paper_id == paper_id)
        .order_by(Chunk.chunk_index)
        .limit(max_chunks)
        .all()
    )
    if not chunks:
        return f"No chunks found for paper {paper_id}."
    parts = []
    for c in chunks:
        parts.append(f"[page={c.page_num} | chunk={c.chunk_index}]\n{c.chunk_text[:500]}")
    return "\n\n---\n\n".join(parts)
```

- [ ] **Step 5: Run tests, verify pass**

```bash
cd backend && python -m pytest tests/tools/test_knowledge_tools.py -v
```

- [ ] **Step 6: Commit**

```bash
git add backend/app/tools/paper_detail.py backend/app/tools/paper_chunks.py backend/tests/tools/test_knowledge_tools.py
git commit -m "feat: implement paper_detail and paper_chunks tools"
```

---

### Task 6: Tools — query_rewrite + evaluate_docs

**Files:**
- Create: `backend/app/tools/query_rewrite.py`
- Create: `backend/app/tools/evaluate_docs.py`
- Create: `backend/tests/tools/test_reasoning_tools.py`

- [ ] **Step 1: Write tests**

Create `backend/tests/tools/test_reasoning_tools.py`:

```python
"""Test reasoning tools (query_rewrite, evaluate_docs)."""
import json
from unittest.mock import patch, MagicMock

from app.tools.query_rewrite import rewrite_query
from app.tools.evaluate_docs import evaluate_docs


def test_rewrite_query_returns_list():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='["BERT pretraining strategy", "GPT pretraining approach"]'
    )
    with patch("app.tools.query_rewrite._get_llm", return_value=mock_llm):
        result = rewrite_query("compare BERT and GPT pretraining", {"type": "comparison", "entities": ["BERT", "GPT"]})

    assert isinstance(result, list)
    assert len(result) == 2
    assert "BERT" in result[0]


def test_rewrite_query_single():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content='["attention mechanism in transformers"]')
    with patch("app.tools.query_rewrite._get_llm", return_value=mock_llm):
        result = rewrite_query("what is attention", {"type": "simple"})

    assert isinstance(result, list)
    assert len(result) == 1


def test_evaluate_docs_sufficient():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='{"sufficient": true, "reason": "Context covers the topic well", "missing_aspects": []}'
    )
    with patch("app.tools.evaluate_docs._get_llm", return_value=mock_llm):
        result = evaluate_docs("what is attention", ["chunk about attention mechanisms"])

    assert result["sufficient"] is True
    assert result["missing_aspects"] == []


def test_evaluate_docs_insufficient():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='{"sufficient": false, "reason": "Missing information about multi-head attention", "missing_aspects": ["multi-head attention details"]}'
    )
    with patch("app.tools.evaluate_docs._get_llm", return_value=mock_llm):
        result = evaluate_docs("explain multi-head attention", ["basic attention info"])

    assert result["sufficient"] is False
    assert len(result["missing_aspects"]) > 0
```

- [ ] **Step 2: Run test, verify failure**

```bash
cd backend && python -m pytest tests/tools/test_reasoning_tools.py -v
```

- [ ] **Step 3: Implement query_rewrite**

Create `backend/app/tools/query_rewrite.py`:

```python
"""Tool: LLM-powered query rewriting and decomposition."""
from __future__ import annotations

import json

from langchain_openai import ChatOpenAI

from app.core.config import get_settings

_REWRITE_PROMPT = """你是一个学术检索查询优化器。根据用户的原始问题和意图分析，生成 1-3 个改写后的检索查询。

规则：
1. 如果是对比类问题，拆分成针对每个对象的独立子查询。
2. 如果是简单问题，优化关键词使其更适合语义检索���
3. 使用英文关键词（学术论文多为英文）。
4. 输出严格 JSON 数组格式：["query1", "query2", ...]

原始问题：{query}
意图分析：{intent}

输出改写后的查询数组："""


def _get_llm() -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        model=s.planner_model or s.llm_model,
        base_url=s.llm_api_base,
        api_key=s.llm_api_key,
        temperature=0.3,
        max_retries=2,
    )


def rewrite_query(original_query: str, intent: dict) -> list[str]:
    """Rewrite and decompose a query into 1-3 optimized sub-queries for retrieval."""
    llm = _get_llm()
    prompt = _REWRITE_PROMPT.format(query=original_query, intent=json.dumps(intent, ensure_ascii=False))
    response = llm.invoke(prompt)
    try:
        queries = json.loads(response.content)
        if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
            return queries[:3]
    except (json.JSONDecodeError, TypeError):
        pass
    return [original_query]
```

- [ ] **Step 4: Implement evaluate_docs**

Create `backend/app/tools/evaluate_docs.py`:

```python
"""Tool: LLM-powered document sufficiency evaluation."""
from __future__ import annotations

import json

from langchain_openai import ChatOpenAI

from app.core.config import get_settings

_EVAL_PROMPT = """你是一个学术检索质量评估器。判断当前检索到的资料是否足以回答用户问题。

用户问题：{query}

已检索到的资料摘要：
{context_summary}

请评估：
1. 这些资料是否足够回答用户的问题？
2. 如果不够，缺少哪些方面的信息？

输出严格 JSON 格式：
{{"sufficient": true/false, "reason": "评估理由", "missing_aspects": ["缺失方面1", ...]}}"""


def _get_llm() -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        model=s.planner_model or s.llm_model,
        base_url=s.llm_api_base,
        api_key=s.llm_api_key,
        temperature=0.1,
        max_retries=2,
    )


def evaluate_docs(query: str, context_texts: list[str]) -> dict:
    """Evaluate whether retrieved documents are sufficient to answer the query.
    Returns {sufficient: bool, reason: str, missing_aspects: list[str]}."""
    context_summary = "\n".join(f"- {t[:200]}" for t in context_texts[:10])
    llm = _get_llm()
    prompt = _EVAL_PROMPT.format(query=query, context_summary=context_summary)
    response = llm.invoke(prompt)
    try:
        result = json.loads(response.content)
        return {
            "sufficient": bool(result.get("sufficient", False)),
            "reason": str(result.get("reason", "")),
            "missing_aspects": list(result.get("missing_aspects", [])),
        }
    except (json.JSONDecodeError, TypeError):
        return {"sufficient": True, "reason": "Failed to parse evaluation", "missing_aspects": []}
```

- [ ] **Step 5: Run tests, verify pass**

```bash
cd backend && python -m pytest tests/tools/test_reasoning_tools.py -v
```

- [ ] **Step 6: Commit**

```bash
git add backend/app/tools/query_rewrite.py backend/app/tools/evaluate_docs.py backend/tests/tools/test_reasoning_tools.py
git commit -m "feat: implement query_rewrite and evaluate_docs reasoning tools"
```

---

## Phase 2: Agent Nodes + Graph

### Task 7: Agent Prompts

**Files:**
- Create: `backend/app/agent/prompts/__init__.py`
- Create: `backend/app/agent/prompts/intent.py`
- Create: `backend/app/agent/prompts/planner.py`
- Create: `backend/app/agent/prompts/synthesis.py`
- Create: `backend/app/agent/prompts/reflection.py`

- [ ] **Step 1: Create prompts package**

Create `backend/app/agent/prompts/__init__.py`:

```python
```

- [ ] **Step 2: Write intent prompt**

Create `backend/app/agent/prompts/intent.py`:

```python
INTENT_PROMPT = """分析用户的学术问题，提取意图信息。

用户问题：{query}

请输出 JSON：
{{
  "type": "simple" | "complex" | "comparison",
  "entities": ["提到的关键实体/论文/方法"],
  "complexity": "low" | "medium" | "high"
}}

判断标准：
- simple: 单一概念查询，一次检索即可回答
- comparison: 需要对比多个对象
- complex: 需要多步推理或综合多方面信息

输出 JSON："""
```

- [ ] **Step 3: Write planner prompt**

Create `backend/app/agent/prompts/planner.py`:

```python
PLANNER_PROMPT = """你是一个学术RAG系统的执行规划器。根据用户问题和意图分析，生成检索与推理的执行计划。

用户问题：{query}
意图分析：{intent}

可用动作：
- query_rewrite: 改写/分解查询（适合复杂或对比类问题）
- retrieve_local: 从本地论文库检索（主要检索手段）
- retrieve_arxiv: 从 arXiv 实时搜索（本地不够时补充）
- search_web: 网页搜索（需要背景知识时）
- evaluate_docs: 评估资料充分性（检索后必须执行）
- reasoning_synthesis: 推理并生成答案（最后执行）

规则：
1. 简单问题不超过 3 步，复杂问题不超过 {max_steps} 步。
2. 必须包含 evaluate_docs（检索完成后）和 reasoning_synthesis（最后）。
3. 对比类问题应先 query_rewrite 分解再分别检索。
4. 每一步必须有 reason 说明为什么需要这一步。

输出 JSON 数组：
[{{"action": "...", "params": {{...}}, "reason": "..."}}]"""


RE_PLANNER_PROMPT = """之前的执行计划未能充分回答问题。根据反思结果，生成补充计划。

用户问题：{query}
反思发现的问题：{issues}
缺失的方面：{missing_aspects}

可用动作：retrieve_local, retrieve_arxiv, search_web, evaluate_docs, reasoning_synthesis

生成补充步骤（不超过 3 步）：
[{{"action": "...", "params": {{...}}, "reason": "..."}}]"""
```

- [ ] **Step 4: Write synthesis prompt**

Create `backend/app/agent/prompts/synthesis.py`:

```python
SYNTHESIS_PROMPT = """你是一个严谨的学术论文问答助手。基于检索到的参考资料回答用户问题。

硬性规则：
1. 回答必须基于下方参考资料；若资料不足以回答，请如实说"参考资料不足以回答该问题"。
2. 每条论据/结论末尾必须追加 [arxiv:PAPER_ID] 引用。多个来源用 [arxiv:ID1][arxiv:ID2]。
3. 禁止编造 paper_id；只能使用参考资料中出现的 paper_id。
4. 输出使用简洁的 Markdown，中文作答（除非用户用英文提问）。

参考资料：
{context}

用户问题：{query}

请作答："""


SYNTHESIS_WITH_ISSUES_PROMPT = """你是一个严谨的学术论文问答助手。请修正之前答案中的问题，重新回答。

之前答案的问题：
{issues}

硬性规则：
1. 回答必须基于下方参考资料；若资料不足以回答，请如实说"参考资料不足以回答该问题"。
2. 每条论据/结论末尾必须追加 [arxiv:PAPER_ID] 引用。
3. 禁止编造 paper_id。
4. 输出使用简洁的 Markdown，中文作答。

参考资料：
{context}

用户问题：{query}

请重新作答，修正上述问题："""
```

- [ ] **Step 5: Write reflection prompt**

Create `backend/app/agent/prompts/reflection.py`:

```python
REFLECTION_PROMPT = """你是一个严格的学术问答质量审核员。请从三个维度评估以下回答。

用户问题：{query}

检索到的参考资料（paper_id 列表）：
{available_paper_ids}

生成的回答：
{answer}

评估维度：
1. Citation Faithfulness（引用忠实度）：回答中每个 [arxiv:ID] 是否都出现在参考资料中？是否有未引用来源的论断？
2. Completeness（完整性）：回答是否完整回答了用户的问题？是否遗漏了重要方面？
3. Logical Consistency（逻辑一致性）：回答内部是否有矛盾？推理链是否连贯？

输出严格 JSON：
{{
  "passed": true/false,
  "citation_ok": true/false,
  "completeness_ok": true/false,
  "logic_ok": true/false,
  "issues": ["具体问题描述..."],
  "fix_strategy": "re_retrieve" | "re_generate" | null
}}

fix_strategy 说明：
- "re_retrieve": 需要补充检索更多资料
- "re_generate": 资料充分但需要重新组织答案
- null: 通过，无需修复"""
```

- [ ] **Step 6: Commit**

```bash
git add backend/app/agent/prompts/
git commit -m "feat: add all agent prompt templates"
```

---

### Task 8: Agent Nodes — intent + planner

**Files:**
- Create: `backend/app/agent/nodes/__init__.py`
- Create: `backend/app/agent/nodes/intent.py`
- Create: `backend/app/agent/nodes/planner.py`
- Create: `backend/tests/agent/test_intent_planner.py`

- [ ] **Step 1: Write tests**

Create `backend/tests/agent/__init__.py` (empty).

Create `backend/tests/agent/test_intent_planner.py`:

```python
"""Test intent and planner nodes."""
import json
from unittest.mock import patch, MagicMock

from app.agent.state import AgentState, StepSpec
from app.agent.nodes.intent import intent_node
from app.agent.nodes.planner import planner_node


def _base_state(**overrides) -> AgentState:
    defaults = {
        "messages": [],
        "intent": None,
        "plan": [],
        "plan_step_index": 0,
        "retrieval_context": [],
        "step_traces": [],
        "reflection_count": 0,
        "final_answer": None,
    }
    defaults.update(overrides)
    return defaults


def test_intent_node_simple_question():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='{"type": "simple", "entities": ["attention"], "complexity": "low"}'
    )
    state = _base_state()
    with patch("app.agent.nodes.intent._get_llm", return_value=mock_llm):
        result = intent_node(state, query="what is attention mechanism")

    assert result["intent"]["type"] == "simple"
    assert result["intent"]["complexity"] == "low"
    assert len(result["step_traces"]) == 1


def test_intent_node_comparison():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='{"type": "comparison", "entities": ["BERT", "GPT"], "complexity": "high"}'
    )
    state = _base_state()
    with patch("app.agent.nodes.intent._get_llm", return_value=mock_llm):
        result = intent_node(state, query="compare BERT and GPT")

    assert result["intent"]["type"] == "comparison"
    assert "BERT" in result["intent"]["entities"]


def test_planner_node_generates_plan():
    mock_llm = MagicMock()
    plan_json = json.dumps([
        {"action": "retrieve_local", "params": {"query": "attention", "top_k": 8}, "reason": "search locally"},
        {"action": "evaluate_docs", "params": {}, "reason": "check sufficiency"},
        {"action": "reasoning_synthesis", "params": {}, "reason": "generate answer"},
    ])
    mock_llm.invoke.return_value = MagicMock(content=plan_json)

    state = _base_state(intent={"type": "simple", "entities": ["attention"], "complexity": "low"})
    with patch("app.agent.nodes.planner._get_llm", return_value=mock_llm):
        result = planner_node(state, query="what is attention")

    assert len(result["plan"]) == 3
    assert result["plan"][0]["action"] == "retrieve_local"
    assert result["plan_step_index"] == 0
```

- [ ] **Step 2: Run test, verify failure**

```bash
cd backend && python -m pytest tests/agent/test_intent_planner.py -v
```

- [ ] **Step 3: Implement intent_node**

Create `backend/app/agent/nodes/__init__.py`:

```python
```

Create `backend/app/agent/nodes/intent.py`:

```python
"""Intent analysis node."""
from __future__ import annotations

import json
import time

from langchain_openai import ChatOpenAI

from app.agent.prompts.intent import INTENT_PROMPT
from app.agent.state import AgentState, StepTrace
from app.core.config import get_settings


def _get_llm() -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        model=s.planner_model or s.llm_model,
        base_url=s.llm_api_base,
        api_key=s.llm_api_key,
        temperature=0.1,
        max_retries=2,
    )


def intent_node(state: AgentState, *, query: str) -> dict:
    """Analyze user intent: type, entities, complexity."""
    t0 = time.perf_counter()
    llm = _get_llm()
    prompt = INTENT_PROMPT.format(query=query)
    response = llm.invoke(prompt)

    try:
        intent = json.loads(response.content)
    except (json.JSONDecodeError, TypeError):
        intent = {"type": "simple", "entities": [], "complexity": "low"}

    duration = round((time.perf_counter() - t0) * 1000, 2)
    trace = StepTrace(
        node="intent_node",
        action="intent_analysis",
        input_summary=query[:100],
        output_summary=f"type={intent.get('type')}, complexity={intent.get('complexity')}",
        duration_ms=duration,
    )
    return {"intent": intent, "step_traces": state["step_traces"] + [trace]}
```

- [ ] **Step 4: Implement planner_node**

Create `backend/app/agent/nodes/planner.py`:

```python
"""Planner and re-planner nodes."""
from __future__ import annotations

import json
import time

from langchain_openai import ChatOpenAI

from app.agent.prompts.planner import PLANNER_PROMPT, RE_PLANNER_PROMPT
from app.agent.state import AgentState, StepSpec, StepTrace
from app.core.config import get_settings


def _get_llm() -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        model=s.planner_model or s.llm_model,
        base_url=s.llm_api_base,
        api_key=s.llm_api_key,
        temperature=0.3,
        max_retries=2,
    )


def _parse_plan(content: str, max_steps: int) -> list[StepSpec]:
    try:
        steps = json.loads(content)
        if isinstance(steps, list):
            return [
                StepSpec(
                    action=s.get("action", ""),
                    params=s.get("params", {}),
                    reason=s.get("reason", ""),
                )
                for s in steps[:max_steps]
            ]
    except (json.JSONDecodeError, TypeError):
        pass
    return [
        StepSpec(action="retrieve_local", params={"query": "", "top_k": 8}, reason="fallback"),
        StepSpec(action="evaluate_docs", params={}, reason="check sufficiency"),
        StepSpec(action="reasoning_synthesis", params={}, reason="generate answer"),
    ]


def planner_node(state: AgentState, *, query: str) -> dict:
    """Generate structured execution plan based on intent."""
    t0 = time.perf_counter()
    settings = get_settings()
    llm = _get_llm()

    intent = state["intent"] or {"type": "simple", "entities": [], "complexity": "low"}
    prompt = PLANNER_PROMPT.format(
        query=query,
        intent=json.dumps(intent, ensure_ascii=False),
        max_steps=settings.agent_max_plan_steps,
    )
    response = llm.invoke(prompt)
    plan = _parse_plan(response.content, settings.agent_max_plan_steps)

    duration = round((time.perf_counter() - t0) * 1000, 2)
    trace = StepTrace(
        node="planner_node",
        action="planning",
        input_summary=f"intent={intent.get('type')}, complexity={intent.get('complexity')}",
        output_summary=f"generated {len(plan)} steps",
        duration_ms=duration,
    )
    return {
        "plan": plan,
        "plan_step_index": 0,
        "step_traces": state["step_traces"] + [trace],
    }


def re_planner_node(state: AgentState, *, query: str, issues: list[str], missing_aspects: list[str]) -> dict:
    """Generate supplementary plan after reflection failure."""
    t0 = time.perf_counter()
    settings = get_settings()
    llm = _get_llm()

    prompt = RE_PLANNER_PROMPT.format(
        query=query,
        issues=json.dumps(issues, ensure_ascii=False),
        missing_aspects=json.dumps(missing_aspects, ensure_ascii=False),
    )
    response = llm.invoke(prompt)
    new_steps = _parse_plan(response.content, 3)

    duration = round((time.perf_counter() - t0) * 1000, 2)
    trace = StepTrace(
        node="re_planner_node",
        action="re_planning",
        input_summary=f"issues: {', '.join(issues[:2])}",
        output_summary=f"generated {len(new_steps)} supplementary steps",
        duration_ms=duration,
    )
    return {
        "plan": state["plan"] + new_steps,
        "step_traces": state["step_traces"] + [trace],
    }
```

- [ ] **Step 5: Run tests, verify pass**

```bash
cd backend && python -m pytest tests/agent/test_intent_planner.py -v
```

- [ ] **Step 6: Commit**

```bash
git add backend/app/agent/nodes/ backend/tests/agent/
git commit -m "feat: implement intent_node and planner_node"
```

---

### Task 9: Agent Nodes — executor

**Files:**
- Create: `backend/app/agent/nodes/executor.py`
- Create: `backend/tests/agent/test_executor.py`

- [ ] **Step 1: Write test**

Create `backend/tests/agent/test_executor.py`:

```python
"""Test executor node."""
import time
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from app.agent.state import AgentState, StepSpec
from app.agent.nodes.executor import executor_node


def _base_state(**overrides) -> AgentState:
    defaults = {
        "messages": [],
        "intent": {"type": "simple", "entities": [], "complexity": "low"},
        "plan": [],
        "plan_step_index": 0,
        "retrieval_context": [],
        "step_traces": [],
        "reflection_count": 0,
        "final_answer": None,
    }
    defaults.update(overrides)
    return defaults


def test_executor_retrieve_local():
    plan = [StepSpec(action="retrieve_local", params={"query": "attention", "top_k": 8}, reason="test")]
    state = _base_state(plan=plan, plan_step_index=0)

    mock_docs = [
        (Document(page_content="attention content", metadata={"paper_id": "1706.03762"}), 0.9),
    ]
    with patch("app.agent.nodes.executor._run_retrieve_local", return_value=mock_docs):
        result = executor_node(state, db=MagicMock())

    assert result["plan_step_index"] == 1
    assert len(result["retrieval_context"]) == 1
    assert len(result["step_traces"]) == 1


def test_executor_query_rewrite():
    plan = [StepSpec(action="query_rewrite", params={"original_query": "test"}, reason="decompose")]
    state = _base_state(plan=plan, plan_step_index=0)

    with patch("app.agent.nodes.executor._run_query_rewrite", return_value=["sub query 1", "sub query 2"]):
        result = executor_node(state, db=MagicMock())

    assert result["plan_step_index"] == 1
    assert len(result["step_traces"]) == 1


def test_executor_advances_index():
    plan = [
        StepSpec(action="retrieve_local", params={"query": "q1", "top_k": 8}, reason="first"),
        StepSpec(action="evaluate_docs", params={}, reason="second"),
    ]
    state = _base_state(plan=plan, plan_step_index=0)

    with patch("app.agent.nodes.executor._run_retrieve_local", return_value=[]):
        result = executor_node(state, db=MagicMock())

    assert result["plan_step_index"] == 1
```

- [ ] **Step 2: Run test, verify failure**

```bash
cd backend && python -m pytest tests/agent/test_executor.py -v
```

- [ ] **Step 3: Implement executor_node**

Create `backend/app/agent/nodes/executor.py`:

```python
"""Executor node: dispatches plan steps to tools."""
from __future__ import annotations

import time
from typing import Optional

from langchain_core.documents import Document
from sqlalchemy.orm import Session

from app.agent.state import AgentState, StepSpec, StepTrace
from app.schemas.chat import ChatFilter
from app.services.retriever import retrieve
from app.tools.query_rewrite import rewrite_query
from app.tools.evaluate_docs import evaluate_docs
from app.tools.retrieve_arxiv import retrieve_arxiv_tool
from app.tools.search_web import search_web_tool
from app.tools.paper_detail import get_paper_detail
from app.tools.paper_chunks import get_paper_chunks


def _run_retrieve_local(params: dict) -> list[tuple[Document, float]]:
    query = params.get("query", "")
    top_k = params.get("top_k", 8)
    flt = None
    if params.get("category") or params.get("year_min") or params.get("year_max"):
        flt = ChatFilter(
            category=params.get("category") or None,
            year_min=params.get("year_min"),
            year_max=params.get("year_max"),
        )
    return retrieve(query, flt=flt, top_k=top_k)


def _run_query_rewrite(params: dict, intent: dict) -> list[str]:
    return rewrite_query(params.get("original_query", ""), intent)


def _run_evaluate_docs(params: dict, query: str, context: list[Document]) -> dict:
    texts = [d.page_content for d in context]
    return evaluate_docs(query, texts)


def executor_node(state: AgentState, *, db: Session) -> dict:
    """Execute the current plan step and advance the index."""
    idx = state["plan_step_index"]
    step = state["plan"][idx]
    action = step["action"]
    params = step["params"]

    t0 = time.perf_counter()
    new_context = list(state["retrieval_context"])
    output_summary = ""

    if action == "retrieve_local":
        docs_scores = _run_retrieve_local(params)
        new_context.extend([d for d, _ in docs_scores])
        output_summary = f"found {len(docs_scores)} chunks"

    elif action == "retrieve_arxiv":
        result = retrieve_arxiv_tool.invoke(params)
        output_summary = f"arXiv results: {len(result.split('---'))} papers"

    elif action == "search_web":
        result = search_web_tool.invoke(params)
        output_summary = f"web results received"

    elif action == "query_rewrite":
        intent = state["intent"] or {}
        queries = _run_query_rewrite(params, intent)
        output_summary = f"rewrote into {len(queries)} sub-queries"
        # Inject rewritten queries into subsequent retrieve steps
        for i, plan_step in enumerate(state["plan"][idx + 1:], start=idx + 1):
            if plan_step["action"] == "retrieve_local" and not plan_step["params"].get("query"):
                if queries:
                    plan_step["params"]["query"] = queries.pop(0)

    elif action == "evaluate_docs":
        query = params.get("query", "")
        if not query:
            # Try to extract from messages
            for msg in reversed(state["messages"]):
                if hasattr(msg, "content") and msg.content:
                    query = msg.content
                    break
        eval_result = _run_evaluate_docs(params, query, new_context)
        output_summary = f"sufficient={eval_result['sufficient']}"

    elif action == "get_paper_detail":
        result = get_paper_detail(db, params.get("paper_id", ""))
        output_summary = f"paper detail retrieved"

    elif action == "get_paper_chunks":
        result = get_paper_chunks(db, params.get("paper_id", ""), params.get("max_chunks", 10))
        output_summary = f"chunks retrieved"

    else:
        output_summary = f"unknown action: {action}"

    duration = round((time.perf_counter() - t0) * 1000, 2)
    trace = StepTrace(
        node="executor_node",
        action=action,
        input_summary=f"{action}({', '.join(f'{k}={v}' for k, v in list(params.items())[:3])})",
        output_summary=output_summary,
        duration_ms=duration,
    )

    return {
        "plan_step_index": idx + 1,
        "retrieval_context": new_context,
        "step_traces": state["step_traces"] + [trace],
    }
```

- [ ] **Step 4: Run tests, verify pass**

```bash
cd backend && python -m pytest tests/agent/test_executor.py -v
```

- [ ] **Step 5: Commit**

```bash
git add backend/app/agent/nodes/executor.py backend/tests/agent/test_executor.py
git commit -m "feat: implement executor_node with tool dispatching"
```

---

### Task 10: Agent Nodes — synthesis + reflection + final_answer

**Files:**
- Create: `backend/app/agent/nodes/synthesis.py`
- Create: `backend/app/agent/nodes/reflection.py`
- Create: `backend/app/agent/nodes/final_answer.py`
- Create: `backend/tests/agent/test_synthesis_reflection.py`

- [ ] **Step 1: Write tests**

Create `backend/tests/agent/test_synthesis_reflection.py`:

```python
"""Test synthesis, reflection, and final_answer nodes."""
import json
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from app.agent.state import AgentState
from app.agent.nodes.synthesis import synthesis_node
from app.agent.nodes.reflection import reflection_node
from app.agent.nodes.final_answer import final_answer_node


def _base_state(**overrides) -> AgentState:
    defaults = {
        "messages": [],
        "intent": {"type": "simple", "entities": [], "complexity": "low"},
        "plan": [],
        "plan_step_index": 0,
        "retrieval_context": [
            Document(page_content="Attention allows global dependencies", metadata={"paper_id": "1706.03762", "title": "Attention Is All You Need", "page_num": 3}),
        ],
        "step_traces": [],
        "reflection_count": 0,
        "final_answer": None,
    }
    defaults.update(overrides)
    return defaults


def test_synthesis_generates_answer():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Attention 机制允许模型捕获全局依赖 [arxiv:1706.03762]")
    state = _base_state()
    with patch("app.agent.nodes.synthesis._get_llm", return_value=mock_llm):
        result = synthesis_node(state, query="what is attention")

    assert "1706.03762" in result["final_answer"]
    assert len(result["step_traces"]) == 1


def test_reflection_passes():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='{"passed": true, "citation_ok": true, "completeness_ok": true, "logic_ok": true, "issues": [], "fix_strategy": null}'
    )
    state = _base_state(final_answer="Answer with [arxiv:1706.03762]")
    with patch("app.agent.nodes.reflection._get_llm", return_value=mock_llm):
        result = reflection_node(state, query="what is attention")

    assert result["reflection_result"]["passed"] is True


def test_reflection_fails_triggers_re_retrieve():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='{"passed": false, "citation_ok": true, "completeness_ok": false, "logic_ok": true, "issues": ["Missing multi-head details"], "fix_strategy": "re_retrieve"}'
    )
    state = _base_state(final_answer="Incomplete answer", reflection_count=0)
    with patch("app.agent.nodes.reflection._get_llm", return_value=mock_llm):
        result = reflection_node(state, query="explain multi-head attention")

    assert result["reflection_result"]["passed"] is False
    assert result["reflection_result"]["fix_strategy"] == "re_retrieve"
    assert result["reflection_count"] == 1


def test_final_answer_extracts_citations():
    state = _base_state(final_answer="This uses attention [arxiv:1706.03762] and BERT [arxiv:1810.04805]")
    mock_db = MagicMock()
    mock_paper = MagicMock()
    mock_paper.title = "Attention Paper"
    mock_paper.authors = ["Vaswani"]
    mock_paper.year = 2017
    mock_paper.primary_category = "cs.CL"
    mock_paper.doi = None
    mock_db.query.return_value.filter.return_value.one_or_none.return_value = mock_paper

    result = final_answer_node(state, db=mock_db)

    assert len(result["sources"]) == 2
```

- [ ] **Step 2: Run test, verify failure**

```bash
cd backend && python -m pytest tests/agent/test_synthesis_reflection.py -v
```

- [ ] **Step 3: Implement synthesis_node**

Create `backend/app/agent/nodes/synthesis.py`:

```python
"""Reasoning synthesis node: generate cited answer from context."""
from __future__ import annotations

import time

from langchain_openai import ChatOpenAI

from app.agent.prompts.synthesis import SYNTHESIS_PROMPT, SYNTHESIS_WITH_ISSUES_PROMPT
from app.agent.state import AgentState, StepTrace
from app.core.config import get_settings


def _get_llm() -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        model=s.llm_model,
        base_url=s.llm_api_base,
        api_key=s.llm_api_key,
        temperature=0.2,
        max_retries=2,
    )


def _format_context(state: AgentState) -> str:
    parts = []
    for d in state["retrieval_context"]:
        md = d.metadata or {}
        header = f"[arxiv:{md.get('paper_id', '?')} | {md.get('title', '')[:100]} | page={md.get('page_num', '?')}]"
        parts.append(f"{header}\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def synthesis_node(state: AgentState, *, query: str, issues: list[str] | None = None) -> dict:
    """Generate a cited answer from accumulated retrieval context."""
    t0 = time.perf_counter()
    llm = _get_llm()
    context = _format_context(state)

    if issues:
        prompt = SYNTHESIS_WITH_ISSUES_PROMPT.format(
            query=query, context=context, issues="\n".join(f"- {i}" for i in issues)
        )
    else:
        prompt = SYNTHESIS_PROMPT.format(query=query, context=context)

    response = llm.invoke(prompt)
    answer = response.content

    duration = round((time.perf_counter() - t0) * 1000, 2)
    trace = StepTrace(
        node="synthesis_node",
        action="reasoning_synthesis",
        input_summary=f"{len(state['retrieval_context'])} chunks as context",
        output_summary=f"generated {len(answer)} chars",
        duration_ms=duration,
    )
    return {
        "final_answer": answer,
        "step_traces": state["step_traces"] + [trace],
    }
```

- [ ] **Step 4: Implement reflection_node**

Create `backend/app/agent/nodes/reflection.py`:

```python
"""Self-reflection node: verify answer quality."""
from __future__ import annotations

import json
import time

from langchain_openai import ChatOpenAI

from app.agent.prompts.reflection import REFLECTION_PROMPT
from app.agent.state import AgentState, ReflectionResult, StepTrace
from app.core.config import get_settings


def _get_llm() -> ChatOpenAI:
    s = get_settings()
    model = s.reflection_model or s.llm_model
    return ChatOpenAI(
        model=model,
        base_url=s.llm_api_base,
        api_key=s.llm_api_key,
        temperature=0.1,
        max_retries=2,
    )


def reflection_node(state: AgentState, *, query: str) -> dict:
    """Verify answer along 3 dimensions: citation, completeness, logic."""
    t0 = time.perf_counter()
    llm = _get_llm()

    paper_ids = set()
    for d in state["retrieval_context"]:
        pid = (d.metadata or {}).get("paper_id")
        if pid:
            paper_ids.add(pid)

    prompt = REFLECTION_PROMPT.format(
        query=query,
        available_paper_ids=", ".join(sorted(paper_ids)),
        answer=state["final_answer"] or "",
    )
    response = llm.invoke(prompt)

    try:
        result = json.loads(response.content)
        reflection = ReflectionResult(
            passed=bool(result.get("passed", False)),
            citation_ok=bool(result.get("citation_ok", True)),
            completeness_ok=bool(result.get("completeness_ok", True)),
            logic_ok=bool(result.get("logic_ok", True)),
            issues=list(result.get("issues", [])),
            fix_strategy=result.get("fix_strategy"),
        )
    except (json.JSONDecodeError, TypeError):
        reflection = ReflectionResult(
            passed=True, citation_ok=True, completeness_ok=True,
            logic_ok=True, issues=[], fix_strategy=None,
        )

    duration = round((time.perf_counter() - t0) * 1000, 2)
    trace = StepTrace(
        node="reflection_node",
        action="self_reflection",
        input_summary=f"verifying answer ({len(state.get('final_answer', '') or '')} chars)",
        output_summary=f"passed={reflection['passed']}, strategy={reflection.get('fix_strategy')}",
        duration_ms=duration,
    )

    return {
        "reflection_result": reflection,
        "reflection_count": state["reflection_count"] + 1,
        "step_traces": state["step_traces"] + [trace],
    }
```

- [ ] **Step 5: Implement final_answer_node**

Create `backend/app/agent/nodes/final_answer.py`:

```python
"""Final answer node: format output and build citation sources."""
from __future__ import annotations

import re

from sqlalchemy.orm import Session

from app.agent.state import AgentState
from app.models.paper import Paper
from app.schemas.chat import Source


_CITATION_RE = re.compile(r"\[arxiv:([0-9]{4}\.[0-9]{4,6})\]")


def final_answer_node(state: AgentState, *, db: Session) -> dict:
    """Extract citations from answer and build Source objects."""
    answer = state["final_answer"] or ""
    cited_ids = []
    for m in _CITATION_RE.finditer(answer):
        pid = m.group(1)
        if pid not in cited_ids:
            cited_ids.append(pid)

    sources = []
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

    return {"sources": sources}
```

- [ ] **Step 6: Run tests, verify pass**

```bash
cd backend && python -m pytest tests/agent/test_synthesis_reflection.py -v
```

- [ ] **Step 7: Commit**

```bash
git add backend/app/agent/nodes/synthesis.py backend/app/agent/nodes/reflection.py backend/app/agent/nodes/final_answer.py backend/tests/agent/test_synthesis_reflection.py
git commit -m "feat: implement synthesis, reflection, and final_answer nodes"
```

---

### Task 11: Agent Graph Assembly

**Files:**
- Create: `backend/app/agent/graph.py`
- Create: `backend/tests/agent/test_graph.py`

- [ ] **Step 1: Write integration test**

Create `backend/tests/agent/test_graph.py`:

```python
"""Test agent graph compilation and basic flow."""
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from app.agent.graph import build_agent_graph, run_agent_sync


def test_graph_compiles():
    mock_db = MagicMock()
    graph = build_agent_graph(mock_db)
    assert graph is not None


def test_run_agent_sync_returns_response():
    mock_db = MagicMock()
    mock_paper = MagicMock()
    mock_paper.title = "Test"
    mock_paper.authors = ["A"]
    mock_paper.year = 2023
    mock_paper.primary_category = "cs.CL"
    mock_paper.doi = None
    mock_db.query.return_value.filter.return_value.one_or_none.return_value = mock_paper

    # Mock all LLM calls
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        # intent
        MagicMock(content='{"type": "simple", "entities": ["attention"], "complexity": "low"}'),
        # planner
        MagicMock(content='[{"action": "retrieve_local", "params": {"query": "attention", "top_k": 8}, "reason": "search"}, {"action": "evaluate_docs", "params": {}, "reason": "check"}, {"action": "reasoning_synthesis", "params": {}, "reason": "answer"}]'),
        # evaluate_docs
        MagicMock(content='{"sufficient": true, "reason": "ok", "missing_aspects": []}'),
        # synthesis
        MagicMock(content="Attention 是一种机制 [arxiv:1706.03762]"),
        # reflection
        MagicMock(content='{"passed": true, "citation_ok": true, "completeness_ok": true, "logic_ok": true, "issues": [], "fix_strategy": null}'),
    ]

    mock_docs = [(Document(page_content="attention text", metadata={"paper_id": "1706.03762"}), 0.9)]

    with patch("app.agent.nodes.intent._get_llm", return_value=mock_llm), \
         patch("app.agent.nodes.planner._get_llm", return_value=mock_llm), \
         patch("app.tools.evaluate_docs._get_llm", return_value=mock_llm), \
         patch("app.agent.nodes.synthesis._get_llm", return_value=mock_llm), \
         patch("app.agent.nodes.reflection._get_llm", return_value=mock_llm), \
         patch("app.services.retriever.retrieve", return_value=mock_docs):

        result = run_agent_sync(mock_db, "what is attention", session_id="test-session")

    assert result.answer is not None
    assert "1706.03762" in result.answer
```

- [ ] **Step 2: Run test, verify failure**

```bash
cd backend && python -m pytest tests/agent/test_graph.py -v
```

- [ ] **Step 3: Implement graph.py**

Create `backend/app/agent/graph.py`:

```python
"""LangGraph agent graph: build, compile, and run."""
from __future__ import annotations

from typing import Optional

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from sqlalchemy.orm import Session

from app.agent.nodes.executor import executor_node
from app.agent.nodes.final_answer import final_answer_node
from app.agent.nodes.intent import intent_node
from app.agent.nodes.planner import planner_node, re_planner_node
from app.agent.nodes.reflection import reflection_node
from app.agent.nodes.synthesis import synthesis_node
from app.agent.state import AgentState
from app.core.config import get_settings
from app.schemas.chat import ChatResponse, Source


def build_agent_graph(db: Session) -> object:
    """Build and compile the agentic RAG graph."""
    settings = get_settings()

    def _intent(state: AgentState) -> dict:
        query = _extract_query(state)
        return intent_node(state, query=query)

    def _planner(state: AgentState) -> dict:
        query = _extract_query(state)
        return planner_node(state, query=query)

    def _executor(state: AgentState) -> dict:
        return executor_node(state, db=db)

    def _synthesis(state: AgentState) -> dict:
        query = _extract_query(state)
        return synthesis_node(state, query=query)

    def _reflection(state: AgentState) -> dict:
        query = _extract_query(state)
        return reflection_node(state, query=query)

    def _re_planner(state: AgentState) -> dict:
        query = _extract_query(state)
        reflection = state.get("reflection_result", {})
        return re_planner_node(
            state,
            query=query,
            issues=reflection.get("issues", []),
            missing_aspects=reflection.get("missing_aspects", []),
        )

    def _final_answer(state: AgentState) -> dict:
        return final_answer_node(state, db=db)

    def _should_continue_executing(state: AgentState) -> str:
        idx = state["plan_step_index"]
        plan = state["plan"]
        if idx < len(plan) and plan[idx]["action"] != "reasoning_synthesis":
            return "executor"
        return "synthesis"

    def _after_reflection(state: AgentState) -> str:
        reflection = state.get("reflection_result", {})
        if reflection.get("passed", True):
            return "final_answer"
        if state["reflection_count"] >= settings.agent_max_reflections:
            return "final_answer"
        return "re_planner"

    graph = StateGraph(AgentState)

    graph.add_node("intent", _intent)
    graph.add_node("planner", _planner)
    graph.add_node("executor", _executor)
    graph.add_node("synthesis", _synthesis)
    graph.add_node("reflection", _reflection)
    graph.add_node("re_planner", _re_planner)
    graph.add_node("final_answer", _final_answer)

    graph.set_entry_point("intent")
    graph.add_edge("intent", "planner")
    graph.add_edge("planner", "executor")
    graph.add_conditional_edges("executor", _should_continue_executing)
    graph.add_edge("synthesis", "reflection")
    graph.add_conditional_edges("reflection", _after_reflection)
    graph.add_edge("re_planner", "executor")
    graph.add_edge("final_answer", END)

    return graph.compile()


def _extract_query(state: AgentState) -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def run_agent_sync(db: Session, query: str, session_id: str = "", history: list[tuple[str, str]] | None = None) -> ChatResponse:
    """Run the agent synchronously, return ChatResponse."""
    graph = build_agent_graph(db)

    messages = [HumanMessage(content=query)]

    initial_state: AgentState = {
        "messages": messages,
        "intent": None,
        "plan": [],
        "plan_step_index": 0,
        "retrieval_context": [],
        "step_traces": [],
        "reflection_count": 0,
        "final_answer": None,
    }

    config = {"recursion_limit": 25}
    result = graph.invoke(initial_state, config=config)

    answer = result.get("final_answer", "Agent failed to produce an answer.")
    sources = result.get("sources", [])
    step_traces = result.get("step_traces", [])

    return ChatResponse(
        answer=answer,
        sources=sources if isinstance(sources, list) else [],
        used_chunks=len(result.get("retrieval_context", [])),
        step_traces=step_traces,
        reflection_result=result.get("reflection_result"),
    )
```

- [ ] **Step 4: Run tests, verify pass**

```bash
cd backend && python -m pytest tests/agent/test_graph.py -v
```

- [ ] **Step 5: Commit**

```bash
git add backend/app/agent/graph.py backend/tests/agent/test_graph.py
git commit -m "feat: assemble LangGraph agent with all nodes and conditional edges"
```

---

### Task 12: SSE Router + Streaming

**Files:**
- Modify: `backend/app/routers/chat.py`
- Modify: `backend/app/schemas/chat.py`
- Create: `backend/tests/test_chat_router.py`

- [ ] **Step 1: Update schemas/chat.py with new fields**

Add to the existing `ChatResponse` model in `backend/app/schemas/chat.py`:

```python
from typing import Optional

class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    used_chunks: int
    step_traces: Optional[list[dict]] = None
    reflection_result: Optional[dict] = None
```

- [ ] **Step 2: Rewrite routers/chat.py**

Replace `backend/app/routers/chat.py` entirely:

```python
"""Chat router: sync and streaming endpoints."""
from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.agent.graph import run_agent_sync
from app.db.mysql import get_db
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat_sync(req: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    """Synchronous chat endpoint. Runs full agent pipeline, returns final result."""
    return run_agent_sync(db, req.query, session_id=req.session_id or "")


@router.post("/stream")
async def chat_stream(req: ChatRequest, db: Session = Depends(get_db)):
    """Streaming chat endpoint. Emits SSE events for each agent step."""
    from app.agent.graph import build_agent_graph, _extract_query
    from app.agent.state import AgentState
    from langchain_core.messages import HumanMessage

    graph = build_agent_graph(db)
    messages = [HumanMessage(content=req.query)]
    initial_state: AgentState = {
        "messages": messages,
        "intent": None,
        "plan": [],
        "plan_step_index": 0,
        "retrieval_context": [],
        "step_traces": [],
        "reflection_count": 0,
        "final_answer": None,
    }

    def event_generator():
        t_start = time.perf_counter()
        config = {"recursion_limit": 25}
        prev_traces_len = 0

        for step_output in graph.stream(initial_state, config=config):
            for node_name, node_state in step_output.items():
                # Emit new step traces
                traces = node_state.get("step_traces", [])
                for trace in traces[prev_traces_len:]:
                    yield f"event: step_done\ndata: {json.dumps(trace, ensure_ascii=False)}\n\n"
                prev_traces_len = len(traces)

                # Emit intent
                if node_name == "intent" and node_state.get("intent"):
                    yield f"event: intent\ndata: {json.dumps(node_state['intent'], ensure_ascii=False)}\n\n"

                # Emit plan
                if node_name == "planner" and node_state.get("plan"):
                    plan_data = {"steps": node_state["plan"], "total_steps": len(node_state["plan"])}
                    yield f"event: plan\ndata: {json.dumps(plan_data, ensure_ascii=False)}\n\n"

                # Emit reflection
                if node_name == "reflection" and node_state.get("reflection_result"):
                    yield f"event: reflection\ndata: {json.dumps(node_state['reflection_result'], ensure_ascii=False)}\n\n"

                # Emit re_plan
                if node_name == "re_planner" and node_state.get("plan"):
                    yield f"event: re_plan\ndata: {json.dumps({'new_steps': node_state['plan'][-3:]}, ensure_ascii=False)}\n\n"

        # Final answer + sources
        final_result = run_agent_sync(db, req.query, session_id=req.session_id or "")
        yield f"event: token\ndata: {json.dumps({'t': final_result.answer}, ensure_ascii=False)}\n\n"

        if final_result.sources:
            sources_data = [s.model_dump() for s in final_result.sources]
            yield f"event: sources\ndata: {json.dumps({'sources': sources_data}, ensure_ascii=False)}\n\n"

        total_ms = round((time.perf_counter() - t_start) * 1000, 2)
        yield f"event: done\ndata: {json.dumps({'total_ms': total_ms}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

- [ ] **Step 3: Update main.py to use new router**

Ensure `backend/app/main.py` imports from the rewritten `routers/chat.py`. The router signature hasn't changed so no import changes should be needed — verify with:

```bash
cd backend && python -c "from app.main import app; print('OK')"
```

- [ ] **Step 4: Write router test**

Create `backend/tests/test_chat_router.py`:

```python
"""Test chat router endpoints."""
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.chat import ChatResponse, Source

client = TestClient(app)


def test_chat_sync_returns_200():
    mock_response = ChatResponse(
        answer="Test answer [arxiv:1706.03762]",
        sources=[Source(paper_id="1706.03762", title="Test", authors=[], year=2017, arxiv_url="https://arxiv.org/abs/1706.03762")],
        used_chunks=3,
    )
    with patch("app.routers.chat.run_agent_sync", return_value=mock_response):
        resp = client.post("/chat", json={"query": "what is attention", "session_id": "test"})

    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert data["answer"] == "Test answer [arxiv:1706.03762]"
```

- [ ] **Step 5: Run test, verify pass**

```bash
cd backend && python -m pytest tests/test_chat_router.py -v
```

- [ ] **Step 6: Commit**

```bash
git add backend/app/routers/chat.py backend/app/schemas/chat.py backend/tests/test_chat_router.py
git commit -m "feat: rewrite chat router with sync + SSE streaming endpoints"
```

---

## Phase 3: Frontend Rewrite

### Task 13: Frontend Scaffold + Design System

**Files:**
- Rewrite: `frontend/package.json`
- Create: `frontend/tailwind.config.js`
- Create: `frontend/postcss.config.js`
- Rewrite: `frontend/src/main.ts`
- Rewrite: `frontend/src/App.vue`
- Create: `frontend/src/styles/base.css`
- Create: `frontend/src/styles/tailwind.css`
- Create: `frontend/src/types/index.ts`
- Create: `frontend/src/utils/colors.ts`

- [ ] **Step 1: Rewrite package.json**

Replace `frontend/package.json`:

```json
{
  "name": "paperrag-frontend",
  "private": true,
  "version": "2.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vue-tsc --noEmit && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "axios": "^1.7.7",
    "markdown-it": "^14.1.0",
    "pinia": "^2.2.6",
    "vue": "^3.5.13",
    "@headlessui/vue": "^1.7.22"
  },
  "devDependencies": {
    "@vitejs/plugin-vue": "^5.1.4",
    "autoprefixer": "^10.4.20",
    "postcss": "^8.4.47",
    "tailwindcss": "^3.4.14",
    "typescript": "^5.6.3",
    "vite": "^5.4.11",
    "vue-tsc": "^2.1.10"
  }
}
```

- [ ] **Step 2: Create Tailwind config**

Create `frontend/tailwind.config.js`:

```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{vue,js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: { primary: '#FAF9F7', secondary: '#F5F4F0', card: '#FFFFFF' },
        text: { primary: '#1A1A1A', secondary: '#6B6560', tertiary: '#9B9590' },
        accent: { DEFAULT: '#D97706', light: '#FEF3C7' },
        border: '#E8E5E0',
      },
      borderRadius: { card: '12px', sm: '8px' },
      boxShadow: { card: '0 1px 3px rgba(0,0,0,0.04)' },
      fontFamily: {
        sans: ['-apple-system', 'Inter', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
    },
  },
  plugins: [],
}
```

Create `frontend/postcss.config.js`:

```javascript
export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
```

- [ ] **Step 3: Create styles**

Create `frontend/src/styles/tailwind.css`:

```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

Create `frontend/src/styles/base.css`:

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
}

* { box-sizing: border-box; margin: 0; padding: 0; }
html, body, #app { height: 100%; }
body {
  background: var(--bg-primary);
  color: var(--text-primary);
  font-family: -apple-system, 'Inter', sans-serif;
  -webkit-font-smoothing: antialiased;
}
```

- [ ] **Step 4: Create types**

Create `frontend/src/types/index.ts`:

```typescript
export interface StepTrace {
  node: string
  action: string
  input_summary: string
  output_summary: string
  duration_ms: number
}

export interface Source {
  paper_id: string
  title: string
  authors: string[]
  year: number | null
  primary_category?: string
  doi?: string
  arxiv_url: string
  score?: number
  page_num?: number
  snippet?: string
}

export interface SSEIntent {
  type: 'simple' | 'complex' | 'comparison'
  entities: string[]
  complexity: 'low' | 'medium' | 'high'
}

export interface SSEPlan {
  steps: { action: string; reason: string; params?: Record<string, unknown> }[]
  total_steps: number
}

export interface SSEReflection {
  pass: boolean
  citation_ok: boolean
  completeness_ok: boolean
  logic_ok: boolean
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  thinking?: ThinkingStep[]
  timestamp: number
}

export interface ThinkingStep {
  index: number
  action: string
  reason: string
  status: 'pending' | 'running' | 'done' | 'failed'
  outputSummary?: string
  durationMs?: number
}
```

- [ ] **Step 5: Create utils/colors.ts**

Create `frontend/src/utils/colors.ts`:

```typescript
export const palette = {
  bgPrimary: '#FAF9F7',
  bgSecondary: '#F5F4F0',
  bgCard: '#FFFFFF',
  textPrimary: '#1A1A1A',
  textSecondary: '#6B6560',
  textTertiary: '#9B9590',
  accent: '#D97706',
  accentLight: '#FEF3C7',
  border: '#E8E5E0',
} as const
```

- [ ] **Step 6: Rewrite main.ts and App.vue**

Rewrite `frontend/src/main.ts`:

```typescript
import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import './styles/tailwind.css'
import './styles/base.css'

const app = createApp(App)
app.use(createPinia())
app.mount('#app')
```

Rewrite `frontend/src/App.vue`:

```vue
<template>
  <ChatLayout />
</template>

<script setup lang="ts">
import ChatLayout from './layouts/ChatLayout.vue'
</script>
```

- [ ] **Step 7: Install deps and verify**

```bash
cd frontend && rm -rf node_modules package-lock.json && npm install
npm run dev -- --host 0.0.0.0 &
sleep 3 && curl -s http://localhost:5173 | head -5
kill %1
```

Expected: HTML response with `<div id="app">`

- [ ] **Step 8: Commit**

```bash
git add frontend/
git commit -m "feat: scaffold frontend with Tailwind + Claude design system"
```

---

### Task 14: Frontend Core — SSE Composable + Chat Store

**Files:**
- Create: `frontend/src/composables/useSSE.ts`
- Create: `frontend/src/composables/useChat.ts`
- Create: `frontend/src/composables/useThinking.ts`
- Create: `frontend/src/stores/chat.ts`
- Create: `frontend/src/api/client.ts`

- [ ] **Step 1: Create API client**

Create `frontend/src/api/client.ts`:

```typescript
import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE || 'http://localhost:8000',
  timeout: 30000,
})

export default api
```

- [ ] **Step 2: Create useSSE composable**

Create `frontend/src/composables/useSSE.ts`:

```typescript
import { ref } from 'vue'
import type { SSEIntent, SSEPlan, SSEReflection, StepTrace, Source } from '../types'

export type SSEEvent =
  | { type: 'intent'; data: SSEIntent }
  | { type: 'plan'; data: SSEPlan }
  | { type: 'step_done'; data: StepTrace }
  | { type: 'reflection'; data: SSEReflection }
  | { type: 're_plan'; data: { new_steps: unknown[] } }
  | { type: 'token'; data: { t: string } }
  | { type: 'sources'; data: { sources: Source[] } }
  | { type: 'done'; data: { total_ms: number } }
  | { type: 'error'; data: { message: string } }

export function useSSE() {
  const isConnected = ref(false)
  let abortController: AbortController | null = null

  async function* streamChat(query: string, sessionId: string): AsyncGenerator<SSEEvent> {
    abortController = new AbortController()
    isConnected.value = true

    const baseUrl = import.meta.env.VITE_API_BASE || 'http://localhost:8000'
    const response = await fetch(`${baseUrl}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, session_id: sessionId }),
      signal: abortController.signal,
    })

    if (!response.ok || !response.body) {
      isConnected.value = false
      yield { type: 'error', data: { message: `HTTP ${response.status}` } }
      return
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        let eventType = ''
        for (const line of lines) {
          if (line.startsWith('event: ')) {
            eventType = line.slice(7)
          } else if (line.startsWith('data: ') && eventType) {
            try {
              const data = JSON.parse(line.slice(6))
              yield { type: eventType, data } as SSEEvent
            } catch { /* skip malformed */ }
            eventType = ''
          }
        }
      }
    } finally {
      isConnected.value = false
    }
  }

  function abort() {
    abortController?.abort()
    isConnected.value = false
  }

  return { streamChat, abort, isConnected }
}
```

- [ ] **Step 3: Create useThinking composable**

Create `frontend/src/composables/useThinking.ts`:

```typescript
import { ref } from 'vue'
import type { ThinkingStep, SSEPlan, StepTrace } from '../types'

export function useThinking() {
  const steps = ref<ThinkingStep[]>([])
  const isThinking = ref(false)

  function startFromPlan(plan: SSEPlan) {
    isThinking.value = true
    steps.value = plan.steps.map((s, i) => ({
      index: i,
      action: s.action,
      reason: s.reason,
      status: i === 0 ? 'running' : 'pending',
    }))
  }

  function markStepDone(trace: StepTrace) {
    const step = steps.value.find(s => s.status === 'running')
    if (step) {
      step.status = 'done'
      step.outputSummary = trace.output_summary
      step.durationMs = trace.duration_ms
    }
    // Start next pending step
    const next = steps.value.find(s => s.status === 'pending')
    if (next) next.status = 'running'
  }

  function markFailed() {
    const step = steps.value.find(s => s.status === 'running')
    if (step) step.status = 'failed'
  }

  function reset() {
    steps.value = []
    isThinking.value = false
  }

  return { steps, isThinking, startFromPlan, markStepDone, markFailed, reset }
}
```

- [ ] **Step 4: Create chat store**

Create `frontend/src/stores/chat.ts`:

```typescript
import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { Message, Source } from '../types'

export const useChatStore = defineStore('chat', () => {
  const messages = ref<Message[]>([])
  const sessionId = ref(crypto.randomUUID())
  const isLoading = ref(false)

  function addUserMessage(content: string) {
    messages.value.push({
      id: crypto.randomUUID(),
      role: 'user',
      content,
      timestamp: Date.now(),
    })
  }

  function addAssistantMessage(content: string, sources?: Source[]) {
    messages.value.push({
      id: crypto.randomUUID(),
      role: 'assistant',
      content,
      sources,
      timestamp: Date.now(),
    })
  }

  function updateLastAssistant(content: string, sources?: Source[]) {
    const last = messages.value[messages.value.length - 1]
    if (last && last.role === 'assistant') {
      last.content = content
      if (sources) last.sources = sources
    }
  }

  function newConversation() {
    messages.value = []
    sessionId.value = crypto.randomUUID()
  }

  return { messages, sessionId, isLoading, addUserMessage, addAssistantMessage, updateLastAssistant, newConversation }
})
```

- [ ] **Step 5: Create useChat composable**

Create `frontend/src/composables/useChat.ts`:

```typescript
import { useChatStore } from '../stores/chat'
import { useSSE } from './useSSE'
import { useThinking } from './useThinking'
import type { ThinkingStep } from '../types'

export function useChat() {
  const store = useChatStore()
  const { streamChat, abort } = useSSE()
  const thinking = useThinking()

  async function sendMessage(query: string) {
    if (!query.trim() || store.isLoading) return

    store.addUserMessage(query)
    store.isLoading = true
    store.addAssistantMessage('')
    thinking.reset()

    try {
      for await (const event of streamChat(query, store.sessionId)) {
        switch (event.type) {
          case 'plan':
            thinking.startFromPlan(event.data)
            break
          case 'step_done':
            thinking.markStepDone(event.data)
            break
          case 'reflection':
            if (!event.data.pass) thinking.markFailed()
            break
          case 'token':
            store.updateLastAssistant(
              (store.messages[store.messages.length - 1]?.content || '') + event.data.t
            )
            break
          case 'sources':
            store.updateLastAssistant(
              store.messages[store.messages.length - 1]?.content || '',
              event.data.sources
            )
            break
          case 'done':
            thinking.isThinking.value = false
            break
          case 'error':
            store.updateLastAssistant(`Error: ${event.data.message}`)
            break
        }
      }
    } catch (e) {
      store.updateLastAssistant('连接中断，请重试。')
    } finally {
      store.isLoading = false
    }
  }

  return { sendMessage, abort, thinking }
}
```

- [ ] **Step 6: Commit**

```bash
git add frontend/src/composables/ frontend/src/stores/chat.ts frontend/src/api/
git commit -m "feat: implement SSE composable, chat store, and thinking state"
```

---

### Task 15: Frontend Components — Layout + Chat UI

**Files:**
- Create: `frontend/src/layouts/ChatLayout.vue`
- Create: `frontend/src/views/ChatView.vue`
- Create: `frontend/src/components/chat/MessageList.vue`
- Create: `frontend/src/components/chat/UserBubble.vue`
- Create: `frontend/src/components/chat/AssistantBubble.vue`
- Create: `frontend/src/components/chat/ThinkingCard.vue`
- Create: `frontend/src/components/chat/StepIndicator.vue`
- Create: `frontend/src/components/chat/InputArea.vue`
- Create: `frontend/src/components/common/LoadingDots.vue`

- [ ] **Step 1: Create ChatLayout**

Create `frontend/src/layouts/ChatLayout.vue`:

```vue
<template>
  <div class="h-full flex bg-bg-primary">
    <!-- Sidebar -->
    <aside v-if="sidebarOpen" class="w-64 border-r border-border bg-bg-secondary flex flex-col">
      <div class="p-4 border-b border-border">
        <h1 class="text-lg font-semibold text-text-primary">PaperRAG</h1>
      </div>
      <div class="p-3">
        <button @click="newChat" class="w-full px-3 py-2 rounded-sm text-sm text-text-secondary hover:bg-bg-primary transition">
          + New Conversation
        </button>
      </div>
    </aside>

    <!-- Main area -->
    <main class="flex-1 flex flex-col min-w-0">
      <!-- Header -->
      <header class="h-12 border-b border-border flex items-center px-4 bg-bg-card">
        <button @click="sidebarOpen = !sidebarOpen" class="mr-3 text-text-tertiary hover:text-text-primary">
          ☰
        </button>
        <span class="text-sm text-text-secondary">Agentic RAG Paper Assistant</span>
      </header>

      <!-- Chat content -->
      <ChatView />
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import ChatView from '../views/ChatView.vue'
import { useChatStore } from '../stores/chat'

const sidebarOpen = ref(true)
const store = useChatStore()

function newChat() {
  store.newConversation()
}
</script>
```

- [ ] **Step 2: Create ChatView**

Create `frontend/src/views/ChatView.vue`:

```vue
<template>
  <div class="flex-1 flex flex-col overflow-hidden">
    <MessageList :messages="store.messages" :thinking="chat.thinking" />
    <InputArea @send="chat.sendMessage" :disabled="store.isLoading" />
  </div>
</template>

<script setup lang="ts">
import { useChatStore } from '../stores/chat'
import { useChat } from '../composables/useChat'
import MessageList from '../components/chat/MessageList.vue'
import InputArea from '../components/chat/InputArea.vue'

const store = useChatStore()
const chat = useChat()
</script>
```

- [ ] **Step 3: Create MessageList**

Create `frontend/src/components/chat/MessageList.vue`:

```vue
<template>
  <div ref="listRef" class="flex-1 overflow-y-auto px-4 py-6 space-y-4">
    <template v-for="msg in messages" :key="msg.id">
      <UserBubble v-if="msg.role === 'user'" :content="msg.content" />
      <template v-else>
        <ThinkingCard v-if="thinking.isThinking.value || thinking.steps.value.length > 0" :steps="thinking.steps.value" />
        <AssistantBubble :content="msg.content" :sources="msg.sources" />
      </template>
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, watchEffect, nextTick } from 'vue'
import type { Message } from '../../types'
import UserBubble from './UserBubble.vue'
import AssistantBubble from './AssistantBubble.vue'
import ThinkingCard from './ThinkingCard.vue'

const props = defineProps<{
  messages: Message[]
  thinking: { steps: { value: unknown[] }; isThinking: { value: boolean } }
}>()

const listRef = ref<HTMLElement>()

watchEffect(async () => {
  props.messages.length
  await nextTick()
  if (listRef.value) {
    listRef.value.scrollTop = listRef.value.scrollHeight
  }
})
</script>
```

- [ ] **Step 4: Create UserBubble + AssistantBubble**

Create `frontend/src/components/chat/UserBubble.vue`:

```vue
<template>
  <div class="flex justify-end">
    <div class="max-w-[70%] px-4 py-3 rounded-card bg-accent-light text-text-primary text-sm">
      {{ content }}
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{ content: string }>()
</script>
```

Create `frontend/src/components/chat/AssistantBubble.vue`:

```vue
<template>
  <div class="flex justify-start">
    <div class="max-w-[80%] px-4 py-3 rounded-card bg-bg-card shadow-card text-sm text-text-primary prose prose-sm" v-html="rendered"></div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { renderMarkdown } from '../../utils/markdown'
import type { Source } from '../../types'

const props = defineProps<{ content: string; sources?: Source[] }>()

const rendered = computed(() => renderMarkdown(props.content, props.sources))
</script>
```

- [ ] **Step 5: Create ThinkingCard + StepIndicator**

Create `frontend/src/components/chat/StepIndicator.vue`:

```vue
<template>
  <div class="flex items-start gap-2 py-1.5">
    <span class="mt-0.5 text-xs">
      <span v-if="step.status === 'running'" class="animate-spin inline-block">◐</span>
      <span v-else-if="step.status === 'done'" class="text-green-600">●</span>
      <span v-else-if="step.status === 'failed'" class="text-red-500">●</span>
      <span v-else class="text-text-tertiary">○</span>
    </span>
    <div class="flex-1 min-w-0">
      <div class="flex items-center gap-2">
        <span class="text-xs font-medium text-text-primary">{{ actionLabel }}</span>
        <span v-if="step.durationMs" class="text-xs text-text-tertiary">{{ step.durationMs }}ms</span>
        <span v-if="step.status === 'done'" class="text-xs text-green-600">✓</span>
        <span v-if="step.status === 'failed'" class="text-xs text-red-500">✗</span>
      </div>
      <p v-if="step.outputSummary" class="text-xs text-text-secondary mt-0.5 truncate">{{ step.outputSummary }}</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { ThinkingStep } from '../../types'

const props = defineProps<{ step: ThinkingStep }>()

const actionLabels: Record<string, string> = {
  intent_analysis: '意图分析',
  planning: '生成计划',
  query_rewrite: '查询改写',
  retrieve_local: '本地检索',
  retrieve_arxiv: 'arXiv 搜索',
  search_web: '网页搜索',
  evaluate_docs: '充分性评估',
  reasoning_synthesis: '推理生成',
  self_reflection: '自我验证',
  re_planning: '重新规划',
}

const actionLabel = computed(() => actionLabels[props.step.action] || props.step.action)
</script>
```

Create `frontend/src/components/chat/ThinkingCard.vue`:

```vue
<template>
  <div v-if="steps.length > 0" class="ml-0 mb-2">
    <div class="rounded-card border border-border bg-bg-secondary px-4 py-3">
      <div class="flex items-center justify-between mb-2 cursor-pointer" @click="expanded = !expanded">
        <span class="text-xs font-medium text-text-secondary">
          ⚡ Agent {{ isRunning ? '正在思考...' : '思考完成' }}
        </span>
        <span class="text-xs text-text-tertiary">{{ expanded ? '收起 ▴' : '展开 ▾' }}</span>
      </div>
      <div v-if="expanded" class="space-y-0.5">
        <StepIndicator v-for="step in steps" :key="step.index" :step="step" />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import type { ThinkingStep } from '../../types'
import StepIndicator from './StepIndicator.vue'

const props = defineProps<{ steps: ThinkingStep[] }>()
const expanded = ref(true)
const isRunning = computed(() => props.steps.some(s => s.status === 'running'))
</script>
```

- [ ] **Step 6: Create InputArea**

Create `frontend/src/components/chat/InputArea.vue`:

```vue
<template>
  <div class="border-t border-border bg-bg-card px-4 py-3">
    <div class="max-w-3xl mx-auto flex items-end gap-2">
      <textarea
        ref="inputRef"
        v-model="input"
        @keydown.enter.exact.prevent="send"
        :disabled="disabled"
        placeholder="输入你的问题..."
        rows="1"
        class="flex-1 resize-none rounded-sm border border-border bg-bg-primary px-3 py-2 text-sm text-text-primary placeholder:text-text-tertiary focus:outline-none focus:border-accent transition"
      />
      <button
        @click="send"
        :disabled="disabled || !input.trim()"
        class="px-4 py-2 rounded-sm bg-accent text-white text-sm font-medium disabled:opacity-40 hover:bg-amber-700 transition"
      >
        发送
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

const props = defineProps<{ disabled: boolean }>()
const emit = defineEmits<{ send: [query: string] }>()

const input = ref('')
const inputRef = ref<HTMLTextAreaElement>()

function send() {
  if (!input.value.trim() || props.disabled) return
  emit('send', input.value.trim())
  input.value = ''
}
</script>
```

- [ ] **Step 7: Create LoadingDots**

Create `frontend/src/components/common/LoadingDots.vue`:

```vue
<template>
  <span class="inline-flex gap-1">
    <span v-for="i in 3" :key="i" class="w-1.5 h-1.5 rounded-full bg-text-tertiary animate-bounce" :style="{ animationDelay: `${i * 150}ms` }" />
  </span>
</template>
```

- [ ] **Step 8: Verify frontend compiles**

```bash
cd frontend && npm run build
```

Expected: No errors

- [ ] **Step 9: Commit**

```bash
git add frontend/src/
git commit -m "feat: implement Claude-style chat UI components"
```

---

### Task 16: Frontend — Markdown + Citation Rendering

**Files:**
- Create: `frontend/src/utils/markdown.ts`
- Create: `frontend/src/components/citation/CitationPill.vue`
- Create: `frontend/src/components/citation/CitationPopover.vue`

- [ ] **Step 1: Create markdown renderer with citation plugin**

Create `frontend/src/utils/markdown.ts`:

```typescript
import MarkdownIt from 'markdown-it'
import type { Source } from '../types'

const md = new MarkdownIt({ html: false, linkify: true, breaks: true })

const CITATION_RE = /\[arxiv:(\d{4}\.\d{4,6})\]/g

export function renderMarkdown(text: string, sources?: Source[]): string {
  if (!text) return ''

  // Replace [arxiv:ID] with numbered citation pills
  const citedIds: string[] = []
  const processed = text.replace(CITATION_RE, (_, id: string) => {
    if (!citedIds.includes(id)) citedIds.push(id)
    const idx = citedIds.indexOf(id) + 1
    const source = sources?.find(s => s.paper_id === id)
    const title = source?.title || id
    return `<span class="citation-pill" data-paper-id="${id}" data-index="${idx}" title="${title}">[${idx}]</span>`
  })

  return md.render(processed)
}
```

- [ ] **Step 2: Create CitationPill component**

Create `frontend/src/components/citation/CitationPill.vue`:

```vue
<template>
  <span
    class="inline-flex items-center px-1.5 py-0.5 rounded text-xs font-medium bg-accent-light text-accent cursor-pointer hover:bg-amber-200 transition"
    @mouseenter="showPopover = true"
    @mouseleave="showPopover = false"
  >
    [{{ index }}]
    <CitationPopover v-if="showPopover && source" :source="source" />
  </span>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import type { Source } from '../../types'
import CitationPopover from './CitationPopover.vue'

defineProps<{ index: number; source?: Source }>()
const showPopover = ref(false)
</script>
```

- [ ] **Step 3: Create CitationPopover component**

Create `frontend/src/components/citation/CitationPopover.vue`:

```vue
<template>
  <div class="absolute z-50 mt-1 w-72 p-3 rounded-card bg-bg-card shadow-lg border border-border text-left">
    <p class="text-sm font-medium text-text-primary leading-tight">{{ source.title }}</p>
    <p class="text-xs text-text-secondary mt-1">
      {{ source.authors?.slice(0, 3).join(', ') }}
      <span v-if="source.year"> · {{ source.year }}</span>
      <span v-if="source.primary_category"> · {{ source.primary_category }}</span>
    </p>
    <p v-if="source.snippet" class="text-xs text-text-tertiary mt-2 line-clamp-3">
      "{{ source.snippet }}"
    </p>
    <a
      :href="source.arxiv_url"
      target="_blank"
      class="inline-block mt-2 text-xs text-accent hover:underline"
    >
      View on arXiv ↗
    </a>
  </div>
</template>

<script setup lang="ts">
import type { Source } from '../../types'
defineProps<{ source: Source }>()
</script>
```

- [ ] **Step 4: Verify build**

```bash
cd frontend && npm run build
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src/utils/markdown.ts frontend/src/components/citation/
git commit -m "feat: implement markdown rendering with citation pills and popovers"
```

---

## Phase 4: Integration + Polish

### Task 17: End-to-End Integration Test

**Files:**
- Create: `backend/tests/test_e2e.py`

- [ ] **Step 1: Write E2E test**

Create `backend/tests/test_e2e.py`:

```python
"""End-to-end integration test: full agent flow."""
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.main import app

client = TestClient(app)


def test_full_agent_flow_sync():
    """Test /chat endpoint runs complete agent pipeline."""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        MagicMock(content='{"type": "simple", "entities": ["transformer"], "complexity": "low"}'),
        MagicMock(content='[{"action": "retrieve_local", "params": {"query": "transformer architecture", "top_k": 8}, "reason": "search"}, {"action": "evaluate_docs", "params": {}, "reason": "check"}, {"action": "reasoning_synthesis", "params": {}, "reason": "answer"}]'),
        MagicMock(content='{"sufficient": true, "reason": "good", "missing_aspects": []}'),
        MagicMock(content="Transformer 使用 self-attention 机制 [arxiv:1706.03762]"),
        MagicMock(content='{"passed": true, "citation_ok": true, "completeness_ok": true, "logic_ok": true, "issues": [], "fix_strategy": null}'),
    ]

    mock_docs = [(Document(page_content="transformer content", metadata={"paper_id": "1706.03762", "title": "Attention"}), 0.9)]
    mock_paper = MagicMock()
    mock_paper.title = "Attention Is All You Need"
    mock_paper.authors = ["Vaswani"]
    mock_paper.year = 2017
    mock_paper.primary_category = "cs.CL"
    mock_paper.doi = None

    with patch("app.agent.nodes.intent._get_llm", return_value=mock_llm), \
         patch("app.agent.nodes.planner._get_llm", return_value=mock_llm), \
         patch("app.tools.evaluate_docs._get_llm", return_value=mock_llm), \
         patch("app.agent.nodes.synthesis._get_llm", return_value=mock_llm), \
         patch("app.agent.nodes.reflection._get_llm", return_value=mock_llm), \
         patch("app.services.retriever.retrieve", return_value=mock_docs), \
         patch("app.agent.nodes.final_answer.Paper") as MockPaper:
        MockPaper.query = MagicMock()

        resp = client.post("/chat", json={"query": "explain transformer", "session_id": "e2e-test"})

    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert len(data["answer"]) > 0


def test_stream_endpoint_returns_sse():
    """Test /chat/stream returns SSE content type."""
    with patch("app.routers.chat.build_agent_graph") as mock_graph, \
         patch("app.routers.chat.run_agent_sync") as mock_sync:
        mock_compiled = MagicMock()
        mock_compiled.stream.return_value = iter([])
        mock_graph.return_value = mock_compiled
        mock_sync.return_value = MagicMock(
            answer="test", sources=[], used_chunks=0,
            step_traces=[], reflection_result=None
        )

        resp = client.post("/chat/stream", json={"query": "test", "session_id": "test"})

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")
```

- [ ] **Step 2: Run E2E test**

```bash
cd backend && python -m pytest tests/test_e2e.py -v
```

- [ ] **Step 3: Fix any issues discovered**

Address any import errors or type mismatches found during integration.

- [ ] **Step 4: Commit**

```bash
git add backend/tests/test_e2e.py
git commit -m "test: add end-to-end integration tests for agent pipeline"
```

---

### Task 18: Docker + Dev Environment Verification

**Files:**
- Modify: `docker-compose.yml` (if needed for new env vars)
- Modify: `.env.example`

- [ ] **Step 1: Update .env.example**

Add new keys to `.env.example`:

```bash
# Agent
AGENT_MAX_PLAN_STEPS=7
AGENT_MAX_REFLECTIONS=2

# Web Search (Tavily)
TAVILY_API_KEY=

# arXiv
ARXIV_MAX_RESULTS=5

# Optional: separate models for agent nodes
PLANNER_MODEL=
REFLECTION_MODEL=
```

- [ ] **Step 2: Start services and verify**

```bash
docker compose up -d mysql qdrant
cd backend && uvicorn app.main:app --reload --port 8000 &
sleep 3
curl -s http://localhost:8000/health | python -m json.tool
curl -s -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"query": "test", "session_id": "verify"}' | python -m json.tool
kill %1
```

- [ ] **Step 3: Start frontend and verify**

```bash
cd frontend && npm run dev &
sleep 3
curl -s http://localhost:5173 | grep -c "app"
kill %1
```

- [ ] **Step 4: Commit**

```bash
git add .env.example docker-compose.yml
git commit -m "chore: update env config and verify dev environment"
```

---

### Task 19: Run All Tests + Final Verification

- [ ] **Step 1: Run full backend test suite**

```bash
cd backend && python -m pytest tests/ -v --tb=short
```

Expected: All tests pass

- [ ] **Step 2: Run frontend build**

```bash
cd frontend && npm run build
```

Expected: No errors

- [ ] **Step 3: Manual smoke test**

Start the full stack and verify:
1. Frontend loads at `localhost:5173`
2. Type a question and submit
3. ThinkingCard appears with step indicators
4. Answer renders with citation pills
5. No console errors

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore: final verification pass - all tests green"
```
