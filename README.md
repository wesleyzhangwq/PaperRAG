# Cite Scope

<p align="center">
  <strong>Agentic research assistant for cited paper Q&A</strong>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> |
  <a href="#architecture">Architecture</a> |
  <a href="#english">English</a> |
  <a href="LICENSE">MIT License</a>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white">
  <img alt="Vue" src="https://img.shields.io/badge/Vue-3-42b883?logo=vuedotjs&logoColor=white">
  <img alt="LangGraph" src="https://img.shields.io/badge/LangGraph-Agentic_RAG-1f2937">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-blue">
</p>

Cite Scope 是一个面向学术论文问答的 **Agentic RAG** 系统。它通过 arXiv ID/URL 拉取官方 metadata 和 PDF，解析入库后用 Qdrant + MySQL 管理论文、片段和会话，再用 LangGraph Agent 进行意图分析、动态规划、多源检索、资料充分性评估、自我反思和带引用回答生成。

这个项目的目标不是做一个最短链路的聊天壳，而是展示一个可解释、可调试、可扩展的论文研究助手：用户能看到系统检索了什么、为什么补充检索、哪些论文被引用、回答可信度为什么高或低。

---

## Highlights

- **Agentic workflow**: 12 个 LangGraph 节点，覆盖 `guard -> intent -> planner -> route -> executor -> evidence -> sufficiency -> synthesis -> groundedness -> citation_gate -> presentation`，并通过 `re_planner` 做有界补充检索。
- **Adaptive retrieval plan**: Planner 根据问题复杂度生成不同检索计划，而不是固定跑一条 RAG pipeline。
- **Hybrid retrieval**: Qdrant dense vector search 与 BM25 sparse ranking 融合，支持 oversampling、alpha 权重和检索缓存。
- **Self-verification**: `sufficiency` 阶段调用 `evaluate_docs` 检查证据充分性，`groundedness` 检查引用、完整性与逻辑；失败后按预算触发补充检索或重新生成。
- **Transparent execution**: 前端通过 SSE 展示每一步耗时、参数、结果摘要、检索片段和调试详情。
- **Corpus overview**: 新对话和论文库页面可以展示当前 RAG 语料库的主题分布、代表论文和建议问题。
- **Cited answers**: 回答保留论文来源卡片和 citation popover，便于回到 arXiv / PDF 证据。
- **Operational basics**: SQLite LangGraph checkpoint、会话历史、反馈接口、异步 arXiv 导入任务、API key 鉴权和限流中间件。
- **Optional web search**: Tavily 作为可选补充检索，未配置或网络失败时会降级，不阻断本地 RAG。

---

## Product Surface

| Area | What it does |
| --- | --- |
| Chat | 提问、流式回答、查看执行步骤、引用来源和调试详情 |
| Papers | 浏览已入库论文、检索论文、查看语料库主题 overview |
| Uploads | 输入 arXiv ID 或 URL，后台拉取 metadata/PDF 并异步入库 |
| Settings | 查看后端连接、模型和基础配置状态 |
| Feedback | 对回答标记有帮助 / 需改进，为后续评估闭环保留数据 |

---

## Architecture

```text
User query + chat history
        |
        v
  guard
  - empty/oversize/injection checks
        |
        v
  intent
        |
        v
  planner
        |
        v
  route
  - local-first source policy
        |
        v
  executor loop
  - retrieve_local
  - retrieve_arxiv
  - search_web
  - query_rewrite
  - get_paper_detail
  - get_paper_chunks
        |
        v
  evidence
  - dedupe/rerank/context budget
        |
        v
  sufficiency
    | insufficient + budget
    +-----------------------> re_planner -> executor
    |
    v
  synthesis (streaming)
        |
        v
  groundedness
    | re-retrieve ---------> re_planner -> executor
    | re-generate ---------> synthesis
    v
  citation_gate
  - resolve and strip unverifiable citations
        |
        v
  presentation -> SSE/UI
```

### Backend

```text
backend/app/
├── agent/          LangGraph state, graph, checkpoint, streaming, nodes
├── tools/          Retrieval, web search, evaluation, paper lookup tools
├── routers/        FastAPI routes for chat, papers, uploads, ingest, feedback
├── services/       PDF ingestion and hybrid retrieval engine
├── db/             MySQL and Qdrant clients
├── models/         SQLAlchemy ORM models
├── schemas/        Pydantic API contracts
└── middleware/     Rate limit, API key auth, request context
```

### Frontend

```text
frontend/src/
├── components/     Chat, answer cards, source cards, corpus overview
├── composables/    SSE and chat orchestration
├── stores/         Pinia state for chat, conversations, theme
├── api/            Typed API clients
├── views/          Chat, papers, uploads, settings
└── utils/          Markdown, durations, thinking-step detail mapping
```

### Storage

| Storage | Purpose |
| --- | --- |
| MySQL 8 | Paper metadata, chunks, conversations, chat history, upload jobs, feedback |
| Qdrant | Dense vectors for paper chunks |
| BM25 cache | Sparse retrieval over chunk text for hybrid ranking |
| SQLite checkpoint | LangGraph thread checkpoint state |
| Local `data/` | PDFs, metadata JSON, checkpoint files and transient artifacts |

---

## Quick Start

### 1. Start infrastructure

```bash
docker compose up -d mysql qdrant
cp .env.example .env
```

Edit `.env` and fill at least:

```bash
LLM_API_KEY=...
EMBEDDING_API_KEY=...
```

Current defaults use:

```bash
LLM_MODEL=MiniMax-M2.7
LLM_API_BASE=https://api.minimax.chat/v1
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_API_BASE=https://api.siliconflow.cn/v1
```

Optional:

```bash
TAVILY_API_KEY=...        # web search supplement
API_AUTH_ENABLED=true     # enable API key auth for deployment
API_KEYS=your-key-1,...
```

### 2. Run backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

`/health` 是进程存活检查，不会调用外部 LLM/Embedding 服务，因此不能证明 API Key 有效。完整就绪验证还应发起一次实际问答，并确认响应中的 `degraded` 为 `false`；若为 `true`，先检查对应供应商凭据与额度。

### 3. Build a paper corpus

Small recent arXiv corpus:

```bash
cd backend
python scripts/download_arxiv.py --limit 50
python scripts/ingest.py
```

Focused AI landmark corpus:

```bash
cd backend
python scripts/curate_ai_landmark_corpus.py --target 500 --workers 6
python scripts/ingest.py --force
```

The curation script writes `data/metadata_filtered.json` and downloads available PDFs into `data/pdfs/`. Papers without accessible PDFs are recorded in `data/raw_metadata/ai_landmark_skipped.json` and are not ingested until a PDF is supplied.

### 4. Run frontend

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

### 5. Docker full stack

```bash
cp .env.example .env
# fill API keys first
docker compose up -d --build
```

修改 `.env` 后 Compose 会重建受影响的容器；修改后端或前端源码后必须保留 `--build`，否则可能继续运行旧镜像。若只需更新后端，可执行 `docker compose up -d --build backend`。

Frontend:

```text
http://localhost:8080
```

Backend:

```text
http://localhost:8000
```

---

## Configuration

Important `.env` groups:

| Group | Keys |
| --- | --- |
| LLM | `LLM_MODEL`, `LLM_API_BASE`, `LLM_API_KEY`, optional `PLANNER_MODEL`, `REFLECTION_MODEL` |
| Embedding | `EMBEDDING_MODEL`, `EMBEDDING_API_BASE`, `EMBEDDING_API_KEY` |
| Retrieval | `RETRIEVAL_K`, `FINAL_CONTEXT_K`, `HYBRID_ALPHA`, `HYBRID_OVERSAMPLE`, cache settings |
| Agent | `AGENT_MAX_PLAN_STEPS`, `AGENT_MAX_REFLECTIONS`, `AGENT_CHECKPOINT_ENABLED`, `AGENT_CHECKPOINT_PATH` |
| Search | `TAVILY_API_KEY`, `ARXIV_MAX_RESULTS` |
| Safety | `RATE_LIMIT_ENABLED`, `API_AUTH_ENABLED`, `API_KEYS`, auth exempt paths |
| Data | `DATA_DIR`, `PDF_DIR`, `METADATA_JSON` |

See [.env.example](.env.example) for the full list.

---

## API Overview

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | backend health check |
| `POST /chat` | synchronous chat response |
| `POST /chat/stream` | SSE streaming chat response |
| `GET /conversations` | list conversations |
| `GET /conversations/{id}/messages` | load persisted messages |
| `GET /papers` | list/search ingested papers |
| `GET /papers/overview` | corpus topic overview |
| `POST /upload/arxiv` | queue arXiv ID/URL import and background ingest |
| `POST /upload` | disabled legacy local PDF upload endpoint |
| `GET /upload/jobs` | list upload jobs |
| `POST /feedback` | store answer feedback |
| `POST /ingest` | admin ingestion endpoint |

FastAPI docs are available at:

```text
http://localhost:8000/docs
```

---

## Validation

Backend:

```bash
backend/.venv/bin/python -m pytest -q
```

Frontend:

```bash
cd frontend
npm test
npm run build
```

The test suite mocks LLM and database boundaries. It should not call real LLM, embedding, Tavily, arXiv or Qdrant services.

---

## Documentation

| File | Description |
| --- | --- |
| [AGENTS.md](AGENTS.md) | Architecture, design philosophy, agent flow and contributor conventions |
| [CLAUDE.md](CLAUDE.md) | Short agent context for future coding sessions |
| [eval/README.md](eval/README.md) | Evaluation framework and retrieval tuning notes |
| [docs/cite-scope-java-architecture-overview.md](docs/cite-scope-java-architecture-overview.md) | High-level architecture review and roadmap status |
| [docs/cite-scope-defect-verification.md](docs/cite-scope-defect-verification.md) | Defect verification notes |
| [docs/cite-scope-update-plan.md](docs/cite-scope-update-plan.md) | Iteration plan |
| [docs/cite-scope-execution-acceptance.md](docs/cite-scope-execution-acceptance.md) | Acceptance notes |

---

## Roadmap

- Stronger feedback consumption: connect answer feedback to evaluation/admin review.
- Richer corpus management: deduplication, import/export, and source-level quality controls.
- Production hardening: multi-instance rate limiting, stricter auth defaults, migrations and deployment profiles.
- Evaluation dashboards: trend reports for retrieval quality, citation faithfulness and answer confidence.

---

## Notes

- This repository does not commit PDFs, vector data, database files or API keys.
- You need your own LLM and embedding API keys before the full RAG path can run.
- Tavily web search is optional. Local retrieval remains the primary path.
- API authentication is configurable and disabled by default for local development. Enable it before public deployment.
- Some legacy internal identifiers for existing databases, collections, metrics, packages and browser storage are intentionally retained for compatibility.

---

## English

Cite Scope is an **Agentic RAG system for academic paper Q&A**. It imports arXiv metadata and PDFs from arXiv IDs or URLs, stores paper metadata and chunks in MySQL, indexes chunk embeddings in Qdrant, and uses a LangGraph agent to plan retrieval, evaluate evidence, synthesize cited answers, and self-reflect before returning the final response.

### Why this project exists

Cite Scope is designed as a portfolio-grade, explainable research assistant. Instead of hiding retrieval behind a black box, it exposes execution steps, retrieved chunks, source cards, citation details, confidence reasons and debug traces in the UI.

### Key capabilities

- A 12-node LangGraph workflow with guardrails, routing, evidence processing, sufficiency checks, groundedness verification, citation gating and presentation.
- Hybrid dense + BM25 retrieval with tunable ranking parameters.
- Optional Tavily web search for evidence supplementation.
- SSE streaming for live answer tokens and execution-step updates.
- Persistent conversations, chat history and LangGraph checkpoints.
- arXiv ID/URL import queue with background ingestion.
- Corpus overview for topic buckets, representative papers and suggested questions.
- Vue 3 frontend with source cards, citation popovers and debug panels.

### Tech stack

| Layer | Technology |
| --- | --- |
| Agent | LangGraph |
| Backend | FastAPI, SQLAlchemy, Pydantic |
| Frontend | Vue 3, Vite, Tailwind CSS, Pinia |
| LLM | MiniMax M2.7 through an OpenAI-compatible API |
| Embedding | SiliconFlow BAAI/bge-m3 by default |
| Vector DB | Qdrant |
| SQL DB | MySQL 8 |
| Streaming | Server-Sent Events |

### Quick start

```bash
docker compose up -d mysql qdrant
cp .env.example .env
# fill LLM_API_KEY and EMBEDDING_API_KEY
```

Backend:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Ingest sample papers:

```bash
cd backend
python scripts/download_arxiv.py --limit 50
python scripts/ingest.py
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

### License

Cite Scope is released under the [MIT License](LICENSE).
