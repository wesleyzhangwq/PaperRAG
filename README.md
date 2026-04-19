# PaperRAG

本地全链路学术论文 RAG 系统：arXiv 论文 → MySQL + Chroma → Ollama (gemma4:e4b + bge-m3) → FastAPI → Vue3 三栏 UI。

## 技术栈

- 后端：Python 3.11 + FastAPI + LangChain + SQLAlchemy
- 向量库：Chroma 0.5（langchain-chroma）
- 关系库：MySQL 8.0（Docker）
- LLM / Embedding：Ollama 本地（`gemma4:e4b` / `bge-m3`）
- 解析：Unstructured（fallback pdfplumber + PyMuPDF）
- 前端：Vue 3.4 + Vite 5 + TypeScript + Naive UI + Pinia

## 快速开始

### 0. 先决条件

```bash
# 宿主机安装 Ollama 并拉模型
ollama pull gemma4:e4b      # 9.6 GB
ollama pull bge-m3           # ~1.2 GB
ollama serve                 # 确保 :11434 可访问

# 起 MySQL
docker compose up -d mysql
```

### 1. 拉取 50 篇 arXiv 论文（Pilot）

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_arxiv.py --limit 50
python scripts/ingest.py
```

### 2. 启动后端

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# 打开 http://localhost:8000/docs
```

### 3. 启动前端

```bash
cd frontend
npm install
npm run dev
# 打开 http://localhost:5173
```

### 4. 一键全量（Docker）

```bash
cp .env.example .env
docker compose up -d
# http://localhost:8080
```

## 架构

```
User Query ──► FastAPI /chat ──► Chroma (vec+metadata) ──► top-k chunks
                                                               │
                                  ──► MySQL (papers/chunks) ◄──┘
                                                               │
                                                               ▼
                              Ollama gemma4:e4b (with citation prompt)
                                                               │
                                                               ▼
                                 {answer, sources[paper_id,title,doi,score]}
                                                               │
                                                               ▼
                               Vue3 三栏：PaperList | ChatWindow | CitationCard
```

## 迭代路线

- v2：BM25 Hybrid + CrossEncoder 重排序 + Chunk 策略 Ablation
- v3：用户上传 PDF 自动入库 + 持久化查询历史
- v4：Agent（自动生成文献综述 / MindMap / 引用图谱）

## 许可证

MIT
