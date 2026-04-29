# PaperRAG

本地全链路学术论文 RAG 系统：arXiv 论文 → MySQL + Qdrant → 云端 LLM/Embedding API（MiniMax + 阿里）→ FastAPI → Vue3 三栏 UI。

## 技术栈

- 后端：Python 3.11+ + FastAPI + LangChain + SQLAlchemy
- 向量库：Qdrant
- 关系库：MySQL 8.0（Docker）
- LLM：MiniMax 官方 API（`MiniMax-M2.7`）
- Embedding：阿里官方 API（`text-embedding-v4`）
- 解析：pdfplumber + PyMuPDF（可选 Unstructured）
- 前端：Vue 3.4 + Vite 5 + TypeScript + Naive UI + Pinia

## 快速开始

### 0. 先决条件

```bash
# 起 MySQL + Qdrant（本地开发）
docker compose up -d mysql qdrant
```
复制环境变量：`cp .env.example .env`，填写 `LLM_API_KEY` 与 `EMBEDDING_API_KEY`，并确认 `MYSQL_HOST`、`QDRANT_URL` 等与你的部署一致。

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

### 5. 从现有 chunks 重建 Qdrant 向量索引

```bash
cd backend
python scripts/rebuild_vectors.py
# 可选：只重建某篇论文
python scripts/rebuild_vectors.py --paper-id 2401.01234
```

### 6. Chunk 策略参数（可选）

可在 `.env` 调整以下参数后重新 ingest：

- `CHUNK_STRATEGY=v2`：默认文档级切分（`v1` 为旧版逐页切分，`v3` 为结构感知+动态切分）
- `CHUNK_SIZE` / `CHUNK_OVERLAP`：切分窗口和重叠
- `CHUNK_MIN_CHARS`：最小 chunk 长度过滤
- `CHUNK_NOISE_SYMBOL_RATIO`：噪音符号比例阈值
- `CHUNK_DROP_REFERENCES`：是否丢弃 References 区域

## 架构

```
User Query ──► FastAPI /chat ──► Qdrant (vec+metadata) ──► top-k chunks
                                                         │
                            ──► MySQL (papers/chunks) ◄──┘
                                                         │
                                                         ▼
                        MiniMax-M2.7 (with citation prompt)
                                                         │
                                                         ▼
                           {answer, sources[paper_id,title,doi,score]}
                                                         │
                                                         ▼
                         Vue3 三栏：PaperList | ChatWindow | CitationCard
```

### 架构文档（新人速览）

- 新人快速理解全链路与关键模块：`docs/architecture-quickstart.md`
- Chunk 策略设计与回滚说明：`docs/chunk-strategy-v2.md`

## 迭代路线

- v2：BM25 Hybrid + CrossEncoder 重排序 + Chunk 策略 Ablation
- v3：用户上传 PDF 自动入库 + 持久化查询历史
- v4：Agent（自动生成文献综述 / MindMap / 引用图谱）

## 许可证

MIT
