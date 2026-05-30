# PaperRAG

Agentic RAG 学术论文问答系统：arXiv 论文 → MySQL + Qdrant → LLM 自适应规划 Agent（意图分析 → 多源检索 → 自我反思 → 带引用回答）→ Vue 3 实时思考过程可视化。

## 技术栈

- **Agent**：LangGraph StateGraph（8 节点 + 条件边）
- **后端**：Python 3.11+ / FastAPI / SQLAlchemy / Pydantic
- **LLM**：MiniMax M2.7（OpenAI 兼容 API）
- **Embedding**：阿里 text-embedding-v4（DashScope）
- **向量库**：Qdrant（向量 + BM25 混合检索）
- **关系库**：MySQL 8.0
- **前端**：Vue 3 + Vite + Tailwind CSS + Pinia
- **流式**：SSE（Server-Sent Events），实时展示 Agent 思考过程

## 快速开始

### 1. 环境准备

```bash
docker compose up -d mysql qdrant
cp .env.example .env
# 填写 LLM_API_KEY 和 EMBEDDING_API_KEY
```

### 2. 后端

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 下载论文 + 入库
python scripts/download_arxiv.py --limit 50
python scripts/ingest.py

# 启动
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. 前端

```bash
cd frontend
npm install
npm run dev
# http://localhost:5173
```

### 4. Docker 一键部署

```bash
docker compose up -d
# http://localhost:8080
```

## 架构

```
User Query + Chat History
        │
        ▼
   ┌─────────┐
   │  intent   │  分析意图：type / entities / complexity
   └────┬──────┘
        ▼
   ┌─────────┐
   │ planner   │  生成执行计划 (retrieve / rewrite / evaluate / search...)
   └────┬──────┘
        ▼
   ┌─────────┐ ◄── re_planner (反思失败时补充检索)
   │ executor  │  调度 7 个工具
   └────┬──────┘
        ▼
   ┌───────────┐
   │ synthesis   │  从多源上下文生成带引用答案 (streaming)
   └────┬───────┘
        ▼
   ┌────────────┐
   │ reflection   │  三维验证：引用忠实 / 完整性 / 逻辑一致
   └────┬────────┘
        │ pass → final_answer → presentation → END
        │ fail → re_planner 或 re_generate
```

前端通过 SSE 实时展示每个步骤的执行状态（ThinkingCard），引用以 Popover 形式展示论文详情。

## 文档

| 文档 | 内容 |
|------|------|
| [`AGENTS.md`](AGENTS.md) | 架构设计、目录结构、SSE 协议、Reflexion 模式、贡献规范 |
| [`eval/README.md`](eval/README.md) | 评估框架：70 题语义数据集、消融实验、指标说明 |
| [`eval/EVALUATION_REPORT.md`](eval/EVALUATION_REPORT.md) | 检索调优报告：Baseline vs Optimized 对比、参数消融数据 |

## 许可证

MIT
