# PaperRAG — Agent 上下文

## 项目简介

Agentic RAG 学术论文问答系统。LangGraph 8 节点 Agent（intent → planner → executor → synthesis → reflection → re_planner → final_answer → presentation），7 个工具，SSE 实时流式。

## 快速理解项目必读

1. `AGENTS.md` — 架构全貌、目录结构、SSE 协议、设计原则、贡献规范
2. `backend/app/agent/state.py` — 核心数据结构（AgentState、StepSpec、StepTrace、ReflectionResult）
3. `backend/app/agent/graph.py` — 流程编排（节点连接、条件路由、run_agent_sync 入口）

按需阅读：
- `backend/app/core/config.py` — 所有配置项（改行为时必看）
- `backend/app/schemas/chat.py` — API 请求/响应契约（改接口时必看）
- `eval/README.md` — 评估框架（跑测试/调参时必看）

## 技术栈

- 后端：Python 3.11 / FastAPI / LangGraph / SQLAlchemy / Pydantic
- LLM：MiniMax M2.7（OpenAI 兼容 API）
- Embedding：阿里 text-embedding-v4
- 向量库：Qdrant（向量 + BM25 混合检索）
- 关系库：MySQL 8.0
- 前端：Vue 3 / Vite / Tailwind CSS / Pinia
- 流式：SSE via thread + Queue

## 关键约定

- Agent 节点是纯函数：`(state, **kwargs) → partial state dict`，不含 HTTP/SSE 副作用
- 一个工具 = `backend/app/tools/` 下一个文件，executor 按 action name 分发
- 测试 mock LLM 和 DB，不调真实 API。synthesis 用 `mock_llm.stream.return_value`（不是 `.invoke`）
- 前端 SSE 事件处理集中在 `frontend/src/composables/useChat.ts` 的 switch 语句
- 提交信息用英文，格式：`feat:/fix:/chore:/docs:` + 简述
