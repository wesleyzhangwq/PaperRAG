# Cite Scope — Agent 上下文

## 项目简介

Cite Scope 是一个 Agentic RAG 学术论文问答系统。LangGraph 13 节点 Agent，主链为 `guard → intent → complexity_router → (planner 或 fast_local 单步计划) → route → executor ⟲ → evidence → sufficiency → synthesis → groundedness → citation_gate → presentation`，另有 `re_planner` 补充检索环。默认 `AGENT_ROUTING_MODE=full_agentic`，始终保留 planner；`auto` 只作为显式启用的保守快路径。无论选择哪条路径，证据充分性、groundedness 与 citation gate 都不会绕过。

## 快速理解项目必读

1. `AGENTS.md` — 架构全貌、目录结构、SSE 协议、设计原则、贡献规范
2. `backend/app/agent/state.py` — 核心数据结构（AgentState、StepSpec、StepTrace、ReflectionResult）
3. `backend/app/agent/graph.py` — 流程编排（节点连接、条件路由、run_agent_sync 入口）

按需阅读：
- `backend/app/core/config.py` — 所有配置项（改行为时必看）
- `backend/app/schemas/chat.py` — API 请求/响应契约（改接口时必看）
- `eval/README.md` — 评估框架（跑测试/调参时必看）

## 技术栈

- 后端：Python 3.12 / FastAPI / LangGraph / SQLAlchemy / Pydantic
- LLM：MiniMax M2.7（OpenAI 兼容 API）
- Embedding：SiliconFlow BAAI/bge-m3
- 向量库：Qdrant（向量 + BM25 混合检索）
- 关系库：MySQL 8.0
- 前端：Vue 3 / Vite / Tailwind CSS / Pinia
- 流式：SSE via LangGraph `astream_events` + custom event adapter

## 关键约定

- Agent 节点是纯函数：`(state, **kwargs) → partial state dict`，不含 HTTP 副作用；节点用 `stages.stage()` 自播 SSE stage 事件（图外自动 no-op）
- complexity router 只在显式 `auto` 模式下允许高置信、低风险问题跳过初始 planner，并生成恰好一个本地检索计划；证据不足或后续 groundedness 触发补充检索时标记为 `fast_escalated`
- 一个工具 = `backend/app/tools/` 下一个文件，executor 按 action name 分发；plan 只含检索类动作（evaluate_docs/reasoning_synthesis 已是图级节点）
- 解析 LLM 输出的 JSON 必须用 `app/utils/llm_json.py` 的 `extract_json`（推理模型带 `<think>` 前缀，裸 json.loads 会静默降级）
- 测试 mock LLM 和 DB，不调真实 API。synthesis 用 `mock_llm.stream.return_value`（不是 `.invoke`）
- 前端 SSE 事件处理集中在 `frontend/src/composables/useChat.ts` 的 switch 语句；timeline 按稳定 id upsert（`utils/timeline.ts`）
- complexity router 的 `dev50`/`frozen200` 延迟与质量评估尚未运行；在冻结评估验收前，不得默认启用 `auto`，也不得宣称性能或质量收益
- 提交信息用英文，格式：`feat:/fix:/chore:/docs:` + 简述
