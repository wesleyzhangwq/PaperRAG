# PaperRAG 更新计划文档

生成时间：2026-06-04

目标：把体检报告中确认存在的缺陷拆成可验收迭代。本轮优先修复“输出质量会出错、成本会被刷爆、产品闭环缺失”的问题；大型架构替换保留为后续阶段。

## 已完成更新

### 1. 反思链路修复

- 修复 reflection JSON 解析失败静默通过的问题。
- reflection 读取 `evaluator_result`，当 evaluator 明确判定资料不足时直接触发重检索。
- 新增回归测试：
  - `test_reflection_parse_failure_triggers_retry`
  - `test_reflection_respects_insufficient_evaluator_result`

### 2. 公共 API 保护

- 新增 `FixedWindowRateLimitMiddleware`。
- 新增 `APIKeyAuthMiddleware`。
- 新增配置项：
  - `RATE_LIMIT_ENABLED`
  - `RATE_LIMIT_REQUESTS`
  - `RATE_LIMIT_WINDOW_SECONDS`
  - `RATE_LIMIT_PATHS`
  - `API_AUTH_ENABLED`
  - `API_KEYS`
  - `API_AUTH_EXEMPT_PATHS`
- 默认限流保护 `/chat`、`/upload`；API Key 鉴权默认关闭，生产环境可显式开启。

### 3. 答案反馈闭环

- 新增 `answer_feedback` 表模型。
- 新增 `POST /feedback`。
- 新增 `GET /feedback` 管理查询接口，供后续 eval/admin 消费。
- 前端 `AnswerCard` 增加“有帮助 / 需改进”反馈按钮。

### 4. 上传任务异步化与错误友好化

- 新增 `upload_jobs` 表模型。
- `POST /upload` 不再在请求线程同步执行 `_ingest_one()`，而是保存文件和 job 后返回 queued。
- 新增 `GET /upload/jobs` 与 `GET /upload/jobs/{job_id}`。
- 前端上传页会加载最近任务，并轮询 queued/running 状态直到 succeeded/failed。
- 上传非 PDF 和超大文件返回结构化错误：
  - `code`
  - `user_message`
  - `action_hint`
  - `retryable`
- 新增上传错误测试。

### 5. 最小产品导航与页面

- `ChatLayout` 增加产品导航。
- 新增：
  - 论文库：`PapersView.vue`
  - 上传管理：`UploadsView.vue`
  - 设置：`SettingsView.vue`
- 新增前端 API 封装：
  - `api/papers.ts`
  - `api/uploads.ts`
  - `api/feedback.ts`

### 6. Graph 路由测试化

- 将 reflection 后置路由抽为 `route_after_reflection()`。
- 新增测试覆盖：
  - passed -> final_answer
  - re_generate -> synthesis
  - re_retrieve -> re_planner
  - retry budget exhausted -> final_answer

### 7. LangGraph checkpoint

- 新增 `backend/app/agent/checkpoint.py`。
- 引入官方 `langgraph-checkpoint-sqlite`。
- 新增配置：
  - `AGENT_CHECKPOINT_ENABLED`
  - `AGENT_CHECKPOINT_PATH`
- sync `/chat` 与 streaming `/chat/stream` 统一使用 `conversation_id/session_id` 作为 LangGraph `thread_id`。
- 默认 checkpoint 文件：`data/langgraph_checkpoints.sqlite`。

### 8. Streaming 原生化

- `backend/app/routers/chat.py` 已移除 Queue、worker thread、ContextVar bridge。
- `/chat/stream` 改为 async StreamingResponse generator。
- 使用 `graph.astream_events(..., version="v2")` 驱动 SSE。
- `emit()` 改为 LangChain custom event，节点不再依赖 per-request queue。
- 现有前端 SSE 协议保持兼容。

### 9. LangGraph 依赖兼容性

- 去掉 `requirements.txt` 中重复的 `langgraph` 声明。
- 将 LangGraph 固定在兼容现有 LangChain 0.3 线的 `langgraph>=0.6.11,<0.7`。
- 将 SQLite checkpoint saver 固定为 `langgraph-checkpoint-sqlite>=2.0.11,<3.0`。
- 已用 `pip check` 验证当前依赖无冲突。

## 后续 P0

### Checkpoint 生产增强

问题：当前已完成单机 SQLite checkpoint，但多实例生产还应升级为 Postgres/MySQL 级别的共享 saver。

计划：
1. 评估 `langgraph-checkpoint-postgres` 或自定义 MySQL saver。
2. 增加 checkpoint 清理策略，避免 SQLite 文件无限增长。
3. 增加管理员级 checkpoint/run inspect 接口。

验收：
- 多实例共享同一 conversation/thread 状态。
- checkpoint 可按 conversation 清理。
- 可查看最近 failed/in-flight run。

## 后续 P1

### Redis 缓存

问题：Python 检索缓存仍是进程内 `TTLCache`。

计划：
1. 增加 `CacheStore` 抽象。
2. 实现 RedisStore 与 InMemoryStore。
3. 缓存 key 纳入 embedding model、collection、filter、hybrid 参数。

验收：
- 多进程共享 retrieval cache。
- Redis 不可用时可降级到 in-memory。

### Alembic 迁移

问题：当前 schema 仍依赖 `create_all()` 和手写迁移。

计划：
1. 初始化 Alembic。
2. 生成当前基线 migration。
3. 将 `chat_history` 手写迁移与 `answer_feedback` 建表纳入 migration。

验收：
- 空库可迁移到当前 schema。
- 已有库可无损升级。

## 后续 P2

- Prompt YAML/JSON 化，引入版本字段。
- LangFuse/LangSmith tracing 和 prompt 管理。
- 扩展 `fix_strategy`：`re_query_rewrite`、`re_specific_paper`、`decompose`、`clarify`。
- `failed_runs` 表与管理员失败会话查看。
- i18n key 与 ARIA 系统化治理。
- 真实用户体系、角色、trial/quota。
- LangChain 深度依赖瘦身。
- embedding 双 collection + alias 切换。
