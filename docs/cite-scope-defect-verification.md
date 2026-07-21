# Cite Scope 缺陷验证文档

生成时间：2026-06-04

依据：`/Users/wesz_station/Desktop/PaperRAG 项目体检报告.html` 与当前源码逐项对照。

## 结论总览

体检报告的 21 项中，确认存在 14 项，部分存在 5 项，报告不完全准确 2 项。本轮已经直接修复或补齐 10 项的关键闭环，其余属于架构级改造或上线前治理任务，已进入更新计划。

## 产品维度

| 编号 | 报告结论 | 源码核验 | 本轮处理 |
| --- | --- | --- | --- |
| P-01 只有一个视图 | 存在 | 原先 `frontend/src/views` 只有 `ChatView.vue`，`App.vue` 直接渲染 `ChatLayout` | 已补 `PapersView.vue`、`UploadsView.vue`、`SettingsView.vue`，并在 `ChatLayout.vue` 增加导航 |
| P-02 上传没法管 | 存在，已修复核心阻塞 | 原先 `backend/app/routers/upload.py` 在请求线程同步执行 `_ingest_one()`；前端原先没有上传入口 | 已改为 upload job + BackgroundTasks；新增 `/upload/jobs` 和 `/upload/jobs/{job_id}`；前端上传页会轮询 queued/running/succeeded/failed |
| P-03 无答案反馈 | 存在，已补读写闭环 | 原先无 `POST /feedback`，也无反馈表 | 已补 `answer_feedback` 模型、`POST /feedback`、`GET /feedback`、前端回答卡片反馈按钮和测试 |
| P-04 错误不友好 | 存在 | `upload.py` 使用字符串 `HTTPException` detail | 已将上传文件类型/大小错误改为结构化 detail，并补测试 |
| P-05 无 i18n/a11y | 部分存在 | `presentation.py` 与多个 Vue 文案仍为中文硬编码；现有按钮有部分 title，但未形成 i18n/ARIA 体系 | 本轮未做完整 i18n/a11y，设置页标注为后续计划 |
| P-06 无论文发现 | 部分存在 | 后端已有 `/papers` 列表和 arXiv/web 工具，但前端原先没有发现/浏览入口 | 已补论文库页面，支持关键词与类别筛选；主动 arXiv/web 搜索页仍未完成 |
| P-07 无用户体系 | 存在 | Python backend 无用户/角色模型，仅 `/ingest` 有 admin key | 本轮补可选 API Key 中间件；完整注册、登录、配额、付费墙未完成 |

## 技术维度

| 编号 | 报告结论 | 源码核验 | 本轮处理 |
| --- | --- | --- | --- |
| T-01 reflection 假通过 | 存在 | `backend/app/agent/nodes/reflection.py` 解析失败时 `passed=True` | 已改为 `passed=False`、`fix_strategy="re_retrieve"`，补回归测试 |
| T-02 手搓 streaming | 存在，已修复核心缺陷 | 原先 `backend/app/routers/chat.py` 使用 Queue、thread、ContextVar，未用 `astream_events` | 已改为 `graph.astream_events(..., version="v2")` + custom event adapter；删除 streaming queue bridge，保留原 SSE 协议 |
| T-03 state 不持久化 | 部分存在，已完成单机 checkpoint | 原先 `graph.compile()` 没有 checkpointer，`AgentState` 只在进程内 | 已接 `langgraph-checkpoint-sqlite`，sync/stream 都使用 `conversation_id` 作为 `thread_id`；后续可升级 Postgres/MySQL saver |
| T-04 缓存全内存 | 存在 | Python `retriever.py` 使用 `cachetools.TTLCache`；Java backend 也只是 Redis 边界和内存默认 | 本轮未接真实 Redis，列入后续 |
| T-05 无 rate limit | 存在 | 原先 `config.py` 无限流配置，`main.py` 无限流中间件 | 已补 `FixedWindowRateLimitMiddleware`，默认保护 `/chat`、`/upload`，补测试 |
| T-06 prompt 硬编码 | 存在 | `backend/app/agent/prompts/*.py` 为代码内 prompt，无 LangFuse/LangSmith | 本轮未迁移，列入后续 |
| T-07 evaluator 是装饰品 | 部分存在 | `executor.py` 会在 evaluator 不足时插补检索并写入 `state_patch["evaluator_result"]`，但 `reflection.py` 原先不读该字段 | 已让 reflection 将 evaluator 不足作为硬失败；`re_retrieve` 明确路由到 `re_planner` 生成补充检索计划；补 reflection 与 graph 路由测试 |
| T-08 embedding 模型硬编码 | 部分不准确 | `embedding_model` 是配置项，默认值为 `text-embedding-v4`；但无双 collection/alias 切换 | 本轮未做向量迁移体系，列入后续 |
| T-09 无 Alembic | 存在 | `mysql.py` 使用 `Base.metadata.create_all()` 和手写 `ALTER TABLE` | 本轮未引入 Alembic，列入后续 |
| T-10 Python 测试不足 | 存在 | 原先无 reflection parse fallback、evaluator->reflection、guardrail、feedback、upload error 测试 | 已补关键回归测试 |
| T-11 LangChain 包散 | 部分存在，已修正版本冲突 | `requirements.txt` 仍包含多种 LangChain 包；原先 `langgraph` 重复声明，且 checkpoint 初版引入过 `langchain-core 1.x` 与 LangChain 0.3 冲突 | 已去掉重复声明，并固定为 `langgraph>=0.6.11,<0.7` + `langgraph-checkpoint-sqlite>=2.0.11,<3.0`；深度依赖瘦身列入后续 |
| T-12 fix_strategy 简单 | 存在 | `graph.py` 只区分 `re_generate` 与默认 `re_planner` | 本轮未扩展策略，列入后续 |
| T-13 无 DLQ/重试 | 存在 | `chat.py` worker exception 只推 SSE error，不落失败运行表 | 本轮未加 failed_runs/DLQ，列入后续 |
| T-14 鉴权开口 | 存在 | 原先 chat/upload/papers/conversations 均无认证 | 已补可选 `APIKeyAuthMiddleware`；完整用户体系未完成 |

## 本轮新增或修改的主要文件

- `backend/app/agent/nodes/reflection.py`
- `backend/app/agent/checkpoint.py`
- `backend/app/agent/streaming.py`
- `backend/app/middleware/guardrails.py`
- `backend/app/models/feedback.py`
- `backend/app/models/upload_job.py`
- `backend/app/routers/feedback.py`
- `backend/app/routers/upload.py`
- `backend/tests/agent/test_graph.py`
- `backend/tests/agent/test_checkpointing.py`
- `backend/tests/agent/test_streaming_events.py`
- `backend/tests/agent/test_synthesis_reflection.py`
- `backend/tests/test_guardrail_middleware.py`
- `backend/tests/test_feedback_router.py`
- `backend/tests/test_upload_errors.py`
- `frontend/src/layouts/ChatLayout.vue`
- `frontend/src/views/PapersView.vue`
- `frontend/src/views/UploadsView.vue`
- `frontend/src/views/SettingsView.vue`
- `frontend/src/components/answer/AnswerCard.vue`
