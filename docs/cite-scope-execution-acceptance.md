# Cite Scope 执行计划验收文档

生成时间：2026-06-04

## 本轮验收范围

本轮以体检报告为输入，完成以下可直接落地的修复：

1. T-01：reflection JSON 解析失败不再静默通过。
2. T-07：evaluator 判定资料不足时，reflection 强制失败并触发重检索。
3. T-05：新增 chat/upload 固定窗口限流。
4. T-14：新增可选 API Key 鉴权中间件。
5. P-03：新增答案反馈表、接口与前端按钮。
6. P-04：上传错误结构化。
7. P-01/P-02/P-06：新增论文库、上传管理、设置视图和前端导航。
8. P-02：上传接口改为 queued job + 后台 ingest，并提供任务查询。
9. T-07：graph reflection 后置路由抽出并测试覆盖。
10. T-03：LangGraph 接入 SQLite checkpoint，并统一 conversation/thread id。
11. T-02：streaming 改为 `graph.astream_events()` 原生事件流，移除 Queue/thread/ContextVar bridge。
12. T-11：修正 LangGraph/checkpoint 依赖版本冲突，保留后续深度瘦身。

## 验收证据

### 已新增测试

- `backend/tests/agent/test_synthesis_reflection.py`
  - `test_reflection_parse_failure_triggers_retry`
  - `test_reflection_respects_insufficient_evaluator_result`
- `backend/tests/test_guardrail_middleware.py`
  - 限流超过阈值返回 429。
  - 非保护路径不受限流影响。
  - API Key 缺失时返回 401。
  - 有效 API Key 可通过。
- `backend/tests/test_feedback_router.py`
  - `POST /feedback` 返回 recorded。
  - `GET /feedback` 返回可消费的反馈信号列表。
- `backend/tests/test_upload_errors.py`
  - 非 PDF 上传返回结构化错误。
  - PDF 上传返回 queued job，且不在请求线程内调用 `_ingest_one()`。
- `backend/tests/agent/test_graph.py`
  - reflection passed / re_generate / re_retrieve / retry budget 路由。
- `backend/tests/agent/test_checkpointing.py`
  - `agent_run_config()` 使用 conversation id 作为 LangGraph `thread_id`。
  - SQLite checkpointer 可跨 saver 实例读取已持久化 checkpoint。
- `backend/tests/agent/test_streaming_events.py`
  - `emit()` 在非 LangGraph 运行上下文中为 no-op。
  - `emit()` 在 LangGraph `astream_events(version="v2")` 中转为 custom event。
  - LangGraph update event 可映射为现有 SSE 协议。
  - chat router 不再包含 Queue/thread/ContextVar bridge。
- `backend/tests/test_chat_router.py`
  - `/chat/stream` 使用 `astream_events(version="v2")`，并传入 conversation id 作为 `thread_id`。

### 已执行的红绿验证

- 新增 reflection 测试首次运行：2 failed, 4 passed。
- 修改 reflection 后重跑：6 passed。
- 新增 guardrail 测试首次运行：模块不存在导致 collection error。
- 实现 guardrail 后重跑：4 passed。
- 新增 feedback 测试首次运行：404。
- 实现 feedback 后重跑：1 passed。
- 新增 upload error 测试首次运行：字符串 detail 导致失败。
- 修改 upload error 后重跑：1 passed。
- 新增 async upload 测试首次运行：缺少 `_run_upload_ingest_job`，失败。
- 实现 upload job 后重跑：2 passed。
- 新增 graph route 测试首次运行：缺少 `route_after_reflection`，失败。
- 抽出 graph route 后重跑：6 passed。
- 新增 feedback list 测试首次运行：405。
- 实现 `GET /feedback` 后重跑：2 passed。
- 新增 checkpoint 测试首次运行：缺少 `app.agent.checkpoint`，collection 失败。
- 实现 checkpoint helper 后重跑：checkpoint 测试通过。
- 新增 streaming event 测试首次运行：缺少 `graph_event_to_sse_events`，失败。
- 改造 streaming helper 和 chat router 后重跑：6 passed。
- agent/chat 局部回归首次运行：checkpoint 序列化 MagicMock 失败。
- 测试环境关闭默认 checkpoint 后重跑：25 passed。
- checkpoint 初版依赖检查：`langchain-core 1.4.0` 与 LangChain 0.3 线冲突。
- 调整为 `langgraph>=0.6.11,<0.7` + `langgraph-checkpoint-sqlite>=2.0.11,<3.0` 后：`pip check` 通过。

## 功能验收清单

| 项目 | 验收状态 | 说明 |
| --- | --- | --- |
| reflection parse fallback | 通过 | 解析失败会触发 `re_retrieve`，并增加 `reflection_count` |
| evaluator -> reflection 决策回流 | 通过 | evaluator insufficient 时跳过 LLM reflection，直接失败 |
| API 限流 | 通过 | 进程内固定窗口，默认保护 `/chat`、`/upload` |
| 可选 API Key | 通过 | 默认关闭，生产通过环境变量启用 |
| 用户反馈接口 | 通过 | `answer_feedback` 表 + `POST /feedback` + `GET /feedback` |
| 前端反馈按钮 | 待浏览器验收 | 已接 API；需在真实浏览器中验证交互 |
| 论文库页面 | 待浏览器验收 | 已接 `/papers`；需真实后端数据验证 |
| 上传管理页面 | 通过后端测试，待浏览器验收 | 已接 `/upload` 和 `/upload/jobs/{job_id}`；后端不再同步 ingest |
| 上传结构化错误 | 通过 | 非 PDF 已有测试 |
| graph reflection 路由 | 通过 | `re_retrieve` 明确走 `re_planner`，避免 executor 执行 stale plan index |
| LangGraph checkpoint | 通过 | sync/stream 均接 checkpointer，使用 conversation id 作为 thread id；当前为单机 SQLite |
| streaming 原生化 | 通过 | `/chat/stream` 使用 `graph.astream_events()`；token 用 LangChain custom event；无 Queue/thread/ContextVar bridge |
| LangGraph 依赖兼容 | 通过 | requirements 已固定在兼容 LangChain 0.3 的 LangGraph 0.6 线，`pip check` 无冲突 |

## 未完成但已确认存在

这些项目不在本轮直接完成范围内，原因是改造面大，需要单独迁移与回归验证：

- Checkpoint 生产增强：SQLite 单机已完成；多实例共享 saver、checkpoint 清理和 inspect API 仍未完成。
- Redis 分布式缓存：仍未完成。
- Alembic migration：仍未完成。
- prompt 版本管理：仍未完成。
- DLQ/failed_runs：仍未完成。
- 完整用户、角色、配额、商业化体系：仍未完成。
- 完整 i18n/a11y：仍未完成。
- embedding 双 collection/alias 切换：仍未完成。
- LangChain 深度依赖瘦身：版本冲突已修正；包级精简仍未完成。

## 最终验收要求

完成本轮代码变更后已执行：

```bash
source backend/.venv/bin/activate && PYTHONPATH=backend pytest backend/tests -q
source backend/.venv/bin/activate && pip check
cd frontend && npm run build
cd java-backend && mvn test
```

结果：

- Python backend：49 passed，10 warnings。
- Python dependency check：No broken requirements found。
- Frontend：`vue-tsc --noEmit && vite build` 通过。
- Java backend：40 tests run，0 failures，0 errors，0 skipped，BUILD SUCCESS。

说明：验证后已清理 `backend` 源码区 `__pycache__`、`.pytest_cache`、`frontend/dist` 和 `java-backend/target` 构建产物。
