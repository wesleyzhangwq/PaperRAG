# PaperRAG 技术架构速览（新人版）

这份文档面向第一次接触本项目的同学，目标是让你在 10 分钟内搞清楚三件事：

1. 请求是怎么从用户问题走到答案的；
2. 缓存命中是怎么做、怎么看收益的；
3. 混合检索和 Chunk 策略为什么这样设计。

---

## 1. 一句话理解项目

PaperRAG 是一个“论文问答”系统：  
把 arXiv 论文解析并切分为 chunks，写入 Qdrant（向量）和 MySQL（结构化元数据），然后通过 FastAPI `/chat` 进行检索增强生成（RAG）并返回带引用的答案。

---

## 2. 总体架构（运行时）

```text
用户问题
  -> FastAPI /chat
  -> Retriever（Qdrant 向量召回 + 可选 BM25 混合重排 + 检索缓存）
  -> Generator（拼 prompt -> 调 MiniMax）
  -> 返回 answer + sources（paper_id/title/page/snippet/score）
```

核心组件：

- `backend/app/services/retriever.py`：检索主流程、混合重排、检索缓存
- `backend/app/db/qdrant.py`：Qdrant 访问、embedding 调用、query embedding 缓存、重试退避
- `backend/app/services/generator.py`：prompt 组装与 LLM 调用
- `backend/app/utils/chunker.py`：`v1/v2/v3` 三种切分策略
- `backend/app/core/config.py`：所有开关与参数

---

## 3. 数据模型与职责边界

### MySQL（结构化）

- 论文元数据（标题、作者、年份、分类、doi）
- chunk 元信息（paper_id、page_num、chunk_index 等）
- 作用：筛选、展示、引用补全（sources）

### Qdrant（向量）

- 每个 chunk 的向量 + 文本 + metadata payload
- 作用：语义召回

### 为什么双存储？

- 向量库负责“找相似内容”
- MySQL 负责“拿结构化事实和展示字段”
- 避免让向量 payload 承担全部职责，检索与展示解耦

---

## 4. 缓存体系（重点）

缓存不是“为了快而快”，而是为了控制两类稳定成本：

- 对外部 embedding API 的重复调用成本（金额 + 延迟 + 限流风险）
- 对向量检索链路的重复计算成本（Qdrant 查询 + 重排）

当前实现为两层缓存，分别在不同阶段拦截重复请求。

## 4.1 Query Embedding 缓存（LRU，发生在向量检索之前）

实现位置：`backend/app/db/qdrant.py` -> `AlibabaEmbeddingClient.embed_query()`

做了什么：

- 在 query 向量化前先查本地 `LRUCache`
- 命中则直接返回历史向量，不再请求 DashScope
- 未命中才走 `_embed_batch([text])`，拿到结果后回填缓存

怎么做的：

- 缓存键是原始 query 字符串
- 值是 embedding 向量 `list[float]`
- 通过 `Lock` 保证并发读写安全（避免并发请求下重复计算/脏写）
- 由 `get_embeddings()` 按配置决定是否开启、容量多大

为什么选 LRU：

- query 分布通常是“热点问题 + 长尾问题”
- LRU 对热点重复访问友好，内存可控，实现成本低

配置：

- `CACHE_EMBEDDING_ENABLED`
- `CACHE_EMBEDDING_MAX_ENTRIES`

---

## 4.2 检索结果缓存（TTL，发生在 Qdrant + 重排之后）

实现位置：`backend/app/services/retriever.py` -> `retrieve()`

做了什么：

- 对完整检索请求做结果级缓存，命中时直接返回 `list[(Document, score)]`
- 缓存范围覆盖：metadata filter、top_k、混合检索开关与参数

怎么做的：

- 用 `_cache_key()` 把 `query + where + k + hybrid 参数` 规范化后做 SHA256
- 缓存容器为 `TTLCache`，并配 `threading.Lock` 保护读写
- 未命中时才真正调用 `vs.similarity_search_with_score(...)` 并执行混合重排
- 命中时记录日志 `phase=cache_hit`

为什么选 TTL：

- 检索结果受数据更新影响，不适合长期常驻
- TTL 能在“命中率”和“结果新鲜度”之间给可控折中

配置：

- `CACHE_RETRIEVAL_ENABLED`
- `CACHE_RETRIEVAL_TTL_SEC`
- `CACHE_RETRIEVAL_MAX_ENTRIES`

---

## 4.3 线上如何判断缓存是否有效

看两组信号：

1. 日志信号：是否出现 `event=rag.retrieve` 且 `phase=cache_hit`
2. 性能信号：同 query 的 `retrieve_done.ms` 是否明显下降

如果命中率低，优先检查：

- query 是否有无意义差异（空格、标点、大小写）
- filter 条件是否经常变化
- TTL 是否设置过短

---

## 5. 混合检索（重点）

实现位置：`backend/app/services/retriever.py`（`retrieve()` + `_hybrid_fuse()`）

这个模块要解决的实际问题是：  
纯向量召回“语义对”，但有时关键词不强，导致 top1 不是你最想要的段落；纯词项匹配又容易漏掉同义表达。  
所以现在采用“向量先召回，BM25 再校正排序”的两阶段混合。

执行流程（按代码顺序）：

1. 计算 `fetch_limit`  
   - `max_fetch = max(k, min(HYBRID_MAX_FETCH, int(k * HYBRID_OVERSAMPLE)))`
   - 含义：先多取一些候选，给重排留空间
2. 调 Qdrant 向量检索  
   - `similarity_search_with_score(query, k, filter, fetch_limit=max_fetch)`
3. 计算 BM25 分  
   - 对候选 chunk 文本分词（中英文混合 token 规则）
   - 用 `BM25Okapi` 对 query 打分
4. 融合排序  
   - 向量分和 BM25 分分别做 min-max 归一化
   - 融合分：`alpha * vec + (1-alpha) * bm25`
   - 按融合分降序取 top-k

工程细节：

- 当候选太少或 token 为空时，会自动降级为原始向量排序，避免无意义重排
- `alpha` 在代码里被 clamp 到 `[0,1]`，避免配置异常导致结果不可控

关键参数（调参抓手）：

- `HYBRID_RETRIEVAL_ENABLED`：开关
- `HYBRID_OVERSAMPLE`：过采样倍率，越大重排空间越大，成本也越高
- `HYBRID_MAX_FETCH`：候选上限，防止大 `k` 时重排过慢
- `HYBRID_ALPHA`：语义权重，越大越偏向向量，越小越偏向关键词

实战建议：

- 问题偏“术语精确匹配”时，适当降低 `alpha`
- 问题偏“语义概括”时，适当提高 `alpha`
- 优先观察 top1/top3 质量，不只看 recall

---

## 6. Chunk 策略（重点）

实现位置：`backend/app/utils/chunker.py`  
设计说明：`docs/chunk-strategy-v2.md`

Chunk 策略决定了“检索的最小语义单元”，它直接影响三件事：

- 召回是否相关（太碎会丢语义，太大又会稀释主题）
- 引用是否可信（页码能不能回溯准确）
- 生成是否稳定（上下文是否连贯）

当前有三套策略：

- `v1`：逐页切分（历史策略，回滚兜底）
- `v2`：文档级切分（默认）
- `v3`：结构感知 + 动态切分（实验中）

### v2 到底做了什么

1. 每页先做基础清洗（去掉明显噪音、统一文本）
2. 把整篇论文拼成一个连续文本后再切分
3. 用字符偏移把每个 chunk 回映射到来源页码
4. 对 chunk 做质量过滤（长度阈值、符号噪声阈值）
5. 可选裁掉 References 区域，减少“参考文献噪声召回”

为什么这样做：

- 逐页切分会在页边界打断语义，文档级切分能保留段落连续性
- 页码回溯保证 `sources.page_num` 仍可用，不牺牲可追溯性
- 噪声过滤减少“乱码块/符号块”进入向量库，提升召回有效密度

关键参数：

- `CHUNK_STRATEGY`：`v1 | v2 | v3`
- `CHUNK_SIZE`、`CHUNK_OVERLAP`：控制上下文粒度
- `CHUNK_MIN_CHARS`：过滤过短块
- `CHUNK_NOISE_SYMBOL_RATIO`：过滤符号噪声块
- `CHUNK_DROP_REFERENCES`：是否丢弃 references 段

调参顺序建议：

1. 固定 `v2`，先看回答是否“有料且引用稳定”
2. 若出现无意义 snippet，先提高 `CHUNK_MIN_CHARS`
3. 若出现乱码/符号段，先降低 `CHUNK_NOISE_SYMBOL_RATIO` 阈值
4. 若 references 频繁抢召回，再开启 `CHUNK_DROP_REFERENCES=true`

---

## 7. 异常治理与可观测性（补充）

### 异常治理

位置：`backend/app/db/qdrant.py`

- embedding API 对 `429/5xx` 做指数退避重试：
  - `HTTP_RETRY_MAX_ATTEMPTS`
  - `HTTP_RETRY_BACKOFF_BASE_SEC`

位置：`backend/app/services/generator.py`

- LLM 调用失败返回友好文案，避免接口直接 500 裸异常。

### 可观测性

位置：

- `backend/app/core/observability.py`
- `backend/app/middleware/request_context.py`

能力：

- JSON 日志（含 `event/phase/ms/chunks/request_id`）
- 请求链路 `x-request-id` 透传（请求头进、响应头出）

---

## 8. 新人排查清单（建议按顺序）

1. 看 `/health` 返回是否为 `mysql + qdrant + minimax + dashscope`；
2. 连续发两次相同 `/chat`，观察 `cache_hit` 日志；
3. 调 `HYBRID_ALPHA`（如 `0.72 -> 0.55`）看排序变化；
4. 调 `CHUNK_*` 参数并重跑 ingest，对比 sources 可读性与相关性。

---

## 9. 关键配置总览

来自 `backend/app/core/config.py`（配在 `.env`）：

- 检索：
  - `RETRIEVAL_K`
  - `FINAL_CONTEXT_K`
- 混合检索：
  - `HYBRID_RETRIEVAL_ENABLED`
  - `HYBRID_OVERSAMPLE`
  - `HYBRID_ALPHA`
  - `HYBRID_MAX_FETCH`
- 缓存：
  - `CACHE_RETRIEVAL_ENABLED`
  - `CACHE_RETRIEVAL_TTL_SEC`
  - `CACHE_RETRIEVAL_MAX_ENTRIES`
  - `CACHE_EMBEDDING_ENABLED`
  - `CACHE_EMBEDDING_MAX_ENTRIES`
- Chunk：
  - `CHUNK_STRATEGY`
  - `CHUNK_SIZE`
  - `CHUNK_OVERLAP`
  - `CHUNK_MIN_CHARS`
  - `CHUNK_NOISE_SYMBOL_RATIO`
  - `CHUNK_DROP_REFERENCES`

---

如果你只想先看 3 个函数，请按这个顺序：

1. `retrieve()`（检索主链路）
2. `_hybrid_fuse()`（混合重排）
3. `embed_query()`（query embedding 缓存）
