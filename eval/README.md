# Cite Scope 评估框架

## 目录结构

```
eval/
├── datasets/
│   ├── qdrant_papers_100_20260708.json # 从当前 Qdrant collection 导出的对齐语料
│   ├── questions_v2.jsonl       # 旧版评测问题集
│   └── questions_v3_200.jsonl   # 200 题纯 RAG 评测集
├── results/                     # 评测产出（CSV / JSON）
├── scripts/
│   ├── export_qdrant_metadata.py # 从 Qdrant chunk payload 聚合 paper metadata
│   ├── gen_questions.py         # LLM 生成评测问题
│   ├── repair_questions.py      # 修复 fallback 样式的低质量生成题
│   ├── repair_agentic_compare_run.py # 定点补跑中断的端到端对照行并重算汇总
│   └── ablation.py              # 参数消融实验（旧版检索评测）
├── run_eval.py                  # 主评测脚本（端到端）
├── run_retrieval_eval.py        # 纯检索评测（不调 LLM 生成）
├── run_rag_eval.py              # 纯 RAG 评测（检索 + top-k 上下文 + 可选固定上下文生成）
├── run_agentic_rag_eval.py      # 固定上下文 Traditional RAG vs 完整 Agentic RAG 对照
├── agentic_compare_metrics.py   # 两分支共享的来源/引用/延迟指标
├── judge.py                     # LLM-as-Judge 评分
├── metrics.py                   # 基础 NDCG@k, Precision@k 等
└── rag_metrics.py               # 纯 RAG 汇总、上下文质量、报告生成
```

## 数据集

每条问题的格式（`questions_v3_200.jsonl`）：

```jsonl
{
  "qid": "c001",
  "query": "哪篇论文提出了用状态空间模型做时间序列分类的框架？",
  "expected_paper_ids": ["2604.15174"],
  "expected_mode": "answer",
  "reference_answer": "论文 MambaSL 提出了...",
  "difficulty": "easy",
  "type": "concept_locate",
  "tags": ["single-paper", "semantic"]
}
```

`questions_v3_200.jsonl` 的问题类型分布：

| 类型 | 难度 | 数量 | 说明 |
|------|------|------|------|
| `concept_locate` | easy | 60 | 用语义描述定位论文 |
| `method_detail` | medium | 40 | 询问具体技术细节 |
| `fact_extract` | medium | 30 | 询问具体事实 |
| `comparison` | hard | 30 | 跨论文对比 |
| `trend_synthesis` | hard | 20 | 多论文趋势综合 |
| `negative` | mixed | 20 | 语料库不涵盖或近邻误召回话题 |

生成与校验口径：

- 先用 `export_qdrant_metadata.py` 从当前 Qdrant collection 导出实际可检索的 100 篇论文，避免 expected paper IDs 与向量库语料不一致。
- 再用 `gen_questions.py` 生成 200 题，并用 `repair_questions.py` 修复明显来自 fallback 模板的低质量题。
- 当前数据集校验：200 行、无重复 qid/query、无占位符、180 个 answer 正例、20 个 insufficient 负例，所有 expected paper IDs 均存在于 Qdrant 导出语料。

## 运行

```bash
# 端到端 Agent 评测
cd backend
python ../eval/run_eval.py --run-id "baseline"

# 带 LLM-as-Judge
python ../eval/run_eval.py --run-id "with-judge" --judge

# 旧版纯检索评测（不调 LLM，成本低）
python ../eval/run_retrieval_eval.py --run-id baseline
```

### 纯 RAG 评测

`run_rag_eval.py` 不经过 LangGraph Agent 的 planner / route / reflection / citation gate，只评估 RAG 本体：

1. 检索排序质量；
2. top-k 上下文命中率、上下文噪声率；
3. negative/out-of-corpus 问题的检索暴露情况；
4. 可选的固定上下文生成质量（`--generate`）。

```bash
# 在项目根目录运行
RETRIEVAL_K=8 HYBRID_ALPHA=0.72 \
  backend/.venv/bin/python eval/run_rag_eval.py \
  --dataset eval/datasets/questions_v3_200.jsonl \
  --run-id rag-v3-200-baseline-20260709 \
  --context-k 5 --k-values 1 3 5 10

RETRIEVAL_K=12 HYBRID_ALPHA=0.3 \
  backend/.venv/bin/python eval/run_rag_eval.py \
  --dataset eval/datasets/questions_v3_200.jsonl \
  --run-id rag-v3-200-optimized-20260709 \
  --context-k 5 --k-values 1 3 5 10 \
  --compare-summary eval/results/rag/rag-v3-200-baseline-20260709/summary.json

# 使用已有 retrieval detail JSON 离线重放，不访问 Qdrant
backend/.venv/bin/python eval/run_rag_eval.py \
  --from-detail-json eval/results/retrieval-optimized-detail.json \
  --run-id rag-replay \
  --context-k 5 --k-values 1 3 5 10

# 固定 top-k context 后生成答案，不经过 Agent 编排
backend/.venv/bin/python eval/run_rag_eval.py \
  --run-id rag-generation-smoke \
  --context-k 5 --k-values 1 3 5 10 \
  --limit 5 --generate

# 粗糙 baseline：不调用 embedding/Qdrant 检索，只在 paper title/abstract 上做 BM25
backend/.venv/bin/python eval/run_rag_eval.py \
  --dataset eval/datasets/questions_v3_200.jsonl \
  --run-id rag-v3-200-rough-bm25-paper-20260709 \
  --retriever lexical_paper \
  --lexical-corpus eval/datasets/qdrant_papers_100_20260708.json \
  --retrieval-top-k 5 --context-k 5 --k-values 1 3 5 10

# 常见上下文提效：按 paper 去重或 MMR 去冗余后再组装 top-k context
RETRIEVAL_K=8 HYBRID_ALPHA=0.72 \
  backend/.venv/bin/python eval/run_rag_eval.py \
  --dataset eval/datasets/questions_v3_200.jsonl \
  --run-id rag-v3-200-default-mmr-dedup \
  --context-strategy mmr_dedup --mmr-lambda 0.65 \
  --context-k 5 --k-values 1 3 5 10
```

### 当前 Traditional RAG vs Agentic RAG 对照（local-only）

`run_agentic_rag_eval.py` 用同一题集同时运行两条路径：

1. Traditional RAG：service retriever（`k=12, alpha=.72`）→ 固定 5 chunks → 单次生成；
2. Agentic RAG：`run_agent_eval_sync`，在同一本地语料上执行 planner、工具路由、补检、反思和 citation gate；最终综合 3 chunks，反思预算≤2。

评测从 LangGraph state 读取两条分支的**原始本地检索论文 ID**计算 Recall / Precision；最终综合上下文单独用于 citation support，因而不再混淆“检索到的来源”和“最终引用来源”。默认 `local-only` 会屏蔽 arXiv/web 工具返回，结果只归因于本地 RAG。

2026-07-10 当前结果目录：`eval/results/agentic/agentic-rag-v5-30-local-bge-m3-20260710/`。200 题集按题型比例抽取 30 题（27 正例、3 负例），两条分支均 30/30 成功：

| 指标 | Traditional RAG | Agentic RAG | 解读 |
|------|-----------------|-------------|------|
| 原始检索 Recall | 86.67% | **89.63%** | 多轮补检提高相关论文召回 |
| 原始检索 Precision | 45.87% | **53.55%** | 补检后原始候选更集中 |
| Citation support | 96.55% | **98.28%** | 最终回答的引用支持率小幅提高 |
| 最终综合 chunks | 5.00 | **3.00** | 生成上下文减少 40% |
| 平均检索步骤 | **1.00** | 7.27 | Agent 的执行成本更高 |
| P90 端到端延迟 | **20.06s** | 221.48s | 现有多轮策略不适合默认走完整链路 |

这是质量-成本边界的诊断，不将“Agent 更慢”回避为单一的正向结论；后续优化应优先增加早停和复杂问题路由，而不是默认执行多轮补检。

```bash
backend/.venv/bin/python eval/run_agentic_rag_eval.py \
  --dataset eval/datasets/questions_v3_200.jsonl \
  --run-id agentic-rag-v5-30-local-bge-m3-20260710 \
  --sample-size 30 --context-k 5 --resume

# 若运行期间仅有部分行因外部模型网络错误失败，保留原始行并定点补跑
backend/.venv/bin/python eval/scripts/repair_agentic_compare_run.py \
  eval/results/agentic/agentic-rag-v5-30-local-bge-m3-20260710 \
  --max-attempts 3 --sleep-sec 8
```

每生成一个逐题结果，runner 都会持久化 JSONL；`--resume` 仅跳过已成功的题目，因此中断或模型网络错误后可以定点补跑。

### 历史 Traditional RAG vs Agentic RAG 对照（不用于简历）

`agentic-rag-v3-30-proportional-20260709` 是在混合向量索引修复前运行的 30 题历史诊断，产物完整保留但不用于简历或当前质量结论。Traditional 分支统计固定 context 命中、Agent 分支统计最终 citation source，指标语义不等价；同时完整 Agent 可调用外部 arXiv/web，无法归因到本地 RAG。历史审计数据保存在对应目录的 `manifest.json`、逐题 JSONL/CSV、`summary.json` 和 `report.md`。

```bash
# 消融实验
python ../eval/scripts/ablation.py --experiment hybrid_alpha
python ../eval/scripts/ablation.py --all --skip-reingest
```

## 指标

**检索指标：**

| 指标 | 含义 |
|------|------|
| NDCG@5 | 排序质量（主指标） |
| MRR | 首个相关结果的排名倒数 |
| Precision@5 | top-5 中相关结果占比 |
| Recall@5 | 相关论文被召回的比例 |

**上下文指标：**

| 指标 | 含义 |
|------|------|
| Context chunk precision | top-k 上下文 chunk 中相关论文 chunk 的占比 |
| Context recall | 期望论文是否进入 top-k 上下文 |
| Context noise rate | top-k 上下文中的无关 chunk 比例 |
| Negative max score | 负例问题仍被检索出的最高相似度，用于观察误召回风险 |

**固定上下文生成指标**（需 `--generate`）：abstention / mode accuracy / citation support / citation precision。

---

## 当前纯 RAG 评测结果

同一 runner、同一批 200 题（180 正例、20 个 negative/out-of-corpus）、同一语料 payload。评测不经过 LangGraph Agent，只测纯 RAG 的检索排序与 top-k 上下文质量。

### 向量迁移修复与参数选择

历史集合 `paperrag` 在 `text-embedding-v4 -> BAAI/bge-m3` 迁移后混入两种向量空间：部分文档以旧模型入库、部分以新模型入库。两者维度相同但语义不可比较，导致新模型 query 对旧向量出现近零甚至负余弦分数；因此旧集合结果只作为缺陷基线保留，不再作为当前服务或简历的质量口径。

从原集合 payload 读取全部文本，以 bge-m3 在单独候选集合 `paperrag-bge-m3-20260710` 无覆盖重嵌入 9,704 chunks，数量与源集合一致。运行时通过 `paperrag-active` alias 指向候选集合，保留旧集合以便回滚。

| 运行 | 集合 / 参数 | Hit@5 | NDCG@5 | Recall@5 | MRR | Context chunk precision | P90 延迟 |
|------|-------------|-------|---------|----------|-----|-------------------------|----------|
| 历史混合索引（仅缺陷基线） | `paperrag`, `k=8, alpha=0.72` | 0.7278 | 0.6153 | 0.6176 | 0.6761 | 0.6000 | 0.4171s |
| 重嵌入候选默认 | `paperrag-bge-m3-20260710`, `k=8, alpha=0.72` | 0.9444 | 0.8270 | 0.8526 | 0.8691 | **0.7756** | 0.3555s |
| 当前运行参数 | `paperrag-active`, `k=12, alpha=0.72` | **0.9500** | **0.8345** | **0.8633** | **0.8745** | 0.7733 | **0.3473s** |

结论：重嵌入并切换 active alias 后，Hit@5 增加 22.22 个百分点，NDCG@5 为 0.6153 -> 0.8345，Recall@5 为 0.6176 -> 0.8633，P90 为 0.4171s -> 0.3473s。候选集合上将 `k` 从 8 增至 12 后，NDCG@5 再提升 0.0075、Recall@5 提升 0.0107，Context chunk precision 仅下降 0.0023，故选用 `k=12, alpha=0.72`。

当前可用于简历的稳健表述：

> 定位向量模型迁移导致的 Qdrant 混合向量空间，在不覆盖原索引的前提下重嵌入 9,704 chunks 并以 alias 切换；200 题纯 RAG 中 NDCG@5 0.615 -> 0.835、Recall@5 0.618 -> 0.863、P90 0.417s -> 0.347s。

注意：negative 问题在纯检索阶段仍会返回 top-k 近邻，不能单独证明“正确拒答”；拒答能力必须通过 `--generate` 或端到端 Agent 评测验证。

### Graph RAG 候选评测

`service_graph` 不实现独立检索逻辑：它严格调用生产 local `retrieve`、`retrieve_graph_context` 和 `evidence_node`。Neo4j 只扩展本地论文 ID，最终回答证据仍由 Qdrant 本地 chunk 二次回捞。

运行前会强制检查评测集的目标论文是否全部存在于 MySQL 图投影源；若 Qdrant 与 MySQL 不是同一语料，runner 会中止而不是把图降级误报为候选结果。

先启动 Neo4j 并同步成功入库的本地论文：

```bash
GRAPH_RAG_ENABLED=true docker compose up -d --build
docker compose exec backend python scripts/sync_graph.py --all
```

使用与当前 Pure RAG 相同的 200 题集进行候选对照：

```bash
GRAPH_RAG_ENABLED=true \
PYTHONPATH=.:backend backend/.venv/bin/python eval/run_rag_eval.py \
  --dataset eval/datasets/questions_v3_200.jsonl \
  --run-id rag-v3-200-service-graph \
  --retriever service_graph \
  --retrieval-top-k 12 --graph-expansion-top-k 12 --context-k 5 \
  --compare-summary eval/results/rag/rag-v3-200-bge-m3-k12-20260710/summary.json \
  --generate
```

候选报告会写入每题图扩展耗时、候选数、降级原因和五项门槛。仅当下列条件同时满足时，Graph RAG 才可合并并启用：

| 门槛 | 要求 |
|------|------|
| comparison Recall@5 | 相对传统 Pure RAG 至少 +0.05 |
| trend_synthesis Recall@5 | 相对传统 Pure RAG 至少 +0.05 |
| 整体 NDCG@5 | 不低于传统 Pure RAG -0.01 |
| fixed-context citation support | 1.00 |
| 图扩展 P95 | 不超过 800ms |

### 各题型表现

| 类型 | Baseline NDCG@5 | Recall-biased NDCG@5 | Baseline Recall@5 | Recall-biased Recall@5 | 观察 |
|------|------------------|------------------------|-------------------|-------------------------|------|
| concept_locate | 0.6170 | 0.6170 | 0.6333 | 0.6333 | 单论文语义定位稳定 |
| method_detail | 0.6250 | 0.6250 | 0.6250 | 0.6250 | 技术细节题对参数不敏感 |
| fact_extract | **0.8354** | 0.8210 | **0.8667** | 0.8333 | 事实题更依赖高精度排序 |
| comparison | **0.4975** | 0.4729 | 0.4833 | **0.5167** | 扩候选集能召回更多对比论文，但排序质量下降 |
| trend_synthesis | 0.4375 | **0.4829** | 0.3833 | **0.4467** | 多论文趋势题受益于更高召回 |

## 历史 64 题纯 RAG 结果

旧版 `questions_v2.jsonl` 过滤占位题后共 64 题，其中 54 题有期望论文、10 题为 negative/out-of-corpus。结果目录：

- `eval/results/rag/rag-baseline-online-20260708/`
- `eval/results/rag/rag-optimized-online-20260708/`

该批次中 `RETRIEVAL_K=12, HYBRID_ALPHA=0.3` 相比 baseline 有小幅全面提升：NDCG@5 0.5138 → 0.5169，Recall@5 0.5169 → 0.5215，MRR 0.5392 → 0.5434，Context chunk precision 0.4519 → 0.4667。由于样本量较小，仅作为历史参考。

## 历史检索消融结果

以下结果来自旧版 `run_retrieval_eval.py`，样本过滤和 detail 保存口径与 `run_rag_eval.py` 不完全一致，仅作为历史参考，不建议直接写进简历。

### Baseline → Optimized 总览

| | Baseline | Optimized | 变化 |
|---|----------|-----------|------|
| NDCG@5 | 0.333 | **0.350** | +5.2% |
| MRR | 0.377 | **0.385** | +2.0% |
| Recall@5 | 0.326 | **0.341** | +4.7% |
| Latency p90 | 1.05s | **0.61s** | -42% |

最优参数：`RETRIEVAL_K=12`, `HYBRID_ALPHA=0.3`（其余保持默认）。

### hybrid_alpha（向量 vs BM25 权重）

| alpha | NDCG@5 | 发现 |
|-------|--------|------|
| 0.0（纯 BM25） | 0.209 | 语义查询完全失效 |
| **0.3** | **0.337** | 最优：BM25 对 hard 类问题提升 10% |
| 0.72（默认） | 0.333 | — |
| 1.0（纯向量） | 0.333 | 与 0.72 无差异 |

### retrieval_k（检索数量）

| k | NDCG@5 | 发现 |
|---|--------|------|
| 4 | 0.326 | 不够 |
| 8 | 0.333 | 默认 |
| **12** | **0.345** | 最优：hard 类提升明显 |
| 20 | 0.348 | 收益递减，延迟翻倍 |

### 各题型表现（优化后）

| 类型 | NDCG@5 | 评价 |
|------|--------|------|
| method_detail | 0.429 | 最强，技术术语匹配好 |
| concept_locate | 0.389 | 语义匹配有效 |
| trend_synthesis | 0.475 | 优化后提升最大（+66%） |
| fact_extract | 0.263 | 较弱，需要精确匹配 |
| comparison | 0.153 | 最弱，单 query 难召回两篇论文 |
| negative | 0.000 | 旧版检索指标不计入负例质量 |

### 瓶颈分析

分数呈二值分布（大部分问题要么 1.0 要么 0.0），说明主要瓶颈是 **embedding 表征质量**，而非排序算法。Agent 的 `query_rewrite`（子查询分解）和 `evaluate_docs`（充分性检查 + 补充检索）正是针对这一瓶颈设计的。
