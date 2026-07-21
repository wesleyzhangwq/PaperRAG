# Cite Scope 评估框架

## 目录结构

```
eval/
├── datasets/
│   ├── mysql_papers_501_20260711.json # MySQL 501 篇 / 54,467 chunks 的证据快照
│   ├── questions_501_dev_50.jsonl     # 仅用于参数选择的开发集
│   ├── questions_501_test_200.jsonl   # paper-disjoint 冻结测试集
│   ├── questions_501_manifest.json    # 语料、切分、hash 与负例审计清单
│   ├── qdrant_papers_100_20260708.json # 从当前 Qdrant collection 导出的对齐语料
│   ├── questions_v2.jsonl       # 旧版评测问题集
│   └── questions_v3_200.jsonl   # 200 题纯 RAG 评测集
├── results/                     # 评测产出（CSV / JSON）
├── scripts/
│   ├── export_qdrant_metadata.py # 从 Qdrant chunk payload 聚合 paper metadata
│   ├── gen_questions.py         # LLM 生成评测问题
│   ├── repair_questions.py      # 修复 fallback 样式的低质量生成题
│   ├── repair_agentic_compare_run.py # 定点补跑中断的端到端对照行并重算汇总
│   ├── compare_rag_runs.py     # 成对 bootstrap 置信区间与 win/tie/loss
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

1. Traditional RAG：tuned hybrid（`alpha=.5`，dense fetch 80，保留 top-20）→ paper 去重后的固定 top-5 context → 单次生成；
2. Agentic RAG：复用同一 tuned hybrid，在本地 501 篇语料上执行 planner、工具路由、补检、反思和 citation gate；最终 context 同为 top-5，反思预算≤2。

评测从 LangGraph state 读取两条分支的**原始本地检索论文 ID**计算 Recall / Precision；最终综合上下文单独用于 citation support，因而不再混淆“检索到的来源”和“最终引用来源”。默认 `local-only` 会屏蔽 arXiv/web 工具返回，结果只归因于本地 RAG。

2026-07-11 当前结果目录：`eval/results/agentic/agentic-rag-501-v2-30-local-tuned/`。从冻结的 200 题 test 中按题型比例抽取 30 题（27 正例、3 负例），两条分支均 30/30 成功，且 Agent 动作审计中外部检索/Graph 动作为 0：

| 指标 | Traditional RAG | Agentic RAG | 解读 |
|------|-----------------|-------------|------|
| 回答/拒答模式正确率 | 73.33% | **93.33%** | +20.00pp，paired bootstrap 95% CI [+3.33,+36.67]pp |
| 原始检索 Recall | 80.74% | **87.04%** | +6.30pp，CI 下界为 0，不声明显著 |
| 原始检索 Precision | 14.72% | **24.79%** | +10.07pp，95% CI [+1.50,+19.22]pp |
| Citation support | 98.33% | **100.00%** | 引用均属于最终 synthesis context |
| 平均检索步骤 | **1.00** | 3.73 | 补检与查询改写增加执行成本 |
| P90 端到端延迟 | **15.91s** | 140.94s | 当前完整 Agent 路径只适合质量优先/复杂问题 |

`回答/拒答模式正确率` 是规则指标：正例需要给出实质回答，负例需要整体拒答；答案中的局部证据限制不等同于整题拒答。当前评测不使用外部 answer-quality judge，因此不把该指标表述为“答案正确率”。这是质量-成本边界的诊断；完整 Agent 路径应按问题复杂度路由，而不是所有请求默认执行。

```bash
GRAPH_RAG_ENABLED=false AGENT_EXTERNAL_RETRIEVAL_ENABLED=false \
HYBRID_ALPHA=.5 HYBRID_OVERSAMPLE=4 HYBRID_MAX_FETCH=96 \
RETRIEVAL_K=20 FINAL_CONTEXT_K=5 CACHE_RETRIEVAL_ENABLED=false \
backend/.venv/bin/python eval/run_agentic_rag_eval.py \
  --dataset eval/datasets/questions_501_test_200.jsonl \
  --run-id agentic-rag-501-v2-30-local-tuned \
  --sample-size 30 --retrieval-top-k 20 --context-k 5 \
  --traditional-context-strategy paper_dedup --resume

# 若运行期间仅有部分行因外部模型网络错误失败，保留原始行并定点补跑
backend/.venv/bin/python eval/scripts/repair_agentic_compare_run.py \
  eval/results/agentic/agentic-rag-501-v2-30-local-tuned \
  --max-attempts 3 --sleep-sec 8
```

每生成一个逐题结果，runner 都会持久化 JSONL；`--resume` 仅跳过已成功的题目，因此中断或模型网络错误后可以定点补跑。恢复前会逐项核对数据集 SHA-256、选中 qid 及顺序、抽样参数、Traditional/Agentic 配置、provider/model/embedding 设置、外部检索开关、billing origin、价格目录内容与版本，以及代码快照；任一不可变输入变化都会拒绝混跑，旧版缺少 `resume_contract` 的 manifest 必须新开 run。

### 生产化任务、延迟、成本与失败兜底指标

`run_agentic_rag_eval.py` 的新版结果同时记录严格端到端 `task_success`、端到端与分阶段延迟、结构化 fallback 遥测，以及覆盖全部内部 LLM 节点的 provider usage。`task_success` 的口径为：请求非终态失败且非降级；结构化回答/拒答模式符合 gold；所有最终引用都属于最终综合上下文；正例最终来源命中 gold；citation gate 后没有非法引用残留。安全降级不算完整任务成功。

延迟汇总包含 P50/P90/P95/mean 和样本数，并将 normal、fallback、terminal 三条路径分开；小样本 P95 仅用于描述该样本集，不应作为稳定生产 SLO。成本只接受 provider 返回的真实 usage，并要求 billing origin、provider、精确模型都能匹配版本化官方价格目录。任一内部调用 usage 缺失、官方目录单独计价的 cache read/write 维度未报告、或精确价格不可证实时，逐题 `cost_status=unknown`、`cost_usd=null`；缺失 cache 维度保留为 null，绝不假定为 0。评测不会用字符串 token 估算，也不会把 unknown 当成 0。当前成本范围为 `LLM-only`，不含 embedding/rerank。

```bash
# 确定性生产图故障注入：10 个场景、无网络、无外部 API 请求
PYTHONPATH=. backend/.venv/bin/python eval/run_failure_injection_eval.py \
  --run-id paperrag-failure-injection-v3-20260721

# 当真实 provider 的 billing origin 或可计费 usage 映射不可验证时，生成 n=0 的 blocked 报告
PYTHONPATH=. backend/.venv/bin/python eval/write_blocked_real_benchmark.py \
  --run-id paperrag-real-provider-blocked-v3-20260721

# 完整真实 provider 评测仅在 billing origin、精确价格与计费 usage 维度均已安全验证后运行
PYTHONPATH=. backend/.venv/bin/python eval/run_agentic_rag_eval.py \
  --dataset eval/datasets/questions_501_test_200.jsonl \
  --run-id <versioned-run-id> --sample-size <n> \
  --billing-origin minimax_paygo
```

每次生产化运行写入独立版本目录，并保留四个规范化产物：`manifest.json`、`per_question.jsonl`、`summary.json`、`report.md`。两个生产证据入口默认拒绝在脏工作树中生成报告；必须先提交 runner、配置和数据集，随后 manifest 才会记录 `dirty=false` 的可追溯 commit。Manifest 还记录数据集 hash、样本数、provider/model、非敏感配置、并发/预热/超时、价格目录版本、命令、时间与运行环境；不得写入 API key、完整 endpoint、prompt 或 provider 私有推理。

故障注入覆盖本地空结果/异常、arXiv 与 web 不可用、planner/sufficiency 不可解析、LLM timeout、groundedness 重新检索/重新生成、补检索预算耗尽，以及预期的终态数据库失败。其恢复率定义为 `fallback_recovery_rate = fallback_recovered / fallback_attempted`，其中 `fallback_recovered` 还必须满足上述严格 `task_success`。

2026-07-21 的可复现确定性结果位于 `eval/results/production/paperrag-failure-injection-v3-20260721/`：10/10 场景符合预期，其中成功恢复 7、降级但安全返回 2、预期终态失败 1；fallback recovery 为 7/9（0.7778），严格 task success 为 7/10（0.7）。该 fixture-only 小样本的 P50/P90/P95/mean 为 0.0104/0.0116/0.0137/0.0105 秒，仅用于验证遥测与恢复状态机，不代表真实 provider 延迟或生产 SLO。该报告绑定包含 runner 与冻结数据集的 commit `320ccb7`，且 manifest 记录 `dirty=false`。

同日真实 provider 运行记录在 `eval/results/production/paperrag-real-provider-blocked-v3-20260721/`：当前配置模型为 `MiniMax-M3`。MiniMax 当前页面公布了按 context 分层的 M3 input/output/cache-read 价格，但本地兼容端点的 billing origin 未验证，且返回 usage 维度无法安全映射到该计费合同；因此版本化目录对 M3 保持 fail-closed，不宣称精确可计费成本。为避免无法执行精确 USD 安全上限，运行在发送请求前被阻断。报告绑定同一 clean source commit 与 `questions_501_test_200.jsonl` 的 SHA-256，明确记录 `n=0`、外部请求 0，所有真实质量/延迟/成本指标为 unknown；不得将其替换为 fixture 数字。

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

当前权威结果已迁移到 501 篇 MySQL 事实语料：Qdrant 与 MySQL 精确对齐为 54,467 chunks，使用 50 题开发集选参，并在 paper-disjoint 的 200 题冻结测试集（180 正例、20 经审计负例）上比较可信 dense-only baseline 与 tuned hybrid。完整方法、消融、逐题型结果、bootstrap 区间和简历安全口径见：

- `eval/results/rag-501/REPORT.md`
- `eval/results/rag-501/test-dense-k20-dedup5/`
- `eval/results/rag-501/test-hybrid-a0.5-o4-k20-dedup5/`
- `eval/results/rag-501/test-comparison/`

冻结测试核心结果：Hit@5 0.8222 -> 0.8500，Recall@5 0.7370 -> 0.7531，Context chunk precision 0.2192 -> 0.2407（相对 +9.84%，成对 bootstrap 95% CI 为 [+0.0013, +0.0417]），P90 0.390s -> 0.418s。NDCG@5 仅 0.7037 -> 0.7063，不能表述为显著排序提升。

```bash
# dense-only frozen test
HYBRID_RETRIEVAL_ENABLED=false CACHE_RETRIEVAL_ENABLED=false \
  backend/.venv/bin/python eval/run_rag_eval.py \
  --dataset eval/datasets/questions_501_test_200.jsonl \
  --run-id test-dense-k20-dedup5 --output-dir eval/results/rag-501 \
  --retrieval-top-k 20 --context-k 5 --context-strategy paper_dedup

# development-selected hybrid frozen test
HYBRID_RETRIEVAL_ENABLED=true HYBRID_ALPHA=0.5 \
HYBRID_OVERSAMPLE=4 HYBRID_MAX_FETCH=96 CACHE_RETRIEVAL_ENABLED=false \
  backend/.venv/bin/python eval/run_rag_eval.py \
  --dataset eval/datasets/questions_501_test_200.jsonl \
  --run-id test-hybrid-a0.5-o4-k20-dedup5 --output-dir eval/results/rag-501 \
  --retrieval-top-k 20 --context-k 5 --context-strategy paper_dedup
```

## 历史 100 篇纯 RAG 评测结果

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
