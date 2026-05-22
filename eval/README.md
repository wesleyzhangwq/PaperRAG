# PaperRAG 评估框架

## 文件说明

```
eval/
├── datasets/
│   └── questions_v2.jsonl       # 评测问题集 (v2: 语义问题)
├── results/
│   ├── summary.csv              # 每次评测汇总 (一行一次 run)
│   └── ablation_*.csv           # 消融实验结果
├── scripts/
│   ├── gen_questions.py         # 用 LLM 生成评测问题
│   └── ablation.py              # 参数消融实验 runner
├── run_eval.py                  # 主评测脚本
├── judge.py                     # LLM-as-Judge 评分
└── metrics.py                   # NDCG@k, Precision@k, Context Precision
```

## 数据集 Schema (v2)

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

**问题类型：**
- `concept_locate` (easy) — 用语义描述定位论文
- `method_detail` (medium) — 询问具体技术细节
- `fact_extract` (medium) — 询问具体事实
- `comparison` (hard) — 跨论文对比
- `trend_synthesis` (hard) — 多论文趋势综合
- `negative` — 语料库不涵盖的话题

## 运行评测

```bash
cd backend

# 基础评测
python ../eval/run_eval.py --run-id "baseline-v2"

# 带 LLM-as-Judge 评分
python ../eval/run_eval.py --run-id "baseline-judge" --judge

# 保存逐题明细
python ../eval/run_eval.py --run-id "baseline-detail" --detail-json ../eval/results/detail.json
```

## 消融实验

```bash
cd backend

# 单个实验
python ../eval/scripts/ablation.py --experiment hybrid_alpha
python ../eval/scripts/ablation.py --experiment retrieval_k
python ../eval/scripts/ablation.py --experiment final_context_k

# 全部实验 (跳过需要 re-ingest 的)
python ../eval/scripts/ablation.py --all --skip-reingest
```

**可用实验：**
| 实验 | 环境变量 | 扫描值 |
|------|---------|--------|
| `hybrid_alpha` | `HYBRID_ALPHA` | 0.0, 0.3, 0.5, 0.72, 0.85, 1.0 |
| `retrieval_k` | `RETRIEVAL_K` | 4, 8, 12, 16 |
| `chunk_size` | `CHUNK_SIZE` | 400, 600, 800, 1000, 1200 |
| `final_context_k` | `FINAL_CONTEXT_K` | 2, 3, 5, 8 |
| `hybrid_oversample` | `HYBRID_OVERSAMPLE` | 1.5, 2.0, 2.5, 3.0 |

## 生成新问题集

```bash
cd backend

# 默认参数 (20 concept + 15 method + 10 fact + 10 compare + 5 trend + 10 negative)
python ../eval/scripts/gen_questions.py

# 自定义
python ../eval/scripts/gen_questions.py --concept-count 30 --method-count 20 --seed 123
```

## 指标说明

**检索指标：**
- `ndcg_5` — Normalized DCG@5
- `precision_5` — Precision@5
- `recall_5` — Recall@5
- `mrr` — Mean Reciprocal Rank
- `ctx_precision` — 引用命中率 (cited ∩ retrieved / retrieved)

**生成指标 (需要 --judge)：**
- `faithfulness_avg` — 答案是否忠实于检索上下文 (1-5)
- `relevance_avg` — 答案是否切题 (1-5)
- `correctness_avg` — 答案事实正确性 (1-5)

**效率指标：**
- `latency_p90` — P90 延迟
- `tokens_per_request` — 平均 token 消耗

**分层报告：**
- 按难度 (easy/medium/hard) 分别报告 NDCG@5 和 MRR
- 按问题类型分别报告
