# PaperRAG 评估框架

## 目录结构

```
eval/
├── datasets/
│   └── questions_v2.jsonl       # 评测问题集（65 题语义问题）
├── results/                     # 评测产出（CSV / JSON）
├── scripts/
│   ├── gen_questions.py         # LLM 生成评测问题
│   └── ablation.py              # 参数消融实验
├── run_eval.py                  # 主评测脚本（端到端）
├── run_retrieval_eval.py        # 纯检索评测（不调 LLM 生成）
├── judge.py                     # LLM-as-Judge 评分
└── metrics.py                   # NDCG@k, Precision@k, MRR 等
```

## 数据集

每条问题的格式（`questions_v2.jsonl`）：

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

问题类型分布：

| 类型 | 难度 | 数量 | 说明 |
|------|------|------|------|
| `concept_locate` | easy | 18 | 用语义描述定位论文 |
| `method_detail` | medium | 14 | 询问具体技术细节 |
| `fact_extract` | medium | 10 | 询问具体事实 |
| `comparison` | hard | 8 | 跨论文对比 |
| `trend_synthesis` | hard | 5 | 多论文趋势综合 |
| `negative` | — | 10 | 语料库不涵盖的话题（应拒答） |

## 运行

```bash
cd backend

# 基础评测
python ../eval/run_eval.py --run-id "baseline"

# 带 LLM-as-Judge
python ../eval/run_eval.py --run-id "with-judge" --judge

# 纯检索评测（不调 LLM，成本极低）
python ../eval/run_retrieval_eval.py --run-id baseline

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

**生成指标**（需 `--judge`）：faithfulness / relevance / correctness，各 1-5 分。

---

## 消融实验结果

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
| negative | 0.000 | 正确拒答，无误报 |

### 瓶颈分析

分数呈二值分布（大部分问题要么 1.0 要么 0.0），说明主要瓶颈是 **embedding 表征质量**，而非排序算法。Agent 的 `query_rewrite`（子查询分解）和 `evaluate_docs`（充分性检查 + 补充检索）正是针对这一瓶颈设计的。
