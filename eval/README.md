# Eval 使用说明（仅汇总 CSV）

本目录用于记录每次优化前后的评测汇总结果，不保存明细文件。

## 文件说明

- `datasets/questions_v1.jsonl`
  - 评测问题集（每行一个 JSON）
  - 字段：
    - `qid`: 问题 ID
    - `query`: 问题文本
    - `expected_paper_ids`: 期望命中的论文 ID 列表（可空）
    - `expected_mode`: `answer` / `insufficient`
    - `tags`: 标签列表
- `results/summary.csv`
  - 每次评测一行汇总指标
- `run_eval.py`
  - 执行评测并向 `summary.csv` 追加一行

## 运行方式

在项目根目录执行：

```bash
cd backend
CHUNK_STRATEGY=v2 .venv/bin/python ../eval/run_eval.py --run-id "chunk-v2-baseline" --notes "baseline before tweak"
```

## 当前输出指标

- `answer_correctness`
- `tokens_per_request`
- `recall`
- `latency_p90`
- `hit_at_5`
- `mrr`
- `insufficient_ratio`

## 注意

- `hit_at_5` / `mrr` / `recall` 只对设置了 `expected_paper_ids` 的题目有意义。
- 如果 `expected_paper_ids` 为空，默认不计入检索分母（但仍计入 answer_correctness 与 insufficient_ratio）。
- `tokens_per_request` 为估算值（优先使用 `tiktoken`，无法编码时回退到字符近似）。
