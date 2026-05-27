# PaperRAG 消融实验运行指南

**开始时间:** 2026-05-26 03:30 UTC  
**预期完成:** 3-4 小时后

## 实验状态

### ✓ 已完成
- [x] Baseline 评测 (70 个问题)
  - NDCG@5: 0.0489
  - MRR: 0.0708
  - Context Precision: 0.4576
  - **诊断:** 检索质量严重不足，需优化

### ⏳ 进行中 (4 个并行消融实验)
- [ ] `retrieval_k`: 检索块数 [4, 8, 12, 16]
- [ ] `hybrid_alpha`: 向量/BM25 权重 [0.0, 0.3, 0.5, 0.72, 0.85, 1.0]
- [ ] `final_context_k`: LLM 上下文块数 [2, 3, 5, 8]
- [ ] `hybrid_oversample`: 过采样因子 [1.5, 2.0, 2.5, 3.0]

## Baseline 分析

### 关键问题
1. **整体 NDCG@5 只有 0.0489** — 距离目标 0.50 差 10 倍
2. **fact_extract 完全失败** — 10 个问题的 NDCG = 0.0
3. **medium 难度最弱** — NDCG = 0.0172 (vs easy=0.0500)
4. **Context Precision 仅 0.46** — 54% 的检索结果无相关引用

### 可能根因
```
检索参数不当
  ├─ RETRIEVAL_K=8 可能过小 → 漏掉相关论文
  ├─ HYBRID_ALPHA=0.72 向量权重可能不当
  ├─ FINAL_CONTEXT_K=3 可能过小
  └─ 向量检索质量问题（嵌入或向量库数据）
```

## 消融策略

### Phase 1: 快速优化 (无需重新摄入)

**优先级排序：**
1. `retrieval_k` → 最直接增加召回率
2. `hybrid_alpha` → 调整向量/BM25 融合
3. `final_context_k` → 优化 LLM 输入
4. `hybrid_oversample` → 微调检索候选

### 预期结果

| 参数 | 扫描范围 | 预期最优 | 预期改善 |
|------|---------|---------|---------|
| retrieval_k | 4~16 | 12~16 | +5~15% |
| hybrid_alpha | 0.0~1.0 | 0.3~0.7 | +2~8% |
| final_context_k | 2~8 | 5 | +1~3% |
| hybrid_oversample | 1.5~3.0 | 2.5 | +0.5~2% |

## 监控方式

### 实时进度
```bash
# 在另一个终端运行
bash /tmp/show_progress.sh
```

### 查看单个实验日志
```bash
tail -f /tmp/ablation_retrieval_k.log
tail -f /tmp/ablation_hybrid_alpha.log
```

### 等待完成并自动分析
```bash
# 已启动的后台监控
bash /tmp/wait_and_analyze.sh
```

## 结果目录

所有结果保存在：
```
eval/results/
├── summary.csv                    # 每次 run 汇总
├── ablation_retrieval_k.csv
├── ablation_hybrid_alpha.csv
├── ablation_final_context_k.csv
└── ablation_hybrid_oversample.csv
```

## 分析脚本

完成后自动运行：
```bash
python3 /tmp/analyze_ablations.py
```

输出：
- 各参数值的 NDCG@5、MRR、延迟对比表
- 最优参数推荐
- 改善幅度排序

## 下一步 (完成后)

### 1. 对比分析
```bash
python3 /tmp/analyze_ablations.py
```

### 2. 组合优化
```bash
cd backend
export PYTHONPATH="/Users/wesz_station/Projects/PaperRAG:/Users/wesz_station/Projects/PaperRAG/backend"

# 运行最优参数组合
RETRIEVAL_K=12 HYBRID_ALPHA=0.5 FINAL_CONTEXT_K=5 python ../eval/run_eval.py --run-id "optimized-v1"
```

### 3. 生成优化报告
- 对比 baseline vs 最优配置
- 按难度、类型的性能改善
- 推荐的部署参数

## 配置文件

参数配置源：
```
backend/app/core/config.py      # 默认参数
backend/app/services/retriever.py   # 检索逻辑
backend/app/services/synthesizer.py # 生成逻辑
```

修改环境变量即可在运行时覆盖：
```bash
export RETRIEVAL_K=12
export HYBRID_ALPHA=0.5
python run_eval.py ...
```

---

**进度追踪:** 用 `/tmp/analysis_progress.log` 查看后台分析日志  
**预计耗时:** 3-4 小时 (每个消折约 20-30 分钟)
