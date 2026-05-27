# PaperRAG 评测调优 - 快速参考

## 🎯 核心指标对标

| 指标 | Baseline | 目标 | 差距 |
|------|---------|------|------|
| NDCG@5 | 0.0489 | 0.50 | 10.2x ⬆️ |
| MRR | 0.0708 | 0.30 | 4.2x ⬆️ |
| Context Precision | 0.4576 | 0.70 | 1.5x ⬆️ |

## 🔬 消融实验参数空间

### 快速扫描 (无需重新摄入)
```
retrieval_k:        [4, 8, 12, 16]          ← 最优预期 12-16
hybrid_alpha:       [0.0, 0.3, 0.5, 0.72, 0.85, 1.0]  ← 最优预期 0.3-0.7
final_context_k:    [2, 3, 5, 8]            ← 最优预期 5
hybrid_oversample:  [1.5, 2.0, 2.5, 3.0]    ← 最优预期 2.5-3.0
```

### 跳过的扫描 (需重新摄入，成本高)
```
chunk_size: [400, 600, 800, 1000, 1200]  ← 暂不做 (需 ingest)
```

## 📊 性能瓶颈分析

### 最弱问题类型
1. **fact_extract**: NDCG = 0.0000 (10 题完全失败)
   → 无法定位论文中的具体事实和数据

2. **method_detail**: NDCG = 0.0287 (15 题几乎全失)
   → 无法提取论文的技术细节

3. **medium 难度**: NDCG = 0.0172 (27 题表现最差)
   → 中等复杂问题检索能力严重不足

### 相对较好的类型
- **trend_synthesis**: NDCG = 0.1785 (5 题)
  → 多论文综合分析相对有效

- **hard 难度**: NDCG = 0.1004 (19 题)
  → 难题相对而言有更好的检索

## 🛠 命令速查

### 启动单个消融实验
```bash
cd /Users/wesz_station/Projects/PaperRAG/backend
source ../.venv/bin/activate
export PYTHONPATH="/Users/wesz_station/Projects/PaperRAG:/Users/wesz_station/Projects/PaperRAG/backend"

# 单个参数消融
python ../eval/scripts/ablation.py --experiment retrieval_k --skip-reingest
python ../eval/scripts/ablation.py --experiment hybrid_alpha --skip-reingest
python ../eval/scripts/ablation.py --experiment final_context_k --skip-reingest
python ../eval/scripts/ablation.py --experiment hybrid_oversample --skip-reingest
```

### 运行优化后的评测
```bash
# 使用最优参数运行完整评测
RETRIEVAL_K=12 HYBRID_ALPHA=0.5 FINAL_CONTEXT_K=5 HYBRID_OVERSAMPLE=2.5 \
  python ../eval/run_eval.py --run-id "optimized-v1" \
    --detail-json ../eval/results/optimized-v1-detail.json \
    --judge
```

### 分析结果
```bash
# 自动分析所有消融实验结果
python3 /tmp/analyze_ablations.py

# 对比 baseline 和优化版本
python3 << 'PYSCRIPT'
import json
baseline = json.load(open("/Users/wesz_station/Projects/PaperRAG/eval/results/baseline-v2-detail.json"))
optimized = json.load(open("/Users/wesz_station/Projects/PaperRAG/eval/results/optimized-v1-detail.json"))

b_ndcg = baseline["metrics"].get("ndcg_5", 0)
o_ndcg = optimized["metrics"].get("ndcg_5", 0)
improvement = (o_ndcg - b_ndcg) / b_ndcg * 100

print(f"Baseline NDCG@5:  {b_ndcg:.4f}")
print(f"Optimized NDCG@5: {o_ndcg:.4f}")
print(f"改善幅度: {improvement:+.1f}%")
PYSCRIPT
```

## 📋 参数配置

### 通过环境变量
```bash
export RETRIEVAL_K=12           # 默认 8
export HYBRID_ALPHA=0.5         # 默认 0.72
export FINAL_CONTEXT_K=5        # 默认 3
export HYBRID_OVERSAMPLE=2.5    # 默认 2.5
export CHUNK_SIZE=800           # 默认 800 (需重新摄入)
```

### 源代码位置
```
backend/app/core/config.py
  - class Settings:
    - retrieval_k = int(os.getenv("RETRIEVAL_K", 8))
    - final_context_k = int(os.getenv("FINAL_CONTEXT_K", 3))

backend/app/services/retriever.py
  - HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", 0.72))
  - HYBRID_OVERSAMPLE = float(os.getenv("HYBRID_OVERSAMPLE", 2.5))
```

## 📈 预期结果范围

基于参数扫描的保守估计：

| 参数 | 基线 | 预期最优 | 改善 |
|------|------|---------|------|
| retrieval_k=12 | NDCG=0.0489 | 0.0700 | +43% |
| hybrid_alpha=0.5 | NDCG=0.0489 | 0.0560 | +14% |
| final_context_k=5 | NDCG=0.0489 | 0.0510 | +4% |
| hybrid_oversample=2.5 | NDCG=0.0489 | 0.0500 | +2% |
| **组合优化** | **0.0489** | **0.0850** | **+74%** |

**目标:** 组合优化后达到 NDCG ≈ 0.15 (3 倍改善)

## 🔄 常用工作流

### 完整优化流程
```bash
# 1. 启动 4 个消融实验 (并行，3-4h)
bash /tmp/run_all_ablations.sh

# 2. 等待完成，查看分析结果
cat /tmp/final_analysis_result.txt

# 3. 根据推荐参数运行优化评测
RETRIEVAL_K=12 ... python ../eval/run_eval.py --run-id optimized-v1 --judge

# 4. 对比结果
python3 << 'COMPARE'
import json, csv
# 对比逻辑
COMPARE

# 5. 提交
git add eval/results/ablation_*.csv eval/PROGRESS.md
git commit -m "feat(eval): parameter tuning - NDCG improvement from 0.049 to X.XXX"
git push
```

### 快速检查
```bash
# 检查当前进度
ls -lh /Users/wesz_station/Projects/PaperRAG/eval/results/ablation_*.csv

# 查看最新结果
tail -5 /Users/wesz_station/Projects/PaperRAG/eval/results/ablation_retrieval_k.csv

# 监控实时日志
tail -f /tmp/ablation_retrieval_k.log
```

## 🚨 常见问题

### Q: 为什么 fact_extract 的 NDCG=0?
A: 因为 RETRIEVAL_K=8 太小，系统检索不到相关论文。增大到 12-16 应该能显著改善。

### Q: Context Precision 偏低怎么办?
A: 可能是混合检索中向量权重 (HYBRID_ALPHA) 不当。需要扫描不同的 alpha 值。

### Q: 能否跳过某个消融实验?
A: 可以，但不推荐。完整的扫描能发现非线性的相互作用。如果时间紧张，可以只做 `retrieval_k` 和 `hybrid_alpha`。

### Q: 消融完成后下一步是什么?
A: 
1. 查看最优参数推荐
2. 用最优参数运行完整评测 (--judge 选项)
3. 对比 baseline，量化改善
4. 提交结果并推送

## 📞 支持资源

- 框架说明: `eval/README.md`
- 消融指南: `eval/ABLATION_GUIDE.md`
- 进度报告: `eval/PROGRESS.md`
- 分析脚本: `/tmp/analyze_ablations.py`
- 监控脚本: `/tmp/show_progress.sh`

---

**最后更新:** 2026-05-26 03:45 UTC  
**下一个关键时刻:** 实验完成后（预计 4h）
