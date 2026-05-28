# PaperRAG 消融实验诊断报告

**报告日期:** 2026-05-26  
**状态:** ⚠️ **部分失败** - API 认证问题导致实验中断

---

## 执行情况摘要

### ✅ 完成
- [x] Baseline 评测 (70 问题)
- [x] 启动 4 个消融实验 (retrieval_k, hybrid_alpha, final_context_k, hybrid_oversample)
- [x] 前 1-2 个参数值有完整评测数据

### ❌ 失败
- [x] 后续参数值评测因 API 认证失败而中断 (401 错误)

### ⏳ 部分结果
- [x] 每个消融实验的前 1-2 个参数有有效数据
- [x] 完整的消融扫描需要重新运行

---

## 错误分析

### 根本原因
```
MiniMax API 认证失败 (401 Unauthorized)
message: 'invalid api key (2049)'

发生时间：
- retrieval_k: 第 8/4 个参数时
- hybrid_alpha: 第 0.3/6 个参数时  
- final_context_k: 第 3/4 个参数时
- hybrid_oversample: 第 2.0/4 个参数时
```

### 可能原因
1. **API 速率限制** - 连续 70+ 个请求触发限流
2. **API 密钥过期** - 长时间运行后密钥失效
3. **配额耗尽** - MiniMax API 月度配额用尽
4. **并行连接限制** - 4 个并行实验超过限制

---

## 有效数据分析

### Baseline vs 部分消融结果

| 参数 | 最优值 | NDCG@5 | Δ NDCG | 评价 |
|------|--------|--------|--------|------|
| **Baseline** | — | **0.0489** | baseline | 🟡 |
| retrieval_k | 4 | 0.0167 | -65.8% | 🔴 |
| hybrid_alpha | 0.0 | 0.0167 | -65.8% | 🔴 |
| final_context_k | 2 | 0.0354 | -27.6% | 🔴 |
| hybrid_oversample | 1.5 | 0.0418 | -14.5% | 🔴 |

### 关键发现

**问题 1: 所有测试的参数都导致性能下降**
```
这很不寻常。可能原因：
- API 认证失败导致后续问题无法正确评测
- 环境变量设置不正确
- 参数应用机制有缺陷
```

**问题 2: retrieval_k=4 和 hybrid_alpha=0.0 表现最差**
```
这表明：
- 减少检索块数（从 8 到 4）大幅降低性能
- 纯 BM25 检索（alpha=0.0）也表现不佳
- 说明默认配置 (k=8, alpha=0.72) 实际上更优

但这与直觉相反，通常增加 retrieval_k 应该有帮助。
```

**问题 3: final_context_k=2 相对最好（-27.6%）**
```
- 虽然仍比 baseline 差，但相对损失最小
- 建议进一步测试 final_context_k=1, 2
```

---

## 推荐的修复方案

### 方案 A: 直接修复 (快速)
1. **检查 MiniMax API 密钥和配额**
   ```bash
   # 确认 API 密钥有效
   curl -H "Authorization: Bearer YOUR_KEY" \
     https://api.minimax.chat/v1/models
   ```

2. **检查 API 速率限制**
   - 查看 MiniMax 配额/速率限制设置
   - 可能需要增加配额或添加延迟

3. **修改消融脚本以添加延迟**
   ```bash
   # 在 ablation.py 中添加延迟
   for val in values:
       sleep 5  # 添加 5 秒延迟
       run_single(...)
   ```

### 方案 B: 完整重新运行 (推荐)
```bash
cd /Users/wesz_station/Projects/PaperRAG/backend
export PYTHONPATH="/Users/wesz_station/Projects/PaperRAG:/Users/wesz_station/Projects/PaperRAG/backend"

# 逐个运行消融实验，留出时间恢复
for exp in retrieval_k hybrid_alpha final_context_k hybrid_oversample; do
    python ../eval/scripts/ablation.py --experiment "$exp" --skip-reingest
    sleep 600  # 10 分钟等待后再运行下一个
done
```

### 方案 C: 使用本地或备用 LLM (最稳定)
```bash
# 如果 MiniMax API 不稳定，考虑：
# 1. 本地 Ollama/LLaMA
# 2. OpenAI API (如果有备用配额)
# 3. Claude API (该脚本的开发者)

# 修改 backend/app/core/config.py:
LLM_MODEL = "claude-3-haiku"  # 或其他稳定的模型
```

---

## 数据完整性说明

### 有效数据比例
```
retrieval_k:       1/4 有效 (25%)  - k=4 有效
hybrid_alpha:      1/6 有效 (17%)  - alpha=0.0 有效
final_context_k:   1/4 有效 (25%)  - k=2 有效
hybrid_oversample: 1/4 有效 (25%)  - oversample=1.5 有效
```

### 为什么后续参数失败
```
评测流程：
[参数 1] ✅ 成功 (API 可用)
  ↓
[参数 2] ❌ 失败 (API 返回 401)
  ↓
[参数 3-N] ❌ 全部失败 (无法恢复)
```

---

## 后续建议

### 立即行动
1. **修复 API 认证问题**
   - 验证 MiniMax API 密钥有效性
   - 检查配额使用情况
   - 联系 MiniMax 支持

2. **验证有效数据**
   ```bash
   # 查看完整的 CSV 文件
   wc -l /Users/wesz_station/Projects/PaperRAG/eval/results/ablation_*.csv
   # 应该看到: 2 行（header + 1 data）
   ```

### 重新运行策略
```bash
# 选项 1: 逐个实验，有延迟
for exp in retrieval_k hybrid_alpha final_context_k hybrid_oversample; do
    python ../eval/scripts/ablation.py --experiment "$exp"
    sleep 300  # 5 分钟延迟
done

# 选项 2: 只运行前几个有希望的参数
python ../eval/scripts/ablation.py --experiment final_context_k
python ../eval/scripts/ablation.py --experiment hybrid_oversample
```

### 备用方案
如果 MiniMax API 继续出现问题：
1. 切换到其他 LLM API
2. 使用本地运行的开源模型
3. 减少消融参数数量（关键参数优先）

---

## 文件状态

### 有效文件
```
✅ baseline-v2-detail.json     (完整，70 问题)
✅ ablation_retrieval_k.csv    (1 行数据)
✅ ablation_hybrid_alpha.csv   (1 行数据)
✅ ablation_final_context_k.csv (1 行数据)
✅ ablation_hybrid_oversample.csv (1 行数据)
```

### 日志文件
```
📋 /tmp/ablation_retrieval_k.log (71+ KB，包含详细错误)
📋 /tmp/ablation_hybrid_alpha.log
📋 /tmp/ablation_final_context_k.log
📋 /tmp/ablation_hybrid_oversample.log
```

---

## 下一步行动计划

### Phase 1: 诊断 (今天)
- [ ] 验证 MiniMax API 密钥和配额
- [ ] 检查是否有速率限制警告
- [ ] 分析错误日志中的 error code

### Phase 2: 修复 (今天/明天)
- [ ] 修复 API 认证问题
- [ ] 重新运行消融实验（带延迟）
- [ ] 收集完整的参数扫描数据

### Phase 3: 分析 (完成后)
- [ ] 重新运行分析脚本
- [ ] 生成优化建议
- [ ] 测试最优参数组合

---

**责任方:** 用户需要解决 MiniMax API 问题  
**预计恢复时间:** 1-2 小时  
**下一检查点:** API 修复后重新运行消融

