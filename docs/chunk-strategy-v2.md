# Chunk 策略 V2 改造说明

## 背景与问题

原有切分策略（V1）按“逐页独立切分”执行，优点是实现简单，但在论文场景会有几个明显问题：

- 跨页语义容易被硬切断，导致检索命中的 chunk 上下文不完整。
- 固定字符切分下，噪声文本（乱码、符号堆）容易被写入向量库。
- 缺少策略开关，难以快速回退对比。
- `References` 区域可能占用召回名额，干扰问答正文质量。

## 本次改造内容

本次改造以“低风险、可回滚”为原则，仅改动切分层与配置层：

1. 新增 `chunk_strategy` 配置（`v1`/`v2`），默认 `v2`。
2. 保留 `chunk_pages_v1()` 旧逻辑，作为回滚兜底。
3. 新增 `chunk_pages_v2()`，采用文档级切分：
   - 先清洗每页文本，再拼接整篇文本；
   - 在整篇文本上切分，避免跨页断裂；
   - 按字符偏移回溯每个 chunk 的 `page_num`。
4. 新增噪音过滤参数：
   - `CHUNK_MIN_CHARS`
   - `CHUNK_NOISE_SYMBOL_RATIO`
5. 新增 References 处理开关：
   - `CHUNK_DROP_REFERENCES=false`（默认不丢弃，便于稳妥上线）

## 改造原因

- **提升检索相关性**：减少噪声 chunk，提升有效 chunk 占比。
- **提升回答可读性**：chunk 更连贯，引用片段更像自然段。
- **提升可维护性**：策略可配置且可回退，方便线上调参与 A/B。
- **降低重构风险**：不改数据库 schema，不改 API，不改前端。

## 预期效果

在相同数据集下，V2 相比 V1 预计有以下改进：

- Top-K 命中中“正文相关片段”比例提升；
- “资料不足”误判率下降（尤其是跨页信息问题）；
- source snippet 的可读性提升（乱码和短噪音块减少）；
- 调参成本降低（只需改 `.env` 并重跑 ingest）。

## 参数建议（初始值）

建议先用以下默认值：

- `CHUNK_STRATEGY=v2`
- `CHUNK_SIZE=800`
- `CHUNK_OVERLAP=100`
- `CHUNK_MIN_CHARS=80`
- `CHUNK_NOISE_SYMBOL_RATIO=0.35`
- `CHUNK_DROP_REFERENCES=false`

如果你发现 references 干扰明显，可将 `CHUNK_DROP_REFERENCES` 改为 `true` 做对比测试。

## 回滚方案

若效果不理想，可直接在 `.env` 设置：

```bash
CHUNK_STRATEGY=v1
```

然后重新执行 ingest（或重建向量）即可回到旧策略，无需改代码。
