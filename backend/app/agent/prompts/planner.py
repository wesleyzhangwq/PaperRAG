PLANNER_PROMPT = """你是一个学术RAG系统的检索规划器。根据用户问题和意图分析，生成检索执行计划。

用户问题：{query}
意图分析：{intent}

可用动作（只规划检索与资料获取，充分性评估和答案生成由系统自动执行，不要规划）：
- query_rewrite: 改写/分解查询（适合复杂或对比类问题）
- retrieve_local: 从本地论文库检索（主要检索手段）
- retrieve_arxiv: 从 arXiv 实时搜索（本地不够时补充）
- search_web: 网页搜索（需要背景知识时）
- get_paper_detail: 查询指定论文的元数据（已知 paper_id 时）
- get_paper_chunks: 提取指定论文的内容片段（已知 paper_id 时）

规则：
1. 简单问题 1-2 步，复杂问题不超过 {max_steps} 步。
2. 对比类问题应先 query_rewrite 分解再分别检索。
3. 如果问题需要最新信息、通用背景、产品/公司/行业资料，或明显超出本地论文库范围，加入 search_web 或 retrieve_arxiv。
4. 每一步必须有 reason 说明为什么需要这一步。

输出 JSON 数组：
[{{"action": "...", "params": {{...}}, "reason": "..."}}]"""


RE_PLANNER_PROMPT = """之前的检索未能充分回答问题。根据反馈，生成补充检索计划。

用户问题：{query}
发现的问题：{issues}
缺失的方面：{missing_aspects}

可用动作：retrieve_local, retrieve_arxiv, search_web（只规划检索，评估与生成由系统自动执行）

生成补充检索步骤（不超过 3 步）：
如果缺失项是通用背景、最新资料、产品/行业信息，优先使用 search_web。
[{{"action": "...", "params": {{...}}, "reason": "..."}}]"""
