INTENT_PROMPT = """分析用户的学术问题，提取意图信息。

用户问题：{query}

请输出 JSON：
{{
  "type": "simple" | "complex" | "comparison",
  "entities": ["提到的关键实体/论文/方法"],
  "complexity": "low" | "medium" | "high"
}}

判断标准：
- simple: 单一概念查询，一次检索即可回答
- comparison: 需要对比多个对象
- complex: 需要多步推理或综合多方面信息

输出 JSON："""
