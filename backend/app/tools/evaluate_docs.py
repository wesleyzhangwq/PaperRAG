"""Tool: LLM-powered document sufficiency evaluation."""
from __future__ import annotations

import json

from langchain_openai import ChatOpenAI

from app.core.config import get_settings

_EVAL_PROMPT = """你是一个学术检索质量评估器。判断当前检索到的资料是否足以回答用户问题。

用户问题：{query}

已检索到的资料摘要：
{context_summary}

请评估：
1. 这些资料是否足够回答用户的问题？
2. 如果不够，缺少哪些方面的信息？

输出严格 JSON 格式：
{{"sufficient": true/false, "reason": "评估理由", "missing_aspects": ["缺失方面1", ...]}}"""


def _get_llm() -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        model=s.planner_model or s.llm_model,
        base_url=s.llm_api_base,
        api_key=s.llm_api_key,
        temperature=0.1,
        max_retries=2,
    )


def evaluate_docs(query: str, context_texts: list[str]) -> dict:
    """Evaluate whether retrieved documents are sufficient to answer the query.
    Returns {sufficient: bool, reason: str, missing_aspects: list[str]}."""
    context_summary = "\n".join(f"- {t[:200]}" for t in context_texts[:10])
    llm = _get_llm()
    prompt = _EVAL_PROMPT.format(query=query, context_summary=context_summary)
    response = llm.invoke(prompt)
    try:
        result = json.loads(response.content)
        return {
            "sufficient": bool(result.get("sufficient", False)),
            "reason": str(result.get("reason", "")),
            "missing_aspects": list(result.get("missing_aspects", [])),
        }
    except (json.JSONDecodeError, TypeError):
        return {"sufficient": True, "reason": "Failed to parse evaluation", "missing_aspects": []}
