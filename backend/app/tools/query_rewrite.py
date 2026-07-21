"""Tool: LLM-powered query rewriting and decomposition."""
from __future__ import annotations

import json

from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.observability.llm_usage import invoke_with_usage

_REWRITE_PROMPT = """你是一个学术检索查询优化器。根据用户的原始问题和意图分析，生成 1-3 个改写后的检索查询。

规则：
1. 如果是对比类问题，拆分成针对每个对象的独立子查询。
2. 如果是简单问题，优化关键词使其更适合语义检索。
3. 使用英文关键词（学术论文多为英文）。
4. 输出严格 JSON 数组格式：["query1", "query2", ...]

原始问题：{query}
意图分析：{intent}

输出改写后的查询数组："""


def _get_llm() -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        model=s.planner_model or s.llm_model,
        base_url=s.llm_api_base,
        api_key=s.llm_api_key,
        temperature=0.3,
        max_retries=2,
        request_timeout=120,
    )


def rewrite_query(original_query: str, intent: dict) -> list[str]:
    """Rewrite and decompose a query into 1-3 optimized sub-queries for retrieval."""
    llm = _get_llm()
    prompt = _REWRITE_PROMPT.format(query=original_query, intent=json.dumps(intent, ensure_ascii=False))
    settings = get_settings()
    response = invoke_with_usage(
        llm,
        prompt,
        node="query_rewrite",
        model=settings.planner_model or settings.llm_model,
        api_base=settings.llm_api_base,
    )
    try:
        queries = json.loads(response.content)
        if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
            return queries[:3]
    except (json.JSONDecodeError, TypeError):
        pass
    return [original_query]
