"""Tool: LLM-powered document sufficiency evaluation.

Returns a structured result; on parse failure we mark `parse_failed=True` and
DO NOT silently default to `sufficient=True` — letting the caller downgrade
confidence instead of assuming success.
"""
from __future__ import annotations

import json
import re

from langchain_openai import ChatOpenAI

from app.core.config import get_settings

_EVAL_PROMPT = """你是一个学术检索质量评估器。判断当前检索到的资料是否足以回答用户问题。

用户问题：{query}

已检索到的资料摘要：
{context_summary}

请评估：
1. 这些资料是否足够回答用户的问题？
2. 如果不够，缺少哪些方面的信息？

只输出严格 JSON，不要任何额外文字：
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


def _extract_json(text: str) -> dict | None:
    """Best-effort JSON extraction from LLM output (tolerates code fences,
    leading prose, etc.)."""
    if not text:
        return None
    text = text.strip()
    # Strip ``` fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()
    # Try direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    # Try to find the first {...} block
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except (json.JSONDecodeError, TypeError):
            return None
    return None


def evaluate_docs(query: str, context_texts: list[str]) -> dict:
    """Evaluate whether retrieved documents are sufficient to answer the query.

    Returns ``{sufficient, reason, missing_aspects, parse_failed, raw}``.
    """
    if not context_texts:
        return {
            "sufficient": False,
            "reason": "no_context",
            "missing_aspects": [],
            "parse_failed": False,
            "raw": "",
        }

    context_summary = "\n".join(f"- {t[:200]}" for t in context_texts[:10])
    llm = _get_llm()
    prompt = _EVAL_PROMPT.format(query=query, context_summary=context_summary)

    try:
        response = llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        return {
            "sufficient": False,
            "reason": f"llm_error: {type(e).__name__}",
            "missing_aspects": [],
            "parse_failed": True,
            "raw": str(e),
        }

    parsed = _extract_json(raw)
    if parsed is None:
        # Parse failed — DO NOT silently set sufficient=True.
        # Caller will downgrade confidence based on parse_failed=True.
        return {
            "sufficient": False,
            "reason": "evaluator_parse_failed",
            "missing_aspects": [],
            "parse_failed": True,
            "raw": raw[:500] if isinstance(raw, str) else str(raw)[:500],
        }

    return {
        "sufficient": bool(parsed.get("sufficient", False)),
        "reason": str(parsed.get("reason", "")),
        "missing_aspects": list(parsed.get("missing_aspects", [])),
        "parse_failed": False,
        "raw": raw[:500] if isinstance(raw, str) else "",
    }
