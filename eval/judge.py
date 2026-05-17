# eval/judge.py
"""LLM-as-Judge: score RAG answers on faithfulness, relevance, correctness."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from langchain_openai import ChatOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.core.config import get_settings

JUDGE_PROMPT = """You are an impartial judge evaluating a RAG (Retrieval-Augmented Generation) system's answer quality.

Given:
- **User Query**: {query}
- **Retrieved Context**: {context}
- **System Answer**: {answer}

Score the answer on these three dimensions (1-5 each):

1. **Faithfulness**: Is the answer strictly based on the provided context? Does it avoid fabricating information not present in the context? (5 = perfectly faithful, 1 = mostly fabricated)

2. **Relevance**: Does the answer directly address the user's question? Is it on-topic and useful? (5 = perfectly relevant, 1 = completely off-topic)

3. **Correctness**: Given the context, is the answer factually correct? Are the claims supported by the retrieved chunks? (5 = all claims correct, 1 = mostly incorrect)

Respond with ONLY a JSON object, no other text:
{{"faithfulness": <int 1-5>, "relevance": <int 1-5>, "correctness": <int 1-5>, "reasoning": "<brief explanation>"}}"""


@dataclass
class JudgeScores:
    faithfulness: int
    relevance: int
    correctness: int
    reasoning: str


def _get_judge_llm() -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        model=s.judge_llm_model or s.llm_model,
        base_url=s.judge_llm_api_base or s.llm_api_base,
        api_key=s.judge_llm_api_key or s.llm_api_key,
        temperature=0.0,
        max_retries=2,
    )


def judge_answer(query: str, context: str, answer: str) -> JudgeScores:
    llm = _get_judge_llm()
    prompt = JUDGE_PROMPT.format(query=query, context=context, answer=answer)
    response = llm.invoke(prompt)
    text = response.content.strip()

    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]

    data = json.loads(text)
    return JudgeScores(
        faithfulness=max(1, min(5, int(data.get("faithfulness", 1)))),
        relevance=max(1, min(5, int(data.get("relevance", 1)))),
        correctness=max(1, min(5, int(data.get("correctness", 1)))),
        reasoning=str(data.get("reasoning", "")),
    )
