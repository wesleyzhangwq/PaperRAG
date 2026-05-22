"""Reasoning synthesis node: generate cited answer from context."""
from __future__ import annotations

import time

from langchain_openai import ChatOpenAI

from app.agent.prompts.synthesis import SYNTHESIS_PROMPT, SYNTHESIS_WITH_ISSUES_PROMPT
from app.agent.state import AgentState, StepTrace
from app.core.config import get_settings


def _get_llm(*, streaming: bool = False) -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        model=s.llm_model,
        base_url=s.llm_api_base,
        api_key=s.llm_api_key,
        temperature=0.2,
        max_retries=2,
        request_timeout=120,
        streaming=streaming,
    )


def _format_context(state: AgentState) -> str:
    parts = []
    for d in state["retrieval_context"]:
        md = d.metadata or {}
        source_tag = md.get("source", "local")
        if source_tag == "arxiv_api":
            header = f"[arxiv_api | {md.get('paper_id', '?')} | {md.get('title', '')[:100]}]"
        elif source_tag == "web_search":
            header = "[web_search]"
        else:
            header = f"[arxiv:{md.get('paper_id', '?')} | {md.get('title', '')[:100]} | page={md.get('page_num', '?')}]"
        parts.append(f"{header}\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def synthesis_node(state: AgentState, *, query: str, issues: list[str] | None = None) -> dict:
    """Generate a cited answer from accumulated retrieval context.

    Uses streaming internally for token-by-token generation (consumed by SSE
    when called via graph.stream()). Falls back to invoke for sync calls.
    """
    t0 = time.perf_counter()
    llm = _get_llm(streaming=True)
    context = _format_context(state)

    if issues:
        prompt = SYNTHESIS_WITH_ISSUES_PROMPT.format(
            query=query, context=context, issues="\n".join(f"- {i}" for i in issues)
        )
    else:
        prompt = SYNTHESIS_PROMPT.format(query=query, context=context)

    # Use streaming to collect tokens (future: yield tokens via callback for SSE)
    chunks = []
    for chunk in llm.stream(prompt):
        if chunk.content:
            chunks.append(chunk.content)
    answer = "".join(chunks)

    duration = round((time.perf_counter() - t0) * 1000, 2)
    trace = StepTrace(
        node="synthesis_node",
        action="reasoning_synthesis",
        input_summary=f"{len(state['retrieval_context'])} chunks as context",
        output_summary=f"generated {len(answer)} chars",
        duration_ms=duration,
    )
    return {
        "final_answer": answer,
        "step_traces": state["step_traces"] + [trace],
    }
