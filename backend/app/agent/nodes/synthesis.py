"""Reasoning synthesis node: generate cited answer from context."""
from __future__ import annotations

import time

from langchain_openai import ChatOpenAI

from app.agent.prompts.synthesis import SYNTHESIS_PROMPT, SYNTHESIS_WITH_ISSUES_PROMPT
from app.agent.stages import stage
from app.agent.state import AgentState, StepTrace
from app.agent.streaming import emit
from app.core.config import get_settings
from app.utils.content_safety import strip_hidden_reasoning


def _route_token(token: str, in_think: bool, reasoning_bucket: list[str]) -> tuple[str, bool]:
    """Split a streaming token into the answer (returned) and reasoning bucket
    (appended), handling <think>/</think> tag boundaries inline.
    Returns (answer_part, new_in_think_state)."""
    if not token:
        return "", in_think
    out: list[str] = []
    i = 0
    while i < len(token):
        if in_think:
            close = token.lower().find("</think>", i)
            if close == -1:
                # whole remainder is reasoning
                reasoning_bucket.append(token[i:])
                i = len(token)
            else:
                reasoning_bucket.append(token[i:close])
                i = close + len("</think>")
                in_think = False
        else:
            open_ = token.lower().find("<think>", i)
            if open_ == -1:
                out.append(token[i:])
                i = len(token)
            else:
                out.append(token[i:open_])
                i = open_ + len("<think>")
                in_think = True
    return "".join(out), in_think


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


def _synthesis_documents(state: AgentState) -> list:
    limit = max(1, int(get_settings().final_context_k))
    return list(state.get("retrieval_context") or [])[:limit]


def _format_context(documents: list) -> str:
    parts = []
    for d in documents:
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

    Streams tokens live into the SSE queue (if one is bound) via ``emit('token', ...)``
    so the frontend can render token-by-token without waiting for the node to finish.
    """
    t0 = time.perf_counter()
    attempt = int(state.get("reflection_count", 0) or 0) + 1
    with stage("synthesis", stage_id=f"synthesis:{attempt}" if attempt > 1 else "synthesis",
               title="生成回答" if attempt == 1 else "重新生成回答") as s:
        llm = _get_llm(streaming=True)
        context_docs = _synthesis_documents(state)
        context = _format_context(context_docs)

        if issues:
            prompt = SYNTHESIS_WITH_ISSUES_PROMPT.format(
                query=query, context=context, issues="\n".join(f"- {i}" for i in issues)
            )
        else:
            prompt = SYNTHESIS_PROMPT.format(query=query, context=context)

        chunks: list[str] = []
        reasoning_chunks: list[str] = []
        in_think = False  # state machine for inline <think>...</think> blocks
        emit("answer_start", {"attempt": attempt, "reset": True})
        for chunk in llm.stream(prompt):
            # Reasoning models (e.g. MiniMax-M2.7) may expose reasoning tokens
            # via additional_kwargs.reasoning_content (incremental, OpenAI-style).
            rk = getattr(chunk, "additional_kwargs", None) or {}
            rtok = rk.get("reasoning_content") if isinstance(rk, dict) else None
            if rtok:
                reasoning_chunks.append(rtok)
            if chunk.content:
                chunks.append(chunk.content)
                # Route inline <think>...</think> tokens to reasoning, not answer.
                stripped, in_think = _route_token(chunk.content, in_think, reasoning_chunks)
                if stripped:
                    emit("token", {"t": stripped})

        answer = strip_hidden_reasoning("".join(chunks))
        s.done(f"{len(answer)} 字符", detail={"chars": len(answer), "attempt": attempt})

    duration = round((time.perf_counter() - t0) * 1000, 2)
    trace = StepTrace(
        node="synthesis_node",
        action="reasoning_synthesis",
        input_summary=f"{len(context_docs)} chunks as context",
        output_summary=f"generated {len(answer)} chars" + (
            f" (+{sum(len(r) for r in reasoning_chunks)} reasoning chars)" if reasoning_chunks else ""
        ),
        duration_ms=duration,
    )
    return {
        "final_answer": answer,
        "synthesis_context_count": len(context_docs),
        "synthesis_context_paper_ids": [
            str((doc.metadata or {}).get("paper_id") or "").strip()
            for doc in context_docs
            if str((doc.metadata or {}).get("paper_id") or "").strip()
        ],
        "step_traces": state["step_traces"] + [trace],
    }
