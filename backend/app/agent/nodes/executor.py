"""Executor node: dispatches plan steps to tools."""
from __future__ import annotations

import time

from langchain_core.documents import Document
from sqlalchemy.orm import Session

from app.agent.state import AgentState, StepTrace
from app.schemas.chat import ChatFilter
from app.services.retriever import retrieve
from app.tools.query_rewrite import rewrite_query
from app.tools.evaluate_docs import evaluate_docs
from app.tools.retrieve_arxiv import retrieve_arxiv_tool
from app.tools.search_web import search_web_tool
from app.tools.paper_detail import get_paper_detail
from app.tools.paper_chunks import get_paper_chunks


def _run_retrieve_local(params: dict) -> list[tuple[Document, float]]:
    query = params.get("query", "")
    top_k = params.get("top_k", 8)
    flt = None
    if params.get("category") or params.get("year_min") or params.get("year_max"):
        flt = ChatFilter(
            category=params.get("category") or None,
            year_min=params.get("year_min"),
            year_max=params.get("year_max"),
        )
    return retrieve(query, flt=flt, top_k=top_k)


def _run_query_rewrite(params: dict, intent: dict) -> list[str]:
    return rewrite_query(params.get("original_query", ""), intent)


def _run_evaluate_docs(params: dict, query: str, context: list[Document]) -> dict:
    texts = [d.page_content for d in context]
    return evaluate_docs(query, texts)


def executor_node(state: AgentState, *, db: Session) -> dict:
    """Execute the current plan step and advance the index."""
    idx = state["plan_step_index"]
    step = state["plan"][idx]
    action = step["action"]
    params = step["params"]

    t0 = time.perf_counter()
    new_context = list(state["retrieval_context"])
    output_summary = ""

    if action == "retrieve_local":
        docs_scores = _run_retrieve_local(params)
        new_context.extend([d for d, _ in docs_scores])
        output_summary = f"found {len(docs_scores)} chunks"

    elif action == "retrieve_arxiv":
        result = retrieve_arxiv_tool.invoke(params)
        output_summary = f"arXiv results: {len(result.split('---'))} papers"

    elif action == "search_web":
        result = search_web_tool.invoke(params)
        output_summary = "web results received"

    elif action == "query_rewrite":
        intent = state["intent"] or {}
        queries = _run_query_rewrite(params, intent)
        output_summary = f"rewrote into {len(queries)} sub-queries"
        for i, plan_step in enumerate(state["plan"][idx + 1:], start=idx + 1):
            if plan_step["action"] == "retrieve_local" and not plan_step["params"].get("query"):
                if queries:
                    plan_step["params"]["query"] = queries.pop(0)

    elif action == "evaluate_docs":
        query = params.get("query", "")
        if not query:
            for msg in reversed(state["messages"]):
                if hasattr(msg, "content") and msg.content:
                    query = msg.content
                    break
        eval_result = _run_evaluate_docs(params, query, new_context)
        output_summary = f"sufficient={eval_result['sufficient']}"

    elif action == "get_paper_detail":
        result = get_paper_detail(db, params.get("paper_id", ""))
        output_summary = "paper detail retrieved"

    elif action == "get_paper_chunks":
        result = get_paper_chunks(db, params.get("paper_id", ""), params.get("max_chunks", 10))
        output_summary = "chunks retrieved"

    else:
        output_summary = f"unknown action: {action}"

    duration = round((time.perf_counter() - t0) * 1000, 2)
    trace = StepTrace(
        node="executor_node",
        action=action,
        input_summary=f"{action}({', '.join(f'{k}={v}' for k, v in list(params.items())[:3])})",
        output_summary=output_summary,
        duration_ms=duration,
    )

    return {
        "plan_step_index": idx + 1,
        "retrieval_context": new_context,
        "step_traces": state["step_traces"] + [trace],
    }
