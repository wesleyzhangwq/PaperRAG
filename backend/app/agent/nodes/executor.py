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


def _parse_arxiv_to_documents(raw_result: str) -> list[Document]:
    """Parse arXiv tool output into Document objects for context."""
    docs = []
    for block in raw_result.split("---"):
        block = block.strip()
        if not block:
            continue
        # Extract paper_id from the text (format: "ID: xxxx.xxxxx")
        paper_id = ""
        title = ""
        lines = block.split("\n")
        content_lines = []
        for line in lines:
            if line.startswith("ID: "):
                paper_id = line[4:].strip()
            elif line.startswith("Title: "):
                title = line[7:].strip()
            else:
                content_lines.append(line)
        content = "\n".join(content_lines).strip() or block
        docs.append(Document(
            page_content=content,
            metadata={"paper_id": paper_id, "title": title, "source": "arxiv_api"},
        ))
    return docs


def _parse_web_to_documents(raw_result: str) -> list[Document]:
    """Parse web search tool output into Document objects for context."""
    docs = []
    for block in raw_result.split("---"):
        block = block.strip()
        if not block:
            continue
        docs.append(Document(
            page_content=block,
            metadata={"source": "web_search"},
        ))
    return docs


def executor_node(state: AgentState, *, db: Session) -> dict:
    """Execute the current plan step and advance the index."""
    idx = state["plan_step_index"]
    step = state["plan"][idx]
    action = step["action"]
    params = step["params"]

    t0 = time.perf_counter()
    new_context = list(state["retrieval_context"])
    output_summary = ""
    # Additional plan steps to append (e.g. from evaluate_docs)
    extra_plan_steps = []

    if action == "retrieve_local":
        docs_scores = _run_retrieve_local(params)
        new_context.extend([d for d, _ in docs_scores])
        output_summary = f"found {len(docs_scores)} chunks"

    elif action == "retrieve_arxiv":
        result = retrieve_arxiv_tool.invoke(params)
        arxiv_docs = _parse_arxiv_to_documents(result)
        new_context.extend(arxiv_docs)
        output_summary = f"arXiv: {len(arxiv_docs)} papers added to context"

    elif action == "search_web":
        result = search_web_tool.invoke(params)
        web_docs = _parse_web_to_documents(result)
        new_context.extend(web_docs)
        output_summary = f"web: {len(web_docs)} results added to context"

    elif action == "query_rewrite":
        intent = state["intent"] or {}
        queries = _run_query_rewrite(params, intent)
        output_summary = f"rewrote into {len(queries)} sub-queries: {queries}"
        # Inject rewritten queries into subsequent retrieve_local steps
        remaining_plan = list(state["plan"][idx + 1:])
        for plan_step in remaining_plan:
            if plan_step["action"] == "retrieve_local" and not plan_step["params"].get("query"):
                if queries:
                    plan_step["params"] = {**plan_step["params"], "query": queries.pop(0)}

    elif action == "evaluate_docs":
        query = params.get("query", "")
        if not query:
            for msg in reversed(state["messages"]):
                if hasattr(msg, "content") and msg.content:
                    query = msg.content
                    break
        eval_result = _run_evaluate_docs(params, query, new_context)
        sufficient = eval_result.get("sufficient", True)
        output_summary = f"sufficient={sufficient}"
        # If not sufficient, inject supplementary retrieval before synthesis
        if not sufficient:
            missing = eval_result.get("missing_aspects", [])
            output_summary += f", missing: {missing}"
            # Add extra retrieval steps for missing aspects
            for aspect in missing[:2]:  # limit to 2 extra retrievals
                extra_plan_steps.append({
                    "action": "retrieve_local",
                    "params": {"query": aspect, "top_k": 4},
                    "reason": f"supplement: {aspect}",
                })

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

    # Build updated plan if extra steps were injected
    updated_plan = state["plan"]
    if extra_plan_steps:
        updated_plan = list(state["plan"])
        # Insert extra steps right after current step
        for i, extra in enumerate(extra_plan_steps):
            updated_plan.insert(idx + 1 + i, extra)

    result = {
        "plan_step_index": idx + 1,
        "retrieval_context": new_context,
        "step_traces": state["step_traces"] + [trace],
    }
    if extra_plan_steps:
        result["plan"] = updated_plan

    return result
