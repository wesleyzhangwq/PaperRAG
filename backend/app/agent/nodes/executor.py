"""Executor node: dispatches plan steps to tools."""
from __future__ import annotations

import time
from hashlib import sha1

from langchain_core.documents import Document
from sqlalchemy.orm import Session

from app.agent.stages import ACTION_LABELS
from app.agent.state import AgentState, StepTrace
from app.agent.streaming import emit
from app.schemas.chat import ChatFilter
from app.services.retriever import retrieve
from app.tools.query_rewrite import rewrite_query
from app.tools.retrieve_arxiv import retrieve_arxiv_tool
from app.tools.search_web import search_web_tool
from app.tools.paper_detail import get_paper_detail
from app.tools.paper_chunks import get_paper_chunks


def _user_query_from_state(state: AgentState) -> str:
    """Return the latest user message content (or empty string)."""
    for msg in reversed(state.get("messages", [])):
        if hasattr(msg, "type") and getattr(msg, "type", "") == "human":
            return getattr(msg, "content", "") or ""
        # Fallback for tuple-like
        if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "user":
            return msg[1] or ""
    return ""


def _run_retrieve_local(
    params: dict, fallback_query: str
) -> tuple[list[tuple[Document, float]], dict]:
    """Run local retrieval, falling back to ``fallback_query`` when the planner
    failed to supply one. Returns (docs_scores, meta)."""
    requested_query = (params.get("query") or "").strip()
    used_fallback = False
    if not requested_query and fallback_query:
        query_used = fallback_query
        used_fallback = True
    else:
        query_used = requested_query

    top_k = params.get("top_k", 8)
    flt = None
    if params.get("category") or params.get("year_min") or params.get("year_max"):
        flt = ChatFilter(
            category=params.get("category") or None,
            year_min=params.get("year_min"),
            year_max=params.get("year_max"),
        )

    docs = retrieve(query_used, flt=flt, top_k=top_k) if query_used else []
    meta = {
        "query_used": query_used,
        "requested_query": requested_query,
        "used_fallback": used_fallback,
        "top_k": top_k,
    }
    return docs, meta


def _doc_key(doc: Document) -> str:
    md = doc.metadata or {}
    for key in ("chunk_id", "id"):
        if md.get(key):
            return f"{key}:{md[key]}"
    paper_id = md.get("paper_id") or ""
    page_num = md.get("page_num") or ""
    chunk_index = md.get("chunk_index") or ""
    if paper_id or page_num or chunk_index:
        return f"{paper_id}:{page_num}:{chunk_index}:{sha1((doc.page_content or '')[:200].encode()).hexdigest()}"
    return sha1((doc.page_content or "").encode()).hexdigest()


def _append_unique_documents(existing: list[Document], incoming: list[Document]) -> tuple[list[Document], int]:
    seen = {_doc_key(d) for d in existing}
    out = list(existing)
    added = 0
    for doc in incoming:
        key = _doc_key(doc)
        if key in seen:
            continue
        seen.add(key)
        out.append(doc)
        added += 1
    return out, added


def _run_query_rewrite(params: dict, intent: dict, fallback_query: str) -> list[str]:
    original = (params.get("original_query") or "").strip() or fallback_query
    return rewrite_query(original, intent)


def _parse_arxiv_to_documents(raw_result: str) -> list[Document]:
    """Parse arXiv tool output into Document objects for context."""
    docs = []
    for block in raw_result.split("---"):
        block = block.strip()
        if not block:
            continue
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
    if _web_result_unavailable(raw_result):
        return []
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


def _web_result_unavailable(raw_result: str) -> bool:
    text = (raw_result or "").strip().lower()
    return text.startswith((
        "web search unavailable",
        "web search is not configured",
        "no web results found",
    ))


def executor_node(state: AgentState, *, db: Session) -> dict:
    """Execute the current plan step and advance the index."""
    idx = state["plan_step_index"]
    step = state["plan"][idx]
    action = step["action"]
    params = dict(step.get("params") or {})
    fallback_query = _user_query_from_state(state)

    # Stable-id step event: the frontend upserts by id — no index guessing.
    emit("stage", {
        "id": f"step:{idx}",
        "stage": "retrieve_step",
        "status": "start",
        "title": ACTION_LABELS.get(action, action),
        "detail": {"action": action, "params": params, "reason": step.get("reason", "")},
    })

    t0 = time.perf_counter()
    new_context = list(state.get("retrieval_context", []))
    output_summary = ""
    output_detail: dict = {}
    state_patch: dict = {}
    trace_params = dict(params)
    trace_reason = step.get("reason", "")

    if action == "retrieve_local":
        docs_scores, meta = _run_retrieve_local(params, fallback_query)
        new_context, added_count = _append_unique_documents(new_context, [d for d, _ in docs_scores])
        output_summary = (
            f"found {len(docs_scores)} chunks"
            + (f", added {added_count} new" if added_count != len(docs_scores) else "")
            + (" (fallback)" if meta["used_fallback"] else "")
        )
        trace_params = {
            **trace_params,
            "query": meta["query_used"],
            "top_k": meta["top_k"],
            "requested_query": meta["requested_query"],
        }
        hits = []
        for d, score in docs_scores[:5]:
            md = d.metadata or {}
            hits.append({
                "paper_id": md.get("paper_id", ""),
                "title": (md.get("title") or "")[:120],
                "score": round(float(score), 4),
                "snippet": (d.page_content or "")[:200],
            })
        output_detail = {
            "hits": hits,
            "total": len(docs_scores),
            "query_used": meta["query_used"],
            "requested_query": meta["requested_query"],
            "used_fallback": meta["used_fallback"],
            "added": added_count,
        }
        if meta["used_fallback"]:
            state_patch["is_fallback"] = True

    elif action == "retrieve_arxiv":
        result = retrieve_arxiv_tool.invoke(params)
        arxiv_docs = _parse_arxiv_to_documents(result)
        new_context, added_count = _append_unique_documents(new_context, arxiv_docs)
        output_summary = f"arXiv: {len(arxiv_docs)} papers added"
        output_detail = {
            "papers": [
                {"paper_id": (d.metadata or {}).get("paper_id", ""),
                 "title": (d.metadata or {}).get("title", "")[:120]}
                for d in arxiv_docs[:5]
            ],
            "total": len(arxiv_docs),
            "added": added_count,
        }

    elif action == "search_web":
        result = search_web_tool.invoke(params)
        web_docs = _parse_web_to_documents(result)
        new_context, added_count = _append_unique_documents(new_context, web_docs)
        if _web_result_unavailable(result):
            output_summary = "web unavailable" if result.lower().startswith("web search unavailable") else "web: 0 results added"
        else:
            output_summary = f"web: {len(web_docs)} results added"
        output_detail = {
            "snippets": [(d.page_content or "")[:200] for d in web_docs[:3]],
            "total": len(web_docs),
            "added": added_count,
        }
        if _web_result_unavailable(result):
            output_detail["error"] = result

    elif action == "query_rewrite":
        intent = state.get("intent") or {}
        queries = _run_query_rewrite(params, intent, fallback_query)
        output_summary = f"rewrote into {len(queries)} sub-queries"
        output_detail = {"queries": queries}
        # Inject rewritten queries into subsequent retrieve_local steps
        remaining_plan = list(state["plan"][idx + 1:])
        for plan_step in remaining_plan:
            if plan_step["action"] == "retrieve_local" and not plan_step["params"].get("query"):
                if queries:
                    plan_step["params"] = {**plan_step["params"], "query": queries.pop(0)}

    elif action == "get_paper_detail":
        result = get_paper_detail(db, params.get("paper_id", ""))
        output_summary = "paper detail retrieved"
        output_detail = {"preview": (str(result) or "")[:300]}

    elif action == "get_paper_chunks":
        result = get_paper_chunks(db, params.get("paper_id", ""), params.get("max_chunks", 10))
        output_summary = "chunks retrieved"
        output_detail = {"preview": (str(result) or "")[:300]}

    else:
        output_summary = f"unknown action: {action}"

    duration = round((time.perf_counter() - t0) * 1000, 2)
    trace = StepTrace(
        node="executor_node",
        action=action,
        input_summary=f"{action}({', '.join(f'{k}={v}' for k, v in list(trace_params.items())[:3])})",
        output_summary=output_summary,
        duration_ms=duration,
        params=trace_params,
        reason=trace_reason,
        detail=output_detail,
    )
    emit("stage", {
        "id": f"step:{idx}",
        "stage": "retrieve_step",
        "status": "warning" if output_detail.get("error") or output_summary.startswith("unknown") else "done",
        "title": ACTION_LABELS.get(action, action),
        "summary": output_summary,
        "detail": {"action": action, "params": trace_params, "result": output_detail},
        "duration_ms": duration,
    })

    result: dict = {
        "plan_step_index": idx + 1,
        "retrieval_context": new_context,
        "step_traces": state["step_traces"] + [trace],
    }
    result.update(state_patch)
    return result
