"""Chat router: sync and streaming endpoints."""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import suppress
from typing import Optional

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy.orm import Session

from app.agent.checkpoint import agent_run_config, open_async_checkpointer
from app.agent.graph import build_agent_graph, initial_agent_state, run_agent_sync
from app.agent.state import AgentState
from app.agent.streaming import encode_sse, graph_event_to_sse_events
from app.core.config import get_settings
from app.db.mysql import SessionLocal, get_db
from app.models.chat_history import ChatHistory
from app.models.conversation import Conversation
from app.observability.llm_usage import collect_llm_usage
from app.schemas.chat import ChatRequest, ChatResponse
from app.utils.content_safety import strip_hidden_reasoning
from app.utils.time import utc_now

router = APIRouter(prefix="/chat", tags=["chat"])


def _load_history(db: Session, conversation_id: str, limit: int) -> list:
    """Load last `limit` messages for this conversation ONLY (no cross-conv,
    no long-term memory, no profile). Returns LangChain message objects."""
    if not conversation_id:
        return []
    rows = (
        db.query(ChatHistory)
        .filter(ChatHistory.conversation_id == conversation_id)
        .order_by(ChatHistory.id.desc())
        .limit(limit)
        .all()
    )
    rows.reverse()
    msgs = []
    for r in rows:
        if r.role == "user":
            msgs.append(HumanMessage(content=r.content))
        else:
            msgs.append(AIMessage(content=strip_hidden_reasoning(r.content)))
    return msgs


def _ensure_conversation(db: Session, conversation_id: str, first_user_msg: str) -> Conversation:
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).one_or_none()
    if conv is None:
        title = (first_user_msg or "新对话").strip().splitlines()[0][:60] or "新对话"
        conv = Conversation(
            id=conversation_id,
            title=title,
            pinned=False,
            created_at=utc_now(),
            updated_at=utc_now(),
        )
        db.add(conv)
        db.commit()
        db.refresh(conv)
    return conv


def _persist_messages(
    conversation_id: str,
    user_query: str,
    answer: str,
    sources: list,
    thinking_traces: list,
    presentation: Optional[dict] = None,
    elapsed_ms: Optional[float] = None,
    llm_usage: Optional[list[dict]] = None,
    fallback_telemetry: Optional[dict] = None,
) -> None:
    """Persist user message + assistant answer to chat_history.
    Uses a fresh session so it is independent from graph execution."""
    try:
        db: Session = SessionLocal()
        try:
            _ensure_conversation(db, conversation_id, user_query)
            now = utc_now()
            db.add(ChatHistory(
                conversation_id=conversation_id,
                session_id=conversation_id,
                role="user",
                content=user_query,
                created_at=now,
            ))
            # Pack thinking + presentation into thinking_json so a reload reproduces the UI.
            thinking_payload = {
                "traces": thinking_traces,
                "presentation": presentation,
                "elapsed_ms": elapsed_ms,
                "llm_usage": llm_usage or [],
                "fallback_telemetry": fallback_telemetry,
            }
            db.add(ChatHistory(
                conversation_id=conversation_id,
                session_id=conversation_id,
                role="assistant",
                content=strip_hidden_reasoning(answer),
                sources_json=json.dumps(sources, ensure_ascii=False) if sources else None,
                thinking_json=json.dumps(thinking_payload, ensure_ascii=False),
                created_at=now,
            ))
            conv = db.query(Conversation).filter(Conversation.id == conversation_id).one_or_none()
            if conv is not None:
                conv.updated_at = now
            db.commit()
        finally:
            db.close()
    except Exception:
        import traceback
        traceback.print_exc()


def _safe_usage_payload(records: list[dict]) -> dict:
    """Return usage-only fields suitable for persistence and SSE."""
    safe_calls = [
        {
            key: record.get(key)
            for key in (
                "call_index",
                "node",
                "provider",
                "model",
                "duration_ms",
                "outcome",
                "error_class",
                "usage_status",
                "usage_source",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "cached_read_tokens",
                "cache_write_tokens",
            )
        }
        for record in records
    ]
    all_known = all(call.get("usage_status") == "known" for call in safe_calls)
    return {
        "usage_status": "known" if all_known else "unknown",
        "call_count": len(safe_calls),
        "input_tokens": sum(int(call.get("input_tokens") or 0) for call in safe_calls) if all_known else None,
        "output_tokens": sum(int(call.get("output_tokens") or 0) for call in safe_calls) if all_known else None,
        "cached_read_tokens": sum(int(call.get("cached_read_tokens") or 0) for call in safe_calls) if all_known else None,
        "cache_write_tokens": sum(int(call.get("cache_write_tokens") or 0) for call in safe_calls) if all_known else None,
        "calls": safe_calls,
    }


@router.post("", response_model=ChatResponse)
def chat_sync(req: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    """Synchronous chat endpoint. Runs full agent pipeline, returns final result."""
    cid = req.conversation_id or req.session_id
    history = _load_history(db, cid, get_settings().chat_history_window)
    resp = run_agent_sync(db, req.query, session_id=cid, history=history)
    # Persist
    _ensure_conversation(db, cid, req.query)
    db.add(ChatHistory(conversation_id=cid, session_id=cid, role="user", content=req.query))
    db.add(ChatHistory(
        conversation_id=cid, session_id=cid, role="assistant", content=resp.answer,
        sources_json=json.dumps([s.model_dump() if hasattr(s, "model_dump") else s for s in resp.sources], ensure_ascii=False),
        thinking_json=json.dumps(
            {
                "presentation": resp.presentation,
                "llm_usage": resp.llm_usage,
                "fallback_telemetry": resp.fallback_telemetry,
            },
            ensure_ascii=False,
        ),
    ))
    db.commit()
    return resp


@router.post("/stream")
async def chat_stream(req: ChatRequest):
    """Streaming chat endpoint. Emits SSE events for each agent step
    AND for each LLM token in real time."""

    cid = req.conversation_id or req.session_id or str(uuid.uuid4())
    user_query = req.query
    settings = get_settings()

    async def event_generator():
        t_start = time.perf_counter()

        history_db: Session = SessionLocal()
        try:
            _ensure_conversation(history_db, cid, user_query)
            history = _load_history(history_db, cid, settings.chat_history_window)
        finally:
            history_db.close()

        yield encode_sse("conversation", {"conversation_id": cid})
        last_heartbeat = time.perf_counter()
        HEARTBEAT_INTERVAL = 0.5  # seconds
        final_state: AgentState = initial_agent_state(
            list(history) + [HumanMessage(content=user_query)]
        )
        runtime = {
            "prev_traces_len": 0,
            "steps_count": 0,
            "reflections_count": 0,
        }

        graph_db: Session = SessionLocal()
        try:
            with collect_llm_usage() as usage_collector:
                async with open_async_checkpointer() as checkpointer:
                    graph = build_agent_graph(graph_db, checkpointer=checkpointer)
                    graph_events = graph.astream_events(
                        final_state,
                        config=agent_run_config(cid),
                        version="v2",
                    )
                    next_event = asyncio.create_task(graph_events.__anext__())

                    try:
                        while True:
                            done, _ = await asyncio.wait({next_event}, timeout=HEARTBEAT_INTERVAL)
                            if not done:
                                elapsed_ms = round((time.perf_counter() - t_start) * 1000, 0)
                                yield encode_sse("elapsed", {"ms": elapsed_ms})
                                continue

                            try:
                                graph_event = next_event.result()
                            except StopAsyncIteration:
                                break

                            next_event = asyncio.create_task(graph_events.__anext__())
                            for sse_event in graph_event_to_sse_events(graph_event, final_state, runtime):
                                yield encode_sse(sse_event["type"], sse_event.get("data", {}))

                            now = time.perf_counter()
                            if now - last_heartbeat >= HEARTBEAT_INTERVAL:
                                elapsed_ms = round((now - t_start) * 1000, 0)
                                yield encode_sse("elapsed", {"ms": elapsed_ms})
                                last_heartbeat = now
                    finally:
                        if not next_event.done():
                            next_event.cancel()
                            with suppress(asyncio.CancelledError):
                                await next_event
                        await graph_events.aclose()
                final_state["llm_usage"] = usage_collector.snapshot()

            answer = final_state.get("final_answer", "") or ""
            sources_raw = final_state.get("sources", []) or []
            sources_data = [
                s.model_dump() if hasattr(s, "model_dump") else s for s in sources_raw
            ]
            if sources_data:
                yield encode_sse("sources", {"sources": sources_data})

            thinking_traces = final_state.get("step_traces", []) or []
            presentation = final_state.get("presentation") or None
            usage_payload = _safe_usage_payload(list(final_state.get("llm_usage") or []))
            fallback_telemetry = final_state.get("fallback_telemetry") or None
            if presentation:
                yield encode_sse("presentation", presentation)

            total_ms = round((time.perf_counter() - t_start) * 1000, 2)
            _persist_messages(
                cid, user_query, answer, sources_data, thinking_traces,
                presentation=presentation,
                elapsed_ms=total_ms,
                llm_usage=usage_payload["calls"],
                fallback_telemetry=fallback_telemetry,
            )
            yield encode_sse("done", {
                "steps_count": runtime["steps_count"],
                "reflections": runtime["reflections_count"],
                "llm_usage": usage_payload,
                "fallback_telemetry": fallback_telemetry,
            })
        except Exception as e:
            yield encode_sse("error", {"message": "Agent execution failed.", "type": type(e).__name__})
        finally:
            graph_db.close()
            total_ms = round((time.perf_counter() - t_start) * 1000, 2)
            yield encode_sse("elapsed", {"ms": total_ms, "final": True})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
