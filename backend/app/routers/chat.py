"""Chat router: sync and streaming endpoints.

Streaming uses a producer/consumer pattern so that synthesis_node tokens
reach the client live (without buffering until the node finishes):

    [event_generator] ───pulls─→ Queue ←───pushes─── [graph worker thread]
                                                       │
                                                       ├─ emit('token', ...)
                                                       ├─ emit('tool_call', ...)
                                                       └─ emit('elapsed', ...)
"""
from __future__ import annotations

import json
import queue as queue_module
import threading
import time
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy.orm import Session

from app.agent.graph import build_agent_graph, run_agent_sync
from app.agent.state import AgentState
from app.agent.streaming import stream_queue_ctx
from app.core.config import get_settings
from app.db.mysql import SessionLocal, get_db
from app.models.chat_history import ChatHistory
from app.models.conversation import Conversation
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])

_SENTINEL = {"type": "__done__"}


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
            msgs.append(AIMessage(content=r.content))
    return msgs


def _ensure_conversation(db: Session, conversation_id: str, first_user_msg: str) -> Conversation:
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).one_or_none()
    if conv is None:
        title = (first_user_msg or "新对话").strip().splitlines()[0][:60] or "新对话"
        conv = Conversation(
            id=conversation_id,
            title=title,
            pinned=False,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
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
) -> None:
    """Persist user message + assistant answer to chat_history.
    Uses a fresh session (we're in a background thread)."""
    try:
        db: Session = SessionLocal()
        try:
            _ensure_conversation(db, conversation_id, user_query)
            now = datetime.utcnow()
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
            }
            db.add(ChatHistory(
                conversation_id=conversation_id,
                session_id=conversation_id,
                role="assistant",
                content=answer,
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
    ))
    db.commit()
    return resp


@router.post("/stream")
def chat_stream(req: ChatRequest):
    """Streaming chat endpoint. Emits SSE events for each agent step
    AND for each LLM token in real time."""

    cid = req.conversation_id or req.session_id or str(uuid.uuid4())
    user_query = req.query
    q: "queue_module.Queue[dict]" = queue_module.Queue(maxsize=4096)
    settings = get_settings()

    # --- Worker: runs the agent graph in a thread, pushes events into queue ---
    def worker(history: list) -> None:
        token = stream_queue_ctx.set(q)
        worker_db: Session = SessionLocal()
        try:
            graph = build_agent_graph(worker_db)
            messages = list(history) + [HumanMessage(content=user_query)]
            initial_state: AgentState = {
                "messages": messages,
                "intent": None,
                "plan": [],
                "plan_step_index": 0,
                "retrieval_context": [],
                "step_traces": [],
                "reflection_count": 0,
                "final_answer": None,
                "reflection_result": None,
                "sources": None,
            }

            prev_traces_len = 0
            final_state = initial_state
            steps_count = 0
            reflections_count = 0

            for step_output in graph.stream(initial_state, config={"recursion_limit": 25}):
                for node_name, node_state in step_output.items():
                    final_state = {**final_state, **node_state}

                    # step_start before executor runs (based on plan)
                    if node_name == "executor":
                        plan = final_state.get("plan", [])
                        step_idx = node_state.get("plan_step_index", 0) - 1
                        if 0 <= step_idx < len(plan):
                            step_spec = plan[step_idx]
                            q.put({"type": "step_start", "data": {
                                "index": step_idx,
                                "action": step_spec.get("action", ""),
                                "reason": step_spec.get("reason", ""),
                            }})

                    # new step traces
                    traces = node_state.get("step_traces", [])
                    for trace in traces[prev_traces_len:]:
                        q.put({"type": "step_done", "data": trace})
                        steps_count += 1
                    prev_traces_len = len(final_state.get("step_traces", []))

                    if node_name == "intent" and node_state.get("intent"):
                        q.put({"type": "intent", "data": node_state["intent"]})

                    if node_name == "planner" and node_state.get("plan"):
                        q.put({"type": "plan", "data": {
                            "steps": node_state["plan"],
                            "total_steps": len(node_state["plan"]),
                        }})

                    if node_name == "reflection" and node_state.get("reflection_result"):
                        reflections_count += 1
                        q.put({"type": "reflection", "data": node_state["reflection_result"]})

                    if node_name == "re_planner" and node_state.get("plan"):
                        q.put({"type": "re_plan", "data": {
                            "new_steps": node_state["plan"][-3:],
                        }})

            # final answer (already streamed token-by-token from synthesis_node)
            answer = final_state.get("final_answer", "") or ""

            sources_raw = final_state.get("sources", []) or []
            sources_data = [
                s.model_dump() if hasattr(s, "model_dump") else s for s in sources_raw
            ]
            if sources_data:
                q.put({"type": "sources", "data": {"sources": sources_data}})

            thinking_traces = final_state.get("step_traces", []) or []

            # The presentation node built a productized payload for the UI.
            presentation = final_state.get("presentation") or None
            if presentation:
                q.put({"type": "presentation", "data": presentation})

            # Persist in worker so the connection can close immediately after 'done'
            _persist_messages(
                cid, user_query, answer, sources_data, thinking_traces,
                presentation=presentation,
            )

            q.put({"type": "done", "data": {
                "steps_count": steps_count,
                "reflections": reflections_count,
            }})

        except Exception as e:
            import traceback
            traceback.print_exc()
            q.put({"type": "error", "data": {
                "message": str(e),
                "type": type(e).__name__,
            }})
        finally:
            stream_queue_ctx.reset(token)
            worker_db.close()
            q.put(_SENTINEL)

    # --- Generator: drains queue, also emits periodic elapsed heartbeats ---
    def event_generator():
        t_start = time.perf_counter()

        # Load history in the request thread (DB session)
        history_db: Session = SessionLocal()
        try:
            _ensure_conversation(history_db, cid, user_query)
            history = _load_history(history_db, cid, settings.chat_history_window)
        finally:
            history_db.close()

        # Tell the client which conversation_id is in use (in case server generated it)
        yield f"event: conversation\ndata: {json.dumps({'conversation_id': cid}, ensure_ascii=False)}\n\n"

        t = threading.Thread(target=worker, args=(history,), daemon=True)
        t.start()

        last_heartbeat = time.perf_counter()
        HEARTBEAT_INTERVAL = 0.5  # seconds

        while True:
            try:
                evt = q.get(timeout=HEARTBEAT_INTERVAL)
            except queue_module.Empty:
                # heartbeat keeps the UI's elapsed counter accurate
                elapsed_ms = round((time.perf_counter() - t_start) * 1000, 0)
                yield f"event: elapsed\ndata: {json.dumps({'ms': elapsed_ms}, ensure_ascii=False)}\n\n"
                last_heartbeat = time.perf_counter()
                continue

            if evt is _SENTINEL or evt.get("type") == "__done__":
                total_ms = round((time.perf_counter() - t_start) * 1000, 2)
                yield f"event: elapsed\ndata: {json.dumps({'ms': total_ms, 'final': True}, ensure_ascii=False)}\n\n"
                break

            etype = evt["type"]
            edata = evt.get("data", {})
            yield f"event: {etype}\ndata: {json.dumps(edata, ensure_ascii=False, default=str)}\n\n"

            # Throttled heartbeat alongside real events
            now = time.perf_counter()
            if now - last_heartbeat >= HEARTBEAT_INTERVAL:
                elapsed_ms = round((now - t_start) * 1000, 0)
                yield f"event: elapsed\ndata: {json.dumps({'ms': elapsed_ms}, ensure_ascii=False)}\n\n"
                last_heartbeat = now

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
