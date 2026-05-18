"""Chat router: sync and streaming endpoints."""
from __future__ import annotations

import json
import time
import traceback

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.agent.graph import build_agent_graph, run_agent_sync
from app.agent.state import AgentState
from app.db.mysql import get_db
from app.schemas.chat import ChatRequest, ChatResponse

from langchain_core.messages import HumanMessage

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat_sync(req: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    """Synchronous chat endpoint. Runs full agent pipeline, returns final result."""
    return run_agent_sync(db, req.query, session_id=req.session_id or "")


@router.post("/stream")
def chat_stream(req: ChatRequest, db: Session = Depends(get_db)):
    """Streaming chat endpoint. Emits SSE events for each agent step."""

    def event_generator():
        t_start = time.perf_counter()

        try:
            graph = build_agent_graph(db)
            messages = [HumanMessage(content=req.query)]
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

            config = {"recursion_limit": 25}
            prev_traces_len = 0
            final_state = initial_state
            steps_count = 0
            reflections_count = 0

            for step_output in graph.stream(initial_state, config=config):
                for node_name, node_state in step_output.items():
                    final_state = {**final_state, **node_state}

                    # Emit step_start before executor runs (based on plan)
                    if node_name == "executor":
                        plan = final_state.get("plan", [])
                        step_idx = node_state.get("plan_step_index", 0) - 1
                        if 0 <= step_idx < len(plan):
                            step_spec = plan[step_idx]
                            start_data = {
                                "index": step_idx,
                                "action": step_spec.get("action", ""),
                                "reason": step_spec.get("reason", ""),
                            }
                            yield f"event: step_start\ndata: {json.dumps(start_data, ensure_ascii=False)}\n\n"

                    # Emit new step traces (step_done events)
                    traces = node_state.get("step_traces", [])
                    for trace in traces[prev_traces_len:]:
                        yield f"event: step_done\ndata: {json.dumps(trace, ensure_ascii=False)}\n\n"
                        steps_count += 1
                    prev_traces_len = len(final_state.get("step_traces", []))

                    # Emit intent
                    if node_name == "intent" and node_state.get("intent"):
                        yield f"event: intent\ndata: {json.dumps(node_state['intent'], ensure_ascii=False)}\n\n"

                    # Emit plan
                    if node_name == "planner" and node_state.get("plan"):
                        plan_data = {"steps": node_state["plan"], "total_steps": len(node_state["plan"])}
                        yield f"event: plan\ndata: {json.dumps(plan_data, ensure_ascii=False)}\n\n"

                    # Emit reflection
                    if node_name == "reflection" and node_state.get("reflection_result"):
                        reflections_count += 1
                        yield f"event: reflection\ndata: {json.dumps(node_state['reflection_result'], ensure_ascii=False)}\n\n"

                    # Emit re_plan
                    if node_name == "re_planner" and node_state.get("plan"):
                        re_plan_data = {"new_steps": node_state["plan"][-3:]}
                        yield f"event: re_plan\ndata: {json.dumps(re_plan_data, ensure_ascii=False)}\n\n"

            # Emit final answer as token
            answer = final_state.get("final_answer", "")
            if answer:
                yield f"event: token\ndata: {json.dumps({'t': answer}, ensure_ascii=False)}\n\n"

            # Emit sources
            sources = final_state.get("sources", [])
            if sources:
                sources_data = [s.model_dump() if hasattr(s, "model_dump") else s for s in sources]
                yield f"event: sources\ndata: {json.dumps({'sources': sources_data}, ensure_ascii=False)}\n\n"

            total_ms = round((time.perf_counter() - t_start) * 1000, 2)
            done_data = {
                "total_ms": total_ms,
                "steps_count": steps_count,
                "reflections": reflections_count,
            }
            yield f"event: done\ndata: {json.dumps(done_data, ensure_ascii=False)}\n\n"

        except Exception as e:
            error_data = {"message": str(e), "type": type(e).__name__}
            yield f"event: error\ndata: {json.dumps(error_data, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
