"""LangGraph agent graph: build, compile, and run.

v3 orchestration uses eight checkpoint/control nodes while retaining the
thirteen former node responsibilities and their fine-grained SSE/StepTrace
observability. Adjacent stages that never branch independently are composed
inside one graph node:

    guard ──blocked──────────────────────────────────────────┐
      │ok                                                    │
    analyze(intent + complexity)                             │
      │                                                      │
    plan(planner/re-planner + route) ←────────────────┐      │
      │                                               │      │
    executor ⟲                                       │      │
      │ plan exhausted                               │      │
    evidence_gate(evidence + sufficiency) ───────────┘      │
      │ sufficient/degraded                                  │
    synthesis ←────────────────────────── re_generate ┐      │
      │                                               │      │
    groundedness ──────────────────────── re_retrieve ┘      │
      │ pass/budget                                          │
    finalize(citation_gate + presentation) ←─────────────────┘
      │
     END

Graph nodes now align with branch, retry, or persistence boundaries. The
internal stage functions remain independently testable and observable.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph
from sqlalchemy.orm import Session

from app.agent.checkpoint import agent_run_config, open_sync_checkpointer
from app.agent.nodes.citation_gate import citation_gate_node
from app.agent.nodes.complexity_router import complexity_router_node
from app.agent.nodes.evidence import evidence_node
from app.agent.nodes.executor import executor_node
from app.agent.nodes.groundedness import groundedness_node
from app.agent.nodes.guard import guard_node
from app.agent.nodes.intent import intent_node
from app.agent.nodes.planner import planner_node, re_planner_node
from app.agent.nodes.presentation import presentation_node
from app.agent.nodes.route import route_node
from app.agent.nodes.sufficiency import after_sufficiency, sufficiency_node
from app.agent.nodes.synthesis import synthesis_node
from app.agent.state import AgentState
from app.agent.telemetry import DEFAULT_FALLBACK_TELEMETRY
from app.core.config import get_settings
from app.observability.llm_usage import collect_llm_usage, current_collector
from app.schemas.chat import ChatResponse


def _extract_query(state: AgentState) -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def route_after_guard(state: AgentState | dict) -> str:
    """Blocked queries short-circuit straight to presentation (refusal text
    was already placed into final_answer by the guard node)."""
    guard = state.get("guard_result") or {}
    return "intent" if guard.get("allowed", True) else "presentation"


def route_after_complexity(state: AgentState | dict) -> str:
    """Fast-local decisions already contain a deterministic plan.

    Every missing, forced, fallback, or escalated decision goes through the
    full planner; this is deliberately fail-closed.
    """
    return "route" if state.get("execution_path") == "fast_local" else "planner"


def route_after_reflection(state: AgentState | dict, max_reflections: int) -> str:
    """Route after the groundedness check.

    `re_retrieve` intentionally goes through `re_planner`: at this point the
    executor index is already past retrieval steps, so the graph needs a fresh
    supplementary plan before executing retrieval again.
    """
    reflection = state.get("reflection_result", {}) or {}
    if reflection.get("passed", True):
        return "citation_gate"
    if state.get("degraded"):
        return "citation_gate"
    if state.get("reflection_count", 0) >= max_reflections:
        return "citation_gate"
    strategy = reflection.get("fix_strategy")
    if strategy == "re_generate":
        return "synthesis"
    return "re_planner"


def build_agent_graph(db: Session, *, checkpointer=None) -> object:
    """Build and compile the lean agentic RAG graph (v3 orchestration)."""
    settings = get_settings()

    def _guard(state: AgentState) -> dict:
        return guard_node(state, query=_extract_query(state))

    def _apply(state: AgentState | dict, *updates: dict) -> tuple[dict, dict]:
        """Apply sequential partial updates without returning the full state.

        Returning a full state would replay reducer-backed fields such as
        ``messages``. Composed nodes therefore maintain a local working view
        while emitting only the accumulated partial update.
        """
        working = dict(state)
        combined: dict = {}
        for update in updates:
            working.update(update)
            combined.update(update)
        return working, combined

    def _analyze(state: AgentState) -> dict:
        query = _extract_query(state)
        intent_update = intent_node(state, query=query)
        working, combined = _apply(state, intent_update)
        complexity_update = complexity_router_node(working, query=query)
        _, combined = _apply(state, combined, complexity_update)
        return combined

    def _plan(state: AgentState) -> dict:
        """Create or supplement a plan, then enforce source-routing policy."""
        query = _extract_query(state)
        working = dict(state)
        combined: dict = {}

        fast_initial_plan = (
            state.get("execution_path") == "fast_local"
            and bool(state.get("plan"))
            and not state.get("fast_path_escalated", False)
        )
        fast_sufficiency_escalation = (
            state.get("execution_path") == "fast_escalated"
            and state.get("fast_path_escalated", False)
            and "evidence_insufficient_escalation"
            in list((state.get("complexity_decision") or {}).get("reason_codes") or [])
            and int(state.get("sufficiency_round", 0)) == 1
        )

        if not fast_initial_plan:
            if not state.get("plan") or fast_sufficiency_escalation:
                plan_update = planner_node(working, query=query)
            else:
                reflection = state.get("reflection_result") or {}
                sufficiency = state.get("sufficiency_result") or {}
                issues = list(reflection.get("issues") or []) or [
                    str(sufficiency.get("reason") or "")
                ]
                missing = list(sufficiency.get("missing_aspects") or [])
                plan_update = re_planner_node(
                    working,
                    query=query,
                    issues=[item for item in issues if item],
                    missing_aspects=missing,
                )
            working, combined = _apply(working, plan_update)

        route_update = route_node(working, query=query)
        _, combined = _apply(state, combined, route_update)
        return combined

    def _executor(state: AgentState) -> dict:
        return executor_node(state, db=db)

    def _evidence_gate(state: AgentState) -> dict:
        evidence_update = evidence_node(state)
        working, combined = _apply(state, evidence_update)
        sufficiency_update = sufficiency_node(working, query=_extract_query(state))
        _, combined = _apply(state, combined, sufficiency_update)
        return combined

    def _synthesis(state: AgentState) -> dict:
        query = _extract_query(state)
        # If re-generating after a groundedness failure, pass issues as constraints
        reflection = state.get("reflection_result") or {}
        issues = reflection.get("issues") if not reflection.get("passed", True) else None
        return synthesis_node(state, query=query, issues=issues or None)

    def _groundedness(state: AgentState) -> dict:
        return groundedness_node(state, query=_extract_query(state))

    def _finalize(state: AgentState) -> dict:
        # A guard refusal contains no citations to resolve and must not touch
        # retrieval persistence. All answer paths still pass citation_gate.
        if not (state.get("guard_result") or {}).get("allowed", True):
            return presentation_node(state, db=db)
        citation_update = citation_gate_node(state, db=db)
        working, combined = _apply(state, citation_update)
        presentation_update = presentation_node(working, db=db)
        _, combined = _apply(state, combined, presentation_update)
        return combined

    def _should_continue_executing(state: AgentState) -> str:
        return "executor" if state["plan_step_index"] < len(state["plan"]) else "evidence_gate"

    def _after_sufficiency(state: AgentState) -> str:
        return after_sufficiency(state)

    def _after_reflection(state: AgentState) -> str:
        return route_after_reflection(state, settings.agent_max_reflections)

    graph = StateGraph(AgentState)

    graph.add_node("guard", _guard)
    graph.add_node("analyze", _analyze)
    graph.add_node("plan", _plan)
    graph.add_node("executor", _executor)
    graph.add_node("evidence_gate", _evidence_gate)
    graph.add_node("synthesis", _synthesis)
    graph.add_node("groundedness", _groundedness)
    graph.add_node("finalize", _finalize)

    graph.set_entry_point("guard")
    graph.add_conditional_edges(
        "guard",
        route_after_guard,
        {"intent": "analyze", "presentation": "finalize"},
    )
    graph.add_edge("analyze", "plan")
    graph.add_edge("plan", "executor")
    graph.add_conditional_edges(
        "executor",
        _should_continue_executing,
        {"executor": "executor", "evidence_gate": "evidence_gate"},
    )
    graph.add_conditional_edges(
        "evidence_gate",
        _after_sufficiency,
        {"synthesis": "synthesis", "re_planner": "plan", "planner": "plan"},
    )
    graph.add_edge("synthesis", "groundedness")
    graph.add_conditional_edges(
        "groundedness",
        _after_reflection,
        {"citation_gate": "finalize", "re_planner": "plan", "synthesis": "synthesis"},
    )
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)


def initial_agent_state(messages: list) -> AgentState:
    """Canonical initial state shared by sync and streaming entrypoints."""
    return {
        "messages": messages,
        "intent": None,
        "intent_status": "unknown",
        "execution_path": None,
        "fast_path_escalated": False,
        "complexity_decision": None,
        "plan": [],
        "plan_step_index": 0,
        "retrieval_context": [],
        "retrieved_paper_ids": [],
        "synthesis_context_count": 0,
        "synthesis_context_paper_ids": [],
        "step_traces": [],
        "reflection_count": 0,
        "sufficiency_round": 0,
        "final_answer": None,
        "reflection_result": None,
        "sources": None,
        "guard_result": None,
        "route_decision": None,
        "evidence_stats": None,
        "sufficiency_result": None,
        "degraded": False,
        "removed_citations": [],
        "fallback_telemetry": {
            **DEFAULT_FALLBACK_TELEMETRY,
            "failure_classes": [],
            "events": [],
        },
        "llm_usage": [],
        "synthesis_failed": False,
    }


def _run_agent_to_state(
    db: Session,
    query: str,
    session_id: str = "",
    history: list | None = None,
) -> dict:
    """Run the graph synchronously and return its final internal state."""
    thread_id = session_id or "default"

    messages = []
    if history:
        for item in history:
            if isinstance(item, tuple) and len(item) == 2:
                role, content = item
                messages.append(
                    HumanMessage(content=content) if role == "user" else AIMessage(content=content)
                )
            else:
                messages.append(item)
    messages.append(HumanMessage(content=query))

    initial_state = initial_agent_state(messages)

    config = agent_run_config(thread_id)

    def invoke_graph() -> dict:
        with open_sync_checkpointer() as checkpointer:
            graph = build_agent_graph(db, checkpointer=checkpointer)
            return graph.invoke(initial_state, config=config)

    collector = current_collector()
    if collector is not None:
        start = len(collector.records)
        result = invoke_graph()
        result["llm_usage"] = collector.snapshot(start)
        return result
    with collect_llm_usage() as run_collector:
        result = invoke_graph()
        result["llm_usage"] = run_collector.snapshot()
        return result


def _response_from_state(result: dict) -> ChatResponse:

    answer = result.get("final_answer", "Agent failed to produce an answer.")
    sources = result.get("sources", [])
    step_traces = result.get("step_traces", [])

    return ChatResponse(
        answer=answer,
        sources=sources if isinstance(sources, list) else [],
        used_chunks=int(
            result.get("synthesis_context_count", len(result.get("retrieval_context", [])))
        ),
        step_traces=step_traces,
        reflection_result=result.get("reflection_result"),
        presentation=result.get("presentation"),
        sufficiency_result=result.get("sufficiency_result"),
        removed_citations=list(result.get("removed_citations") or []),
        synthesis_context_paper_ids=_unique_paper_ids(
            result.get("synthesis_context_paper_ids") or []
        ),
        fallback_telemetry=result.get("fallback_telemetry"),
        llm_usage=list(result.get("llm_usage") or []),
        degraded=bool(result.get("degraded")),
        execution_path=result.get("execution_path"),
        complexity_decision=result.get("complexity_decision"),
    )


def _unique_paper_ids(items: list) -> list[str]:
    paper_ids: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = str(item or "").strip()
        if value and value not in seen:
            paper_ids.append(value)
            seen.add(value)
    return paper_ids


def run_agent_sync(
    db: Session,
    query: str,
    session_id: str = "",
    history: list | None = None,
) -> ChatResponse:
    """Run the agent synchronously, return the public ChatResponse.

    `history` may be either a list of LangChain message objects (preferred)
    or a list of (role, content) tuples (legacy).
    """
    return _response_from_state(_run_agent_to_state(db, query, session_id, history))


def run_agent_eval_sync(
    db: Session,
    query: str,
    session_id: str = "",
    history: list | None = None,
) -> tuple[ChatResponse, list[str], list[str]]:
    """Run the agent and expose retrieval evidence for offline evaluation only."""
    result = _run_agent_to_state(db, query, session_id, history)
    retrieved_paper_ids = _unique_paper_ids(result.get("retrieved_paper_ids") or [])
    context_values = result.get("synthesis_context_paper_ids")
    if context_values is None:
        context_values = [
            (document.metadata or {}).get("paper_id")
            for document in result.get("retrieval_context") or []
        ]
    context_paper_ids = _unique_paper_ids(context_values)
    return _response_from_state(result), retrieved_paper_ids, context_paper_ids
