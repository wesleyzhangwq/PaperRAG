"""LangGraph agent graph: build, compile, and run.

v2 orchestration — node-per-stage, mirroring the enterprise agentic RAG
pipeline (安全校验 → 意图 → 规划 → 检索路由 → 多源检索 → 证据处理 →
充分性判断 → 生成 → groundedness → 引用过滤 → 输出):

    guard ──blocked──────────────────────────────────────────┐
      │ok                                                    │
    intent → planner → route → executor ⟲                    │
                                  │ (plan exhausted)         │
                               evidence                      │
                                  │                          │
                             sufficiency ──insufficient──→ re_planner
                                  │ sufficient/degraded        │
                              synthesis ←──re_generate──┐     │
                                  │                     │     │
                             groundedness ──re_retrieve─┼──→ re_planner
                                  │ pass/budget         │     (loops back to executor)
                             citation_gate              │
                                  │                     │
                             presentation ←─────────────┘
                                  │
                                 END
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph
from sqlalchemy.orm import Session

from app.agent.checkpoint import agent_run_config, open_sync_checkpointer
from app.agent.nodes.citation_gate import citation_gate_node
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
from app.core.config import get_settings
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


def route_after_reflection(state: AgentState | dict, max_reflections: int) -> str:
    """Route after the groundedness check.

    `re_retrieve` intentionally goes through `re_planner`: at this point the
    executor index is already past retrieval steps, so the graph needs a fresh
    supplementary plan before executing retrieval again.
    """
    reflection = state.get("reflection_result", {}) or {}
    if reflection.get("passed", True):
        return "citation_gate"
    if state.get("reflection_count", 0) >= max_reflections:
        return "citation_gate"
    strategy = reflection.get("fix_strategy")
    if strategy == "re_generate":
        return "synthesis"
    return "re_planner"


def build_agent_graph(db: Session, *, checkpointer=None) -> object:
    """Build and compile the agentic RAG graph (v2 orchestration)."""
    settings = get_settings()

    def _guard(state: AgentState) -> dict:
        return guard_node(state, query=_extract_query(state))

    def _intent(state: AgentState) -> dict:
        return intent_node(state, query=_extract_query(state))

    def _planner(state: AgentState) -> dict:
        return planner_node(state, query=_extract_query(state))

    def _route(state: AgentState) -> dict:
        return route_node(state, query=_extract_query(state))

    def _executor(state: AgentState) -> dict:
        return executor_node(state, db=db)

    def _evidence(state: AgentState) -> dict:
        return evidence_node(state)

    def _sufficiency(state: AgentState) -> dict:
        return sufficiency_node(state, query=_extract_query(state))

    def _synthesis(state: AgentState) -> dict:
        query = _extract_query(state)
        # If re-generating after a groundedness failure, pass issues as constraints
        reflection = state.get("reflection_result") or {}
        issues = reflection.get("issues") if not reflection.get("passed", True) else None
        return synthesis_node(state, query=query, issues=issues or None)

    def _groundedness(state: AgentState) -> dict:
        return groundedness_node(state, query=_extract_query(state))

    def _re_planner(state: AgentState) -> dict:
        query = _extract_query(state)
        # Entered from sufficiency (missing aspects) or groundedness (issues).
        reflection = state.get("reflection_result") or {}
        sufficiency = state.get("sufficiency_result") or {}
        issues = list(reflection.get("issues") or []) or [str(sufficiency.get("reason") or "")]
        missing = list(sufficiency.get("missing_aspects") or [])
        return re_planner_node(
            state,
            query=query,
            issues=[i for i in issues if i],
            missing_aspects=missing,
        )

    def _citation_gate(state: AgentState) -> dict:
        return citation_gate_node(state, db=db)

    def _presentation(state: AgentState) -> dict:
        return presentation_node(state, db=db)

    def _should_continue_executing(state: AgentState) -> str:
        return "executor" if state["plan_step_index"] < len(state["plan"]) else "evidence"

    def _after_sufficiency(state: AgentState) -> str:
        return after_sufficiency(state)

    def _after_reflection(state: AgentState) -> str:
        return route_after_reflection(state, settings.agent_max_reflections)

    graph = StateGraph(AgentState)

    graph.add_node("guard", _guard)
    graph.add_node("intent", _intent)
    graph.add_node("planner", _planner)
    graph.add_node("route", _route)
    graph.add_node("executor", _executor)
    graph.add_node("evidence", _evidence)
    graph.add_node("sufficiency", _sufficiency)
    graph.add_node("synthesis", _synthesis)
    graph.add_node("groundedness", _groundedness)
    graph.add_node("re_planner", _re_planner)
    graph.add_node("citation_gate", _citation_gate)
    graph.add_node("presentation", _presentation)

    graph.set_entry_point("guard")
    graph.add_conditional_edges(
        "guard",
        route_after_guard,
        {"intent": "intent", "presentation": "presentation"},
    )
    graph.add_edge("intent", "planner")
    graph.add_edge("planner", "route")
    graph.add_edge("route", "executor")
    graph.add_conditional_edges(
        "executor",
        _should_continue_executing,
        {"executor": "executor", "evidence": "evidence"},
    )
    graph.add_edge("evidence", "sufficiency")
    graph.add_conditional_edges(
        "sufficiency",
        _after_sufficiency,
        {"synthesis": "synthesis", "re_planner": "re_planner"},
    )
    graph.add_edge("re_planner", "executor")
    graph.add_edge("synthesis", "groundedness")
    graph.add_conditional_edges(
        "groundedness",
        _after_reflection,
        {"citation_gate": "citation_gate", "re_planner": "re_planner", "synthesis": "synthesis"},
    )
    graph.add_edge("citation_gate", "presentation")
    graph.add_edge("presentation", END)

    return graph.compile(checkpointer=checkpointer)


def initial_agent_state(messages: list) -> AgentState:
    """Canonical initial state shared by sync and streaming entrypoints."""
    return {
        "messages": messages,
        "intent": None,
        "plan": [],
        "plan_step_index": 0,
        "retrieval_context": [],
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
    }


def run_agent_sync(
    db: Session,
    query: str,
    session_id: str = "",
    history: list | None = None,
) -> ChatResponse:
    """Run the agent synchronously, return ChatResponse.

    `history` may be either a list of LangChain message objects (preferred)
    or a list of (role, content) tuples (legacy).
    """
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
    with open_sync_checkpointer() as checkpointer:
        graph = build_agent_graph(db, checkpointer=checkpointer)
        result = graph.invoke(initial_state, config=config)

    answer = result.get("final_answer", "Agent failed to produce an answer.")
    sources = result.get("sources", [])
    step_traces = result.get("step_traces", [])

    return ChatResponse(
        answer=answer,
        sources=sources if isinstance(sources, list) else [],
        used_chunks=len(result.get("retrieval_context", [])),
        step_traces=step_traces,
        reflection_result=result.get("reflection_result"),
    )
