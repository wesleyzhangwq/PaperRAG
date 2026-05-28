"""LangGraph agent graph: build, compile, and run."""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph
from sqlalchemy.orm import Session

from app.agent.nodes.executor import executor_node
from app.agent.nodes.final_answer import final_answer_node
from app.agent.nodes.intent import intent_node
from app.agent.nodes.planner import planner_node, re_planner_node
from app.agent.nodes.presentation import presentation_node
from app.agent.nodes.reflection import reflection_node
from app.agent.nodes.synthesis import synthesis_node
from app.agent.state import AgentState
from app.core.config import get_settings
from app.schemas.chat import ChatResponse


def _extract_query(state: AgentState) -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def build_agent_graph(db: Session) -> object:
    """Build and compile the agentic RAG graph."""
    settings = get_settings()

    def _intent(state: AgentState) -> dict:
        query = _extract_query(state)
        return intent_node(state, query=query)

    def _planner(state: AgentState) -> dict:
        query = _extract_query(state)
        return planner_node(state, query=query)

    def _executor(state: AgentState) -> dict:
        return executor_node(state, db=db)

    def _synthesis(state: AgentState) -> dict:
        query = _extract_query(state)
        # If re-generating after reflection failure, pass issues as constraints
        reflection = state.get("reflection_result") or {}
        issues = reflection.get("issues") if not reflection.get("passed", True) else None
        return synthesis_node(state, query=query, issues=issues or None)

    def _reflection(state: AgentState) -> dict:
        query = _extract_query(state)
        return reflection_node(state, query=query)

    def _re_planner(state: AgentState) -> dict:
        query = _extract_query(state)
        reflection = state.get("reflection_result", {})
        return re_planner_node(
            state,
            query=query,
            issues=reflection.get("issues", []),
            missing_aspects=reflection.get("missing_aspects", []),
        )

    def _final_answer(state: AgentState) -> dict:
        return final_answer_node(state, db=db)

    def _presentation(state: AgentState) -> dict:
        return presentation_node(state, db=db)

    def _should_continue_executing(state: AgentState) -> str:
        idx = state["plan_step_index"]
        plan = state["plan"]
        if idx < len(plan) and plan[idx]["action"] != "reasoning_synthesis":
            return "executor"
        return "synthesis"

    def _after_reflection(state: AgentState) -> str:
        reflection = state.get("reflection_result", {})
        if reflection.get("passed", True):
            return "final_answer"
        if state["reflection_count"] >= settings.agent_max_reflections:
            return "final_answer"
        strategy = reflection.get("fix_strategy")
        if strategy == "re_generate":
            # Context is sufficient but answer has issues — re-synthesize
            return "synthesis"
        # Default: re_retrieve or unrecognized → re_planner for new retrieval
        return "re_planner"

    graph = StateGraph(AgentState)

    graph.add_node("intent", _intent)
    graph.add_node("planner", _planner)
    graph.add_node("executor", _executor)
    graph.add_node("synthesis", _synthesis)
    graph.add_node("reflection", _reflection)
    graph.add_node("re_planner", _re_planner)
    graph.add_node("final_answer", _final_answer)
    graph.add_node("presentation", _presentation)

    graph.set_entry_point("intent")
    graph.add_edge("intent", "planner")
    graph.add_edge("planner", "executor")
    graph.add_conditional_edges("executor", _should_continue_executing)
    graph.add_edge("synthesis", "reflection")
    graph.add_conditional_edges(
        "reflection",
        _after_reflection,
        {"final_answer": "final_answer", "re_planner": "re_planner", "synthesis": "synthesis"},
    )
    graph.add_edge("re_planner", "executor")
    graph.add_edge("final_answer", "presentation")
    graph.add_edge("presentation", END)

    return graph.compile()


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
    graph = build_agent_graph(db)

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
