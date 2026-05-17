# backend/app/services/agent.py
"""LangGraph ReAct agent for PaperRAG."""
from __future__ import annotations

import logging
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.services.tools import (
    compare_papers,
    filter_papers,
    get_paper_detail,
    search_papers,
)

log = logging.getLogger("app.services.agent")

AGENT_SYSTEM_PROMPT = """你是一个严谨的学术论文问答助手，可以使用工具来搜索和分析论文。

你有以下工具可用：
1. search_papers: 搜索相关论文（使用向量+BM25混合检索）
2. filter_papers: 按类别或年份范围过滤搜索论文
3. get_paper_detail: 获取某篇论文的完整元数据（标题、摘要、作者等）
4. compare_papers: 并排对比多篇论文

硬性规则：
1. 必须先使用工具搜索论文，再基于搜索结果回答。禁止凭空回答。
2. 每条论据/结论末尾必须追加 [arxiv:PAPER_ID] 引用。
3. 禁止编造 paper_id；只能使用工具返回的 paper_id。
4. 若工具返回结果不足以回答，如实说"参考资料不足以回答该问题"。
5. 输出使用简洁的 Markdown，中文作答（除非用户用英文提问）。"""

MAX_STEPS = 5


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def _get_agent_llm() -> ChatOpenAI:
    s = get_settings()
    if not s.llm_api_key:
        raise RuntimeError("Missing LLM_API_KEY in .env")
    return ChatOpenAI(
        model=s.llm_model,
        base_url=s.llm_api_base,
        api_key=s.llm_api_key,
        temperature=0.2,
        max_retries=max(0, s.llm_max_retries),
    )


def build_tools(db: Session) -> list:
    @tool
    def get_paper_detail_tool(paper_id: str) -> str:
        """Get full metadata for a specific paper by its arxiv paper_id.
        Returns title, authors, year, categories, abstract."""
        return get_paper_detail(db, paper_id)

    @tool
    def compare_papers_tool(paper_ids: list[str]) -> str:
        """Compare multiple papers side by side.
        Returns a formatted comparison of titles, authors, years, and abstracts."""
        return compare_papers(db, paper_ids)

    return [search_papers, filter_papers, get_paper_detail_tool, compare_papers_tool]


def build_graph(db: Session) -> StateGraph:
    tools = build_tools(db)
    tool_map = {t.name: t for t in tools}
    llm = _get_agent_llm().bind_tools(tools)

    def agent_think(state: AgentState) -> AgentState:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    def execute_tool(state: AgentState) -> AgentState:
        last = state["messages"][-1]
        results = []
        for call in last.tool_calls:
            fn = tool_map.get(call["name"])
            if fn is None:
                results.append(ToolMessage(
                    content=f"Unknown tool: {call['name']}",
                    tool_call_id=call["id"],
                ))
                continue
            try:
                output = fn.invoke(call["args"])
            except Exception as e:
                output = f"Tool error: {type(e).__name__}: {e}"
            results.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
        return {"messages": results}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "execute_tool"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent_think", agent_think)
    graph.add_node("execute_tool", execute_tool)
    graph.set_entry_point("agent_think")
    graph.add_conditional_edges("agent_think", should_continue)
    graph.add_edge("execute_tool", "agent_think")

    return graph.compile()


def run_agent(db: Session, query: str, history: list[tuple[str, str]]) -> str:
    graph = build_graph(db)

    messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)]
    for role, content in history:
        if role == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=query))

    config = {"recursion_limit": MAX_STEPS * 2 + 1}
    result = graph.invoke({"messages": messages}, config=config)

    final = result["messages"][-1]
    return final.content if hasattr(final, "content") else str(final)
