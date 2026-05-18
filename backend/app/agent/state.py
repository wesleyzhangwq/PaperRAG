"""Agent state schema and supporting types."""
from __future__ import annotations

from typing import Annotated, Optional, TypedDict

from langchain_core.documents import Document
from langgraph.graph.message import add_messages


class StepSpec(TypedDict):
    action: str
    params: dict
    reason: str


class StepTrace(TypedDict):
    node: str
    action: str
    input_summary: str
    output_summary: str
    duration_ms: float


class ReflectionResult(TypedDict):
    passed: bool
    citation_ok: bool
    completeness_ok: bool
    logic_ok: bool
    issues: list[str]
    fix_strategy: Optional[str]


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: Optional[dict]
    plan: list[StepSpec]
    plan_step_index: int
    retrieval_context: list[Document]
    step_traces: list[StepTrace]
    reflection_count: int
    final_answer: Optional[str]
