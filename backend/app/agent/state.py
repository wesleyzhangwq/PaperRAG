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


class AgentState(TypedDict, total=False):
    # required core
    messages: Annotated[list, add_messages]
    intent: Optional[dict]
    plan: list[StepSpec]
    plan_step_index: int
    retrieval_context: list[Document]
    step_traces: list[StepTrace]
    reflection_count: int
    final_answer: Optional[str]
    reflection_result: Optional[dict]
    sources: Optional[list]
    # optional metadata (filled progressively for the presentation layer)
    is_fallback: bool                       # any retrieve_local fell back to user query
    evaluator_result: Optional[dict]        # latest evaluate_docs output
    evaluator_parse_failed: bool            # short-cut flag for confidence calc
    presentation: Optional[dict]            # final structured payload for the UI
