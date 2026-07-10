"""Agent state schema and supporting types."""
from __future__ import annotations

from typing import Annotated, Optional, TypedDict

from langchain_core.documents import Document
from langgraph.graph.message import add_messages


class StepSpec(TypedDict):
    action: str
    params: dict
    reason: str


class StepTrace(TypedDict, total=False):
    index: int
    node: str
    action: str
    input_summary: str
    output_summary: str
    duration_ms: float
    params: dict
    reason: str
    detail: dict


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
    retrieved_paper_ids: list[str]
    synthesis_context_count: int
    synthesis_context_paper_ids: list[str]
    step_traces: list[StepTrace]
    reflection_count: int
    final_answer: Optional[str]
    reflection_result: Optional[dict]
    sources: Optional[list]
    # enterprise pipeline stages (v2 orchestration)
    guard_result: Optional[dict]            # {allowed, reason, flags} from the guard node
    route_decision: Optional[dict]          # {sources, adjustments} from the retrieval router
    evidence_stats: Optional[dict]          # {before, after, dropped_*} from evidence processing
    sufficiency_result: Optional[dict]      # structured evaluate_docs output (graph-level node)
    sufficiency_round: int                  # supplementary-retrieval budget counter
    degraded: bool                          # sufficiency budget exhausted → answer with caveat
    removed_citations: list[str]            # hallucinated citation ids stripped by citation_gate
    # optional metadata (filled progressively for the presentation layer)
    is_fallback: bool                       # any retrieve_local fell back to user query
    evaluator_result: Optional[dict]        # latest evaluate_docs output
    evaluator_parse_failed: bool            # short-cut flag for confidence calc
    presentation: Optional[dict]            # final structured payload for the UI
