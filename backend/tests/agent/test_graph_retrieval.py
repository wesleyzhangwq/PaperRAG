from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from app.agent.state import StepSpec


def _state() -> dict:
    return {
        "messages": [HumanMessage(content="compare A and B")],
        "intent": {"type": "comparison", "complexity": "high"},
        "plan": [
            StepSpec(
                action="retrieve_graph",
                params={"query": "compare A and B", "top_k": 3},
                reason="cross-paper expansion",
            )
        ],
        "plan_step_index": 0,
        "retrieval_context": [
            Document(
                page_content="seed",
                metadata={
                    "paper_id": "A",
                    "retrieval_score": 0.9,
                    "retrieval_source": "local",
                },
            )
        ],
        "step_traces": [],
        "reflection_count": 0,
    }


def test_executor_graph_action_adds_only_second_pass_local_chunks() -> None:
    from app.agent.nodes.executor import executor_node

    expanded = Document(
        page_content="local evidence",
        metadata={
            "paper_id": "B",
            "retrieval_score": 0.8,
            "graph_score": 0.9,
            "retrieval_source": "graph_local",
        },
    )
    report = SimpleNamespace(
        seed_paper_ids=("A",),
        candidates=(SimpleNamespace(paper_id="B"),),
        added_chunks=1,
        fallback_reason=None,
        graph_elapsed_ms=12.5,
    )
    with patch(
        "app.agent.nodes.executor.retrieve_graph_context",
        return_value=([expanded], report),
    ):
        result = executor_node(_state(), db=MagicMock())

    assert [doc.metadata["paper_id"] for doc in result["retrieval_context"]] == ["A", "B"]
    assert result["retrieved_paper_ids"] == ["B"]
    assert result["step_traces"][-1]["detail"]["added"] == 1
    assert result["step_traces"][-1]["detail"]["fallback_reason"] is None


def test_executor_graph_action_warns_and_keeps_context_on_fallback() -> None:
    from app.agent.nodes.executor import executor_node

    state = _state()
    report = SimpleNamespace(
        seed_paper_ids=("A",),
        candidates=(),
        added_chunks=0,
        fallback_reason="neo4j_unavailable",
        graph_elapsed_ms=0.0,
    )
    with patch(
        "app.agent.nodes.executor.retrieve_graph_context",
        return_value=(state["retrieval_context"], report),
    ):
        result = executor_node(state, db=MagicMock())

    assert len(result["retrieval_context"]) == 1
    assert result["step_traces"][-1]["detail"]["fallback_reason"] == "neo4j_unavailable"


def test_evidence_prefers_retrieval_score_over_legacy_score() -> None:
    from app.agent.nodes.evidence import _score

    doc = Document(
        page_content="evidence",
        metadata={"score": 0.1, "retrieval_score": 0.8},
    )

    assert _score(doc) == 0.8
