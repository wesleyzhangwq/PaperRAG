"""Tests for productized presentation payload."""
from unittest.mock import MagicMock

from langchain_core.documents import Document

from app.agent.nodes.presentation import presentation_node


def test_retrieval_summary_distinguishes_retrieved_from_cited_papers():
    db = MagicMock()
    db.query.return_value.filter.return_value.one_or_none.return_value = None
    context = [
        Document(page_content=f"chunk {i}", metadata={"paper_id": f"2604.0000{i}", "title": f"Paper {i}"})
        for i in range(5)
    ]
    state = {
        "final_answer": "Only one paper cited [arxiv:2604.00000]",
        "sources": [{"paper_id": "2604.00000", "title": "Paper 0", "authors": [], "year": 2026}],
        "retrieval_context": context,
        "step_traces": [],
        "evaluator_result": {"sufficient": True},
        "reflection_result": {"passed": True},
    }

    result = presentation_node(state, db=db)
    summary = result["presentation"]["retrieval_summary"]

    assert summary["total_papers"] == 5
    assert summary["cited_papers"] == 1
    assert "最终回答引用" in summary["narrative"]


def test_steps_use_per_trace_detail_not_latest_evaluator_state():
    db = MagicMock()
    db.query.return_value.filter.return_value.one_or_none.return_value = None
    state = {
        "final_answer": "Answer [arxiv:2604.00001]",
        "sources": [{"paper_id": "2604.00001", "title": "Paper", "authors": [], "year": 2026}],
        "retrieval_context": [
            Document(page_content="chunk", metadata={"paper_id": "2604.00001", "title": "Paper"}),
        ],
        "evaluator_result": {
            "sufficient": False,
            "missing_aspects": ["latest missing"],
            "parse_failed": False,
            "reason": "latest",
        },
        "reflection_result": {"passed": False},
        "step_traces": [
            {
                "node": "executor_node",
                "action": "evaluate_docs",
                "input_summary": "evaluate_docs()",
                "output_summary": "sufficient=True",
                "duration_ms": 10,
                "params": {},
                "reason": "first eval",
                "detail": {"sufficient": True, "parse_failed": False, "missing_aspects": [], "reason": "enough"},
            },
            {
                "node": "executor_node",
                "action": "evaluate_docs",
                "input_summary": "evaluate_docs()",
                "output_summary": "sufficient=False, missing: 1 aspects",
                "duration_ms": 12,
                "params": {},
                "reason": "second eval",
                "detail": {
                    "sufficient": False,
                    "parse_failed": False,
                    "missing_aspects": ["method details"],
                    "reason": "incomplete",
                },
            },
        ],
    }

    result = presentation_node(state, db=db)
    steps = result["presentation"]["steps"]

    assert steps[0]["debug"]["extra"]["sufficient"] is True
    assert steps[0]["status"] == "completed"
    assert steps[1]["debug"]["extra"]["missing_aspects"] == ["method details"]
    assert steps[1]["status"] == "warning"
