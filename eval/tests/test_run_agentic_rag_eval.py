from unittest.mock import patch

from app.schemas.chat import ChatResponse
from eval.run_agentic_rag_eval import run_agentic_case
from eval.run_agentic_rag_eval import run_cases_with_checkpoint


def test_run_agentic_case_separates_raw_retrieval_from_final_context() -> None:
    response = ChatResponse(
        answer="Answer [arxiv:1111.1111]",
        sources=[],
        used_chunks=1,
        step_traces=[],
        reflection_result={"passed": True},
    )
    item = {
        "qid": "q1",
        "query": "test query",
        "type": "comparison",
        "difficulty": "hard",
        "expected_paper_ids": ["1111.1111", "2222.2222"],
        "expected_mode": "answer",
    }

    with patch(
        "app.agent.graph.run_agent_eval_sync",
        return_value=(response, ["1111.1111", "2222.2222"], ["1111.1111"]),
    ):
        row = run_agentic_case(item, idx=1, total=1, run_id="test", db=object())

    assert row["source_pids"] == ["1111.1111", "2222.2222"]
    assert row["context_pids"] == ["1111.1111"]
    assert row["source_recall"] == 1.0
    assert row["citation_support_rate"] == 1.0


def test_run_cases_with_checkpoint_skips_existing_rows_and_persists_each_new_row() -> None:
    questions = [
        {"qid": "q1"},
        {"qid": "q2"},
        {"qid": "q3"},
    ]
    calls: list[tuple[str, int, int]] = []
    snapshots: list[list[str]] = []

    def run_case(item: dict, idx: int, total: int) -> dict:
        calls.append((item["qid"], idx, total))
        return {"qid": item["qid"], "value": idx}

    def persist(rows: list[dict]) -> None:
        snapshots.append([row["qid"] for row in rows])

    rows = run_cases_with_checkpoint(
        questions,
        existing_rows=[{"qid": "q1", "value": 1}],
        run_case=run_case,
        persist=persist,
    )

    assert calls == [("q2", 2, 3), ("q3", 3, 3)]
    assert snapshots == [["q1", "q2"], ["q1", "q2", "q3"]]
    assert [row["qid"] for row in rows] == ["q1", "q2", "q3"]
