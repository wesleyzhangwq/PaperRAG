from __future__ import annotations

from types import SimpleNamespace

from eval.scripts.repair_questions import repair_question


class _FakeLlm:
    def invoke(self, prompt: str) -> SimpleNamespace:
        assert "[2603.17450::0]" in prompt
        return SimpleNamespace(content='{"question":"如何缓解多模态序列推荐中的单模态主导问题？","reference_answer":"该方法通过专门的表示约束平衡视觉与文本信号。"}')


def test_repair_question_uses_evidence_and_marks_successful_llm_generation() -> None:
    question = {
        "qid": "m002",
        "query": "fallback",
        "reference_answer": "fallback",
        "expected_paper_ids": ["2603.17450"],
        "type": "method_detail",
        "generation_status": "fallback",
        "evidence_chunk_ids": ["2603.17450::0"],
    }
    papers = {
        "2603.17450": {
            "paper_id": "2603.17450",
            "title": "VLM2Rec",
            "primary_category": "cs.IR",
            "evidence_text": "[2603.17450::0] Method evidence about modality collapse.",
            "evidence_chunks": [
                {"chunk_id": "2603.17450::0", "paper_id": "2603.17450", "text": "Method evidence"}
            ],
        }
    }

    repaired = repair_question(_FakeLlm(), question, papers)

    assert repaired["generation_status"] == "llm"
    assert repaired["evidence_chunk_ids"] == ["2603.17450::0"]
    assert "VLM2Rec" not in repaired["query"]
