"""Test synthesis, groundedness, and citation_gate nodes."""
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from app.agent.state import AgentState
from app.agent.nodes.synthesis import synthesis_node
from app.agent.nodes.groundedness import groundedness_node
from app.agent.nodes.citation_gate import citation_gate_node


def _base_state(**overrides) -> AgentState:
    defaults = {
        "messages": [],
        "intent": {"type": "simple", "entities": [], "complexity": "low"},
        "plan": [],
        "plan_step_index": 0,
        "retrieval_context": [
            Document(page_content="Attention allows global dependencies", metadata={"paper_id": "1706.03762", "title": "Attention Is All You Need", "page_num": 3}),
        ],
        "step_traces": [],
        "reflection_count": 0,
        "final_answer": None,
        "reflection_result": None,
        "sources": None,
    }
    defaults.update(overrides)
    return defaults


def test_synthesis_generates_answer():
    mock_llm = MagicMock()
    # synthesis uses llm.stream() which yields chunk objects
    mock_llm.stream.return_value = [MagicMock(content="Attention 机制允许模型捕获全局依赖 [arxiv:1706.03762]")]
    state = _base_state()
    with patch("app.agent.nodes.synthesis._get_llm", return_value=mock_llm):
        result = synthesis_node(state, query="what is attention")

    assert "1706.03762" in result["final_answer"]
    assert len(result["step_traces"]) == 1


def test_synthesis_emits_answer_start_before_tokens():
    mock_llm = MagicMock()
    mock_llm.stream.return_value = [MagicMock(content="Final answer [arxiv:1706.03762]")]
    state = _base_state(reflection_count=1)

    with patch("app.agent.nodes.synthesis._get_llm", return_value=mock_llm), \
         patch("app.agent.nodes.synthesis.emit") as mock_emit:
        synthesis_node(state, query="what is attention")

    assert mock_emit.call_args_list[0].args == (
        "answer_start",
        {"attempt": 2, "reset": True},
    )
    assert mock_emit.call_args_list[1].args == (
        "token",
        {"t": "Final answer [arxiv:1706.03762]"},
    )


def test_groundedness_passes():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='{"passed": true, "citation_ok": true, "completeness_ok": true, "logic_ok": true, "issues": [], "fix_strategy": null}'
    )
    state = _base_state(final_answer="Answer with [arxiv:1706.03762]")
    with patch("app.agent.nodes.groundedness._get_llm", return_value=mock_llm):
        result = groundedness_node(state, query="what is attention")

    assert result["reflection_result"]["passed"] is True


def test_groundedness_fails_triggers_re_retrieve():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='{"passed": false, "citation_ok": true, "completeness_ok": false, "logic_ok": true, "issues": ["Missing multi-head details"], "fix_strategy": "re_retrieve"}'
    )
    state = _base_state(final_answer="Incomplete answer", reflection_count=0)
    with patch("app.agent.nodes.groundedness._get_llm", return_value=mock_llm):
        result = groundedness_node(state, query="explain multi-head attention")

    assert result["reflection_result"]["passed"] is False
    assert result["reflection_result"]["fix_strategy"] == "re_retrieve"
    assert result["reflection_count"] == 1


def test_groundedness_parse_failure_triggers_retry():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="not valid json")
    state = _base_state(final_answer="Answer with [arxiv:1706.03762]", reflection_count=0)

    with patch("app.agent.nodes.groundedness._get_llm", return_value=mock_llm):
        result = groundedness_node(state, query="what is attention")

    assert result["reflection_result"]["passed"] is False
    assert result["reflection_result"]["fix_strategy"] == "re_generate"
    assert result["reflection_count"] == 1


def test_groundedness_deterministic_precheck_catches_fabricated_citation():
    """A citation id absent from the retrieval context must fail WITHOUT
    any LLM call — the deterministic precheck catches it."""
    mock_llm = MagicMock()
    state = _base_state(
        final_answer="Fabricated claim [arxiv:9999.99999]",
        reflection_count=0,
    )
    with patch("app.agent.nodes.groundedness._get_llm", return_value=mock_llm):
        result = groundedness_node(state, query="what is attention")

    assert result["reflection_result"]["passed"] is False
    assert result["reflection_result"]["citation_ok"] is False
    assert result["reflection_result"]["fix_strategy"] == "re_generate"
    assert result["reflection_count"] == 1
    mock_llm.invoke.assert_not_called()


def test_citation_gate_extracts_citations():
    state = _base_state(final_answer="This uses attention [arxiv:1706.03762] and BERT [arxiv:1810.04805]")
    mock_db = MagicMock()
    mock_paper = MagicMock()
    mock_paper.title = "Attention Paper"
    mock_paper.authors = ["Vaswani"]
    mock_paper.year = 2017
    mock_paper.primary_category = "cs.CL"
    mock_paper.doi = None
    mock_db.query.return_value.filter.return_value.one_or_none.return_value = mock_paper

    result = citation_gate_node(state, db=mock_db)

    assert len(result["sources"]) == 2
    assert result["removed_citations"] == []


def test_citation_gate_strips_unresolvable_citation():
    """A citation that is neither in the DB nor in the retrieval context is a
    hallucination — the marker must be stripped from the answer."""
    state = _base_state(
        final_answer="Real [arxiv:1706.03762] and fake [arxiv:9999.99999]."
    )
    mock_db = MagicMock()

    def _query_filter(pid_filter):
        return None

    mock_paper = MagicMock()
    mock_paper.title = "Attention Paper"
    mock_paper.authors = ["Vaswani"]
    mock_paper.year = 2017
    mock_paper.primary_category = "cs.CL"
    mock_paper.doi = None

    # First lookup (1706.03762) hits, second (9999.99999) misses.
    mock_db.query.return_value.filter.return_value.one_or_none.side_effect = [mock_paper, None]

    result = citation_gate_node(state, db=mock_db)

    assert [s.paper_id for s in result["sources"]] == ["1706.03762"]
    assert result["removed_citations"] == ["9999.99999"]
    assert "[arxiv:9999.99999]" not in result["final_answer"]
    assert "[arxiv:1706.03762]" in result["final_answer"]


def test_citation_gate_keeps_context_only_citation_with_minimal_source():
    """A paper present in retrieval context (e.g. fresh arXiv API hit) but not
    yet ingested into MySQL keeps its citation with a minimal source."""
    state = _base_state(
        final_answer="From context [arxiv:1706.03762].",
    )
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.one_or_none.return_value = None

    result = citation_gate_node(state, db=mock_db)

    assert len(result["sources"]) == 1
    assert result["sources"][0].paper_id == "1706.03762"
    assert result["sources"][0].title == "Attention Is All You Need"
    assert result["removed_citations"] == []
