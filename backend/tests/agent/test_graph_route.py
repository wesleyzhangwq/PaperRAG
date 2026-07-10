"""Deterministic Graph RAG routing policy tests."""
from unittest.mock import patch

from app.agent.nodes.route import route_node
from app.agent.state import StepSpec


@patch("app.agent.nodes.route.get_settings")
def test_route_injects_graph_after_final_local_retrieval_for_comparison(mock_settings) -> None:
    mock_settings.return_value.graph_rag_enabled = True
    mock_settings.return_value.retrieval_k = 8
    mock_settings.return_value.tavily_api_key = None
    mock_settings.return_value.arxiv_max_results = 5
    state = {
        "intent": {"type": "comparison", "complexity": "high"},
        "plan": [
            StepSpec(action="query_rewrite", params={"original_query": "compare A B"}, reason="split"),
            StepSpec(action="retrieve_local", params={"query": "A", "top_k": 8}, reason="seed A"),
            StepSpec(action="retrieve_local", params={"query": "B", "top_k": 8}, reason="seed B"),
        ],
        "step_traces": [],
    }

    result = route_node(state, query="compare A and B")

    assert [step["action"] for step in result["plan"]] == [
        "query_rewrite", "retrieve_local", "retrieve_local", "retrieve_graph"
    ]
    assert result["plan"][-1]["params"] == {"query": "compare A and B", "top_k": 8}
    assert "injected_graph_retrieval" in result["route_decision"]["adjustments"]
    assert "retrieve_graph" in result["route_decision"]["sources"]


@patch("app.agent.nodes.route.get_settings")
def test_route_removes_graph_action_when_feature_is_disabled(mock_settings) -> None:
    mock_settings.return_value.graph_rag_enabled = False
    mock_settings.return_value.retrieval_k = 8
    mock_settings.return_value.tavily_api_key = None
    mock_settings.return_value.arxiv_max_results = 5
    state = {
        "intent": {"type": "trend_synthesis", "complexity": "high"},
        "plan": [StepSpec(action="retrieve_graph", params={"query": "trend"}, reason="graph")],
        "step_traces": [],
    }

    result = route_node(state, query="trend")

    assert "retrieve_graph" not in [step["action"] for step in result["plan"]]
    assert result["plan"][0]["action"] == "retrieve_local"
    assert "dropped_graph_retrieval" in result["route_decision"]["adjustments"]
