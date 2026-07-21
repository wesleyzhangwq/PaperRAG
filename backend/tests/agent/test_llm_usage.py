from contextlib import nullcontext
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, AIMessageChunk

from app.agent.graph import run_agent_sync
from app.observability.llm_usage import (
    collect_llm_usage,
    invoke_with_usage,
    normalize_provider_usage,
    stream_with_usage,
)


class UsageLlm:
    def invoke(self, _prompt):
        return AIMessage(
            content="ok",
            usage_metadata={
                "input_tokens": 11,
                "output_tokens": 3,
                "total_tokens": 14,
                "input_token_details": {"cache_read": 2},
            },
        )

    def stream(self, _prompt):
        yield AIMessageChunk(content="answer")
        yield AIMessageChunk(
            content="",
            usage_metadata={
                "input_tokens": 20,
                "output_tokens": 5,
                "total_tokens": 25,
                "input_token_details": {"cache_read": 4, "cache_creation": 1},
            },
        )


def test_usage_collector_captures_invoke_and_final_stream_usage() -> None:
    llm = UsageLlm()
    with collect_llm_usage() as collector:
        invoke_with_usage(
            llm, "prompt", node="planner", model="MiniMax-M2.7", api_base="https://api.minimax.io/v1"
        )
        list(
            stream_with_usage(
                llm,
                "prompt",
                node="synthesis",
                model="MiniMax-M2.7",
                api_base="https://api.minimax.io/v1",
            )
        )

    records = collector.snapshot()
    assert [record["node"] for record in records] == ["planner", "synthesis"]
    assert records[0]["provider"] == "minimax"
    assert records[0]["cached_read_tokens"] == 2
    assert records[1]["input_tokens"] == 20
    assert records[1]["cached_read_tokens"] == 4
    assert records[1]["cache_write_tokens"] == 1


def test_sync_graph_response_aggregates_all_call_records_per_run() -> None:
    llm = UsageLlm()

    class FakeGraph:
        def invoke(self, _state, config):
            invoke_with_usage(
                llm, "intent", node="intent", model="MiniMax-M2.7", api_base="https://api.minimax.io/v1"
            )
            invoke_with_usage(
                llm, "planner", node="planner", model="MiniMax-M2.7", api_base="https://api.minimax.io/v1"
            )
            list(
                stream_with_usage(
                    llm,
                    "synthesis",
                    node="synthesis",
                    model="MiniMax-M2.7",
                    api_base="https://api.minimax.io/v1",
                )
            )
            return {
                "final_answer": "answer",
                "sources": [],
                "synthesis_context_count": 0,
                "step_traces": [],
                "reflection_result": None,
                "fallback_telemetry": {},
            }

    with (
        patch("app.agent.graph.build_agent_graph", return_value=FakeGraph()),
        patch("app.agent.graph.open_sync_checkpointer", return_value=nullcontext(None)),
    ):
        response = run_agent_sync(MagicMock(), "query", session_id="usage-sync")

    assert [record["node"] for record in response.llm_usage] == ["intent", "planner", "synthesis"]
    assert response.llm_usage[-1]["usage_status"] == "known"


def test_usage_parser_rejects_fractional_and_boolean_token_counts() -> None:
    fractional = normalize_provider_usage(
        {"usage": {"input_tokens": 1.9, "output_tokens": 2.8}}
    )
    boolean = normalize_provider_usage(
        {"usage": {"input_tokens": True, "output_tokens": 2}}
    )

    assert fractional["usage_status"] == "unknown"
    assert fractional["input_tokens"] is None
    assert fractional["output_tokens"] is None
    assert fractional["total_tokens"] is None
    assert boolean["usage_status"] == "unknown"
    assert boolean["input_tokens"] is None


def test_usage_parser_accepts_only_canonical_integer_strings() -> None:
    exact = normalize_provider_usage(
        {
            "usage": {
                "prompt_tokens": "12",
                "completion_tokens": "3",
                "total_tokens": "15",
                "prompt_tokens_details": {
                    "cached_tokens": "2",
                    "cache_creation": "1",
                },
            }
        }
    )
    noncanonical = normalize_provider_usage(
        {"usage": {"input_tokens": " 12", "output_tokens": "3.0"}}
    )

    assert exact["usage_status"] == "known"
    assert exact["input_tokens"] == 12
    assert exact["output_tokens"] == 3
    assert exact["cached_read_tokens"] == 2
    assert exact["cache_write_tokens"] == 1
    assert noncanonical["usage_status"] == "unknown"


def test_usage_parser_rejects_invalid_or_conflicting_aliases_and_total() -> None:
    invalid_primary = normalize_provider_usage(
        {
            "usage": {
                "input_tokens": 1.9,
                "prompt_tokens": 12,
                "output_tokens": 3,
            }
        }
    )
    conflicting_aliases = normalize_provider_usage(
        {
            "usage": {
                "input_tokens": 12,
                "prompt_tokens": 13,
                "output_tokens": 3,
            }
        }
    )
    invalid_total = normalize_provider_usage(
        {"usage": {"input_tokens": 12, "output_tokens": 3, "total_tokens": 14.5}}
    )

    assert invalid_primary["usage_status"] == "unknown"
    assert invalid_primary["input_tokens"] is None
    assert conflicting_aliases["usage_status"] == "unknown"
    assert conflicting_aliases["input_tokens"] is None
    assert invalid_total["usage_status"] == "unknown"
    assert invalid_total["total_tokens"] is None
