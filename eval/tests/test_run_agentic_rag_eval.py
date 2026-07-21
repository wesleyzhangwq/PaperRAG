import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

from app.schemas.chat import ChatResponse
from eval.run_agentic_rag_eval import _build_resume_contract
from eval.run_agentic_rag_eval import _settings_manifest
from eval.run_agentic_rag_eval import _validate_resume_contract
from eval.run_agentic_rag_eval import run_agentic_case
from eval.run_agentic_rag_eval import run_cases_with_checkpoint


def _resume_contract(tmp_path: Path) -> tuple[dict, Path, Path, list[dict]]:
    dataset = tmp_path / "questions.jsonl"
    questions = [{"qid": "q1"}, {"qid": "q2"}]
    dataset.write_text(
        "".join(json.dumps(question) + "\n" for question in questions),
        encoding="utf-8",
    )
    catalog = tmp_path / "pricing.json"
    catalog.write_text(
        json.dumps(
            {
                "catalog_version": "catalog-v1",
                "provider": "minimax",
                "billing_origin": "minimax_paygo",
                "currency": "USD",
            }
        ),
        encoding="utf-8",
    )
    contract = _build_resume_contract(
        dataset_path=dataset,
        pricing_catalog_path=catalog,
        questions=questions,
        selection={"source_count": 2, "sample_size": 2, "per_type": None, "limit": None},
        traditional={
            "retriever": "service",
            "context_k": 5,
            "retrieval_top_k": 20,
            "context_strategy": "paper_dedup",
            "mmr_lambda": 0.65,
        },
        agentic={
            "entrypoint": "app.agent.graph.run_agent_eval_sync",
            "retrieval_actions": ["retrieve_local"],
            "local_only": True,
        },
        settings={
            "llm_provider": "minimax",
            "llm_model": "MiniMax-M2.7",
            "embedding_model": "BAAI/bge-m3",
            "retrieval_k": 20,
        },
        execution={
            "concurrency": 1,
            "warmup": False,
            "request_timeout_s": 120,
            "external_api_allowed": False,
        },
        pricing={
            "catalog_version": "catalog-v1",
            "catalog_provider": "minimax",
            "catalog_billing_origin": "minimax_paygo",
            "currency": "USD",
            "billing_origin": "minimax_paygo",
            "cost_scope": "llm_only",
        },
        git_commit="abc123",
    )
    return contract, dataset, catalog, questions


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


def test_run_cases_with_checkpoint_rejects_rows_outside_frozen_selection() -> None:
    with pytest.raises(ValueError, match="not in the frozen selection"):
        run_cases_with_checkpoint(
            [{"qid": "q1"}],
            existing_rows=[{"qid": "q-other"}],
            run_case=lambda item, idx, total: {"qid": item["qid"]},
            persist=lambda _rows: None,
        )


def test_resume_contract_records_dataset_hash_and_exact_qid_order(tmp_path: Path) -> None:
    contract, dataset, catalog, _questions = _resume_contract(tmp_path)

    assert contract["dataset"]["path"] == str(dataset.resolve())
    assert contract["dataset"]["selected_qids"] == ["q1", "q2"]
    assert len(contract["dataset"]["sha256"]) == 64
    assert contract["pricing"]["catalog_path"] == str(catalog.resolve())
    assert len(contract["pricing"]["catalog_sha256"]) == 64
    assert len(contract["implementation"]["evaluation_source_sha256"]) == 64
    _validate_resume_contract({"resume_contract": contract}, deepcopy(contract))


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("dataset", "sha256"), "changed-dataset"),
        (("dataset", "selected_qids"), ["q2", "q1"]),
        (("selection", "sample_size"), 1),
        (("traditional", "context_k"), 6),
        (("traditional", "retrieval_top_k"), 12),
        (("traditional", "context_strategy"), "mmr_dedup"),
        (("traditional", "mmr_lambda"), 0.5),
        (("agentic", "local_only"), False),
        (("settings", "llm_provider"), "openai"),
        (("settings", "llm_model"), "different-model"),
        (("settings", "embedding_model"), "different-embedding"),
        (("pricing", "billing_origin"), "unknown"),
        (("pricing", "catalog_version"), "catalog-v2"),
        (("pricing", "catalog_sha256"), "changed-catalog"),
        (("pricing", "currency"), "CNY"),
        (("execution", "request_timeout_s"), 60),
        (("implementation", "git_commit"), "def456"),
        (("implementation", "evaluation_source_sha256"), "changed-source"),
    ],
)
def test_resume_contract_rejects_any_immutable_input_change(
    tmp_path: Path, path: tuple[str, str], replacement: object
) -> None:
    stored, _dataset, _catalog, _questions = _resume_contract(tmp_path)
    current = deepcopy(stored)
    current[path[0]][path[1]] = replacement

    with pytest.raises(ValueError, match="immutable evaluation inputs changed"):
        _validate_resume_contract({"resume_contract": stored}, current)


def test_resume_contract_rejects_legacy_manifest_without_contract(tmp_path: Path) -> None:
    contract, _dataset, _catalog, _questions = _resume_contract(tmp_path)

    with pytest.raises(ValueError, match="has no immutable resume_contract"):
        _validate_resume_contract({"dataset": "legacy.jsonl"}, contract)


def test_settings_manifest_freezes_connections_without_endpoint_or_secret_values() -> None:
    settings = _settings_manifest()

    assert not any("api_key" in key or "password" in key for key in settings)
    assert not any("://" in str(value) for value in settings.values())
    for key in (
        "llm_connection_sha256",
        "embedding_connection_sha256",
        "mysql_connection_sha256",
        "qdrant_connection_sha256",
    ):
        assert len(settings[key]) == 64
