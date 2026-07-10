from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_reembed_pending_points_preserves_payload_and_skips_existing_ids() -> None:
    try:
        from app.services.qdrant_reindex import reembed_pending_points
    except ImportError as exc:
        pytest.fail(f"candidate collection reindexer is not implemented: {exc}")

    points = [
        SimpleNamespace(id="done", payload={"text": "already indexed", "metadata": {"paper_id": "A"}}),
        SimpleNamespace(id="p2", payload={"text": "second chunk", "metadata": {"paper_id": "B"}}),
        SimpleNamespace(id="p3", payload={"text": "third chunk", "metadata": {"paper_id": "C"}}),
    ]
    embedded: list[list[str]] = []
    upserted: list[list] = []

    def embed_documents(texts: list[str]) -> list[list[float]]:
        embedded.append(texts)
        return [[0.1, 0.2], [0.3, 0.4]]

    stats = reembed_pending_points(
        points,
        existing_ids={"done"},
        embed_documents=embed_documents,
        upsert=upserted.append,
        batch_size=2,
    )

    assert embedded == [["second chunk", "third chunk"]]
    assert stats == {
        "source_points": 3,
        "reindexed_points": 2,
        "skipped_existing": 1,
    }
    assert [point.id for point in upserted[0]] == ["p2", "p3"]
    assert upserted[0][0].payload == {
        "text": "second chunk",
        "metadata": {"paper_id": "B"},
    }


def test_reembed_pending_points_rejects_partial_embedding_batches() -> None:
    try:
        from app.services.qdrant_reindex import reembed_pending_points
    except ImportError as exc:
        pytest.fail(f"candidate collection reindexer is not implemented: {exc}")

    points = [
        SimpleNamespace(id="p2", payload={"text": "second chunk", "metadata": {}}),
        SimpleNamespace(id="p3", payload={"text": "third chunk", "metadata": {}}),
    ]

    with pytest.raises(RuntimeError, match="expected 2 vectors, got 1"):
        reembed_pending_points(
            points,
            existing_ids=set(),
            embed_documents=lambda texts: [[0.1, 0.2]],
            upsert=lambda points: None,
            batch_size=2,
        )
