"""Re-embed Qdrant payloads into a separate candidate collection."""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from qdrant_client import models


def reembed_pending_points(
    points: Iterable[Any],
    *,
    existing_ids: set[str],
    embed_documents: Callable[[list[str]], list[list[float]]],
    upsert: Callable[[list[models.PointStruct]], Any],
    batch_size: int,
) -> dict[str, int]:
    """Embed source payload text while preserving point IDs and metadata."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    source = list(points)
    pending = [point for point in source if str(point.id) not in existing_ids]
    reindexed = 0

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        payloads = [dict(point.payload or {}) for point in batch]
        texts = [str(payload.get("text") or "").strip() for payload in payloads]
        if any(not text for text in texts):
            raise RuntimeError("Source point is missing payload.text")

        vectors = embed_documents(texts)
        if len(vectors) != len(batch):
            raise RuntimeError(
                f"Embedding batch mismatch: expected {len(batch)} vectors, got {len(vectors)}"
            )
        upsert(
            [
                models.PointStruct(id=point.id, vector=vector, payload=payload)
                for point, vector, payload in zip(batch, vectors, payloads)
            ]
        )
        reindexed += len(batch)

    return {
        "source_points": len(source),
        "reindexed_points": reindexed,
        "skipped_existing": len(source) - len(pending),
    }
