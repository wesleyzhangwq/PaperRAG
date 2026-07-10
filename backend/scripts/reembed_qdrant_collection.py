"""Build a BGE-M3 candidate collection from payloads in an existing collection.

The source collection is never changed. The target is created once and can be
continued only with --resume, making embedding-model migrations auditable.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from qdrant_client import QdrantClient, models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.db.qdrant import get_embeddings  # noqa: E402
from app.services.qdrant_reindex import reembed_pending_points  # noqa: E402


def _scroll_points(
    client: QdrantClient,
    collection: str,
    *,
    page_size: int,
    with_payload: bool,
):
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=page_size,
            offset=offset,
            with_payload=with_payload,
            with_vectors=False,
        )
        yield points
        if offset is None:
            return


def _collection_exists(client: QdrantClient, name: str) -> bool:
    return any(item.name == name for item in client.get_collections().collections)


def _existing_ids(client: QdrantClient, collection: str, page_size: int) -> set[str]:
    return {
        str(point.id)
        for page in _scroll_points(client, collection, page_size=page_size, with_payload=False)
        for point in page
    }


def _manifest_path(raw: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip(".-")
    return Path(__file__).resolve().parents[1] / "data" / f"vector-rebuild-{safe}.json"


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Re-embed a Qdrant collection into a new candidate collection."
    )
    parser.add_argument("--source-collection", default=settings.qdrant_collection)
    parser.add_argument("--target-collection", required=True)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.source_collection == args.target_collection:
        raise ValueError("Target collection must differ from source collection.")
    if args.batch_size < 1 or args.page_size < 1:
        raise ValueError("batch-size and page-size must be at least 1")

    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        check_compatibility=False,
    )
    source_count = client.count(args.source_collection, exact=True).count
    target_exists = _collection_exists(client, args.target_collection)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "source_collection": args.source_collection,
                    "target_collection": args.target_collection,
                    "source_points": source_count,
                    "target_exists": target_exists,
                    "embedding_model": settings.embedding_model,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if target_exists and not args.resume:
        raise ValueError(
            f"Target collection already exists: {args.target_collection}. Use --resume to continue it."
        )

    embeddings = get_embeddings()
    if not target_exists:
        vector_size = len(embeddings.embed_query("dimension probe"))
        client.create_collection(
            collection_name=args.target_collection,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    existing = _existing_ids(client, args.target_collection, args.page_size)
    totals = {"source_points": 0, "reindexed_points": 0, "skipped_existing": 0}
    for page in _scroll_points(
        client,
        args.source_collection,
        page_size=args.page_size,
        with_payload=True,
    ):
        stats = reembed_pending_points(
            page,
            existing_ids=existing,
            embed_documents=embeddings.embed_documents,
            upsert=lambda points: client.upsert(
                collection_name=args.target_collection,
                points=points,
                wait=True,
            ),
            batch_size=args.batch_size,
        )
        for key, value in stats.items():
            totals[key] += value
        existing.update(str(point.id) for point in page)
        print(
            f"reindex progress: {totals['reindexed_points'] + totals['skipped_existing']}/{source_count}",
            file=sys.stderr,
        )

    target_count = client.count(args.target_collection, exact=True).count
    if target_count != source_count:
        raise RuntimeError(
            f"Candidate count mismatch: source={source_count}, target={target_count}"
        )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_collection": args.source_collection,
        "target_collection": args.target_collection,
        "embedding_model": settings.embedding_model,
        "source_points": source_count,
        "target_points": target_count,
        "stats": totals,
    }
    path = _manifest_path(args.target_collection)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**manifest, "manifest": str(path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
