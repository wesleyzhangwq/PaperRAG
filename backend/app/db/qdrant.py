"""Qdrant vector store helpers."""
from __future__ import annotations

import uuid
from functools import lru_cache
from typing import Optional

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient, models

from app.core.config import get_settings

settings = get_settings()


@lru_cache
def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )


def _to_qdrant_filter(metadata_filter: Optional[dict]) -> Optional[models.Filter]:
    if not metadata_filter:
        return None

    def _parse_condition(item: dict) -> list[models.FieldCondition]:
        if len(item) != 1:
            return []
        field, op_obj = next(iter(item.items()))
        if not isinstance(op_obj, dict) or len(op_obj) != 1:
            return []
        op, value = next(iter(op_obj.items()))
        key = f"metadata.{field}"
        if op == "$eq":
            return [models.FieldCondition(key=key, match=models.MatchValue(value=value))]
        if op == "$in" and isinstance(value, list):
            return [
                models.FieldCondition(key=key, match=models.MatchAny(any=value)),
            ]
        if op == "$gte":
            return [models.FieldCondition(key=key, range=models.Range(gte=value))]
        if op == "$lte":
            return [models.FieldCondition(key=key, range=models.Range(lte=value))]
        return []

    conditions: list[models.FieldCondition] = []
    if "$and" in metadata_filter and isinstance(metadata_filter["$and"], list):
        for cond in metadata_filter["$and"]:
            if isinstance(cond, dict):
                conditions.extend(_parse_condition(cond))
    elif isinstance(metadata_filter, dict):
        conditions.extend(_parse_condition(metadata_filter))

    if not conditions:
        return None
    return models.Filter(must=conditions)


class QdrantVectorStore:
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embedding: OllamaEmbeddings,
    ) -> None:
        self._client = client
        self._collection_name = collection_name
        self._embedding = embedding
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = self._client.get_collections().collections
        if any(c.name == self._collection_name for c in collections):
            return
        vector_size = len(self._embedding.embed_query("dimension probe"))
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def add_texts(
        self,
        texts: list[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
    ) -> None:
        metadatas = metadatas or [{} for _ in texts]
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        vectors = self._embedding.embed_documents(texts)
        points: list[models.PointStruct] = []
        for point_id, text, metadata, vector in zip(ids, texts, metadatas, vectors):
            payload = {"text": text, "metadata": metadata}
            points.append(models.PointStruct(id=point_id, vector=vector, payload=payload))

        self._client.upsert(collection_name=self._collection_name, points=points, wait=True)

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=models.PointIdsList(points=ids),
            wait=True,
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> list[tuple[Document, float]]:
        query_vector = self._embedding.embed_query(query)
        qdrant_filter = _to_qdrant_filter(filter)
        points = []
        if hasattr(self._client, "query_points"):
            resp = self._client.query_points(
                collection_name=self._collection_name,
                query=query_vector,
                limit=k,
                query_filter=qdrant_filter,
                with_payload=True,
            )
            points = resp.points
        else:
            points = self._client.search(
                collection_name=self._collection_name,
                query_vector=query_vector,
                limit=k,
                query_filter=qdrant_filter,
                with_payload=True,
            )
        results: list[tuple[Document, float]] = []
        for p in points:
            payload = p.payload or {}
            doc = Document(
                page_content=payload.get("text", ""),
                metadata=payload.get("metadata", {}),
            )
            results.append((doc, float(p.score)))
        return results


@lru_cache
def get_qdrant_vector_store() -> QdrantVectorStore:
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        check_compatibility=False,
    )
    return QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=get_embeddings(),
    )
