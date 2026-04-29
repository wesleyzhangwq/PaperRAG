"""Qdrant vector store helpers."""
from __future__ import annotations

import time
import uuid
from functools import lru_cache
from threading import Lock
from typing import Optional

import requests
from cachetools import LRUCache
from langchain_core.documents import Document
from qdrant_client import QdrantClient, models

from app.core.config import get_settings

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class AlibabaEmbeddingClient:
    """Minimal client for Alibaba text-embedding-v4 API."""

    def __init__(
        self,
        model: str,
        api_base: str,
        api_key: str,
        *,
        query_cache_max: int = 0,
    ) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        # Official DashScope embeddings endpoint.
        # Accept both base styles:
        # - https://dashscope.aliyuncs.com
        # - https://dashscope.aliyuncs.com/compatible-mode/v1
        root = self.api_base
        if "/compatible-mode/v1" in root:
            root = root.split("/compatible-mode/v1")[0]
        self.endpoint = f"{root}/api/v1/services/embeddings/text-embedding/text-embedding"
        # DashScope text-embedding-v4 accepts at most 10 inputs per request.
        self.batch_size = 10
        self._query_cache_max = max(0, query_cache_max)
        if self._query_cache_max > 0:
            self._query_cache: Optional[LRUCache[str, list[float]]] = LRUCache(
                maxsize=self._query_cache_max
            )
            self._query_lock = Lock()
        else:
            self._query_cache = None
            self._query_lock = None

    def _post_embed_chunk(self, headers: dict, chunk: list[str]) -> requests.Response:
        cfg = get_settings()
        payload = {"model": self.model, "input": {"texts": chunk}}
        attempts = max(1, cfg.http_retry_max_attempts)
        resp: Optional[requests.Response] = None
        for attempt in range(attempts):
            resp = requests.post(
                self.endpoint, headers=headers, json=payload, timeout=60
            )
            if resp.status_code < 400:
                return resp
            if (
                resp.status_code not in _RETRYABLE_STATUS
                or attempt >= attempts - 1
            ):
                return resp
            time.sleep(cfg.http_retry_backoff_base_sec * (2**attempt))
        assert resp is not None
        return resp

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            resp = self._post_embed_chunk(headers, chunk)
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"Embedding API error {resp.status_code}: {resp.text[:500]}"
                )
            data = resp.json()

            # Expected DashScope shape
            if isinstance(data, dict):
                output = data.get("output") or {}
                embeddings = output.get("embeddings") or []
                if embeddings:
                    vectors = []
                    for item in embeddings:
                        vec = item.get("embedding")
                        if isinstance(vec, list):
                            vectors.append(vec)
                    if vectors:
                        all_vectors.extend(vectors)
                        continue
            raise RuntimeError(f"Unexpected embedding response shape: {data}")

        return all_vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed_batch(texts)

    def embed_query(self, text: str) -> list[float]:
        if self._query_cache is not None and self._query_lock is not None:
            with self._query_lock:
                if text in self._query_cache:
                    return list(self._query_cache[text])
        vectors = self._embed_batch([text])
        if not vectors:
            raise RuntimeError("Embedding API returned empty vector for query.")
        vec = vectors[0]
        if self._query_cache is not None and self._query_lock is not None:
            with self._query_lock:
                self._query_cache[text] = list(vec)
        return vec


@lru_cache
def get_embeddings() -> AlibabaEmbeddingClient:
    s = get_settings()
    if not s.embedding_api_key:
        raise RuntimeError("Missing EMBEDDING_API_KEY in .env")
    qmax = (
        s.cache_embedding_max_entries if s.cache_embedding_enabled else 0
    )
    return AlibabaEmbeddingClient(
        model=s.embedding_model,
        api_base=s.embedding_api_base,
        api_key=s.embedding_api_key,
        query_cache_max=qmax,
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


def _to_point_id(raw_id: str) -> str:
    """Normalize arbitrary chunk ids into Qdrant-compatible UUID point ids."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw_id))


class QdrantVectorStore:
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embedding: AlibabaEmbeddingClient,
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
        for source_id, text, metadata, vector in zip(ids, texts, metadatas, vectors):
            point_id = _to_point_id(source_id)
            payload = {"text": text, "metadata": metadata, "chunk_id": source_id}
            points.append(models.PointStruct(id=point_id, vector=vector, payload=payload))

        self._client.upsert(collection_name=self._collection_name, points=points, wait=True)

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        point_ids = [_to_point_id(i) for i in ids]
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=models.PointIdsList(points=point_ids),
            wait=True,
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        fetch_limit: Optional[int] = None,
    ) -> list[tuple[Document, float]]:
        query_vector = self._embedding.embed_query(query)
        qdrant_filter = _to_qdrant_filter(filter)
        limit = int(fetch_limit) if fetch_limit is not None else int(k)
        if limit < 1:
            limit = 1
        points = []
        if hasattr(self._client, "query_points"):
            resp = self._client.query_points(
                collection_name=self._collection_name,
                query=query_vector,
                limit=limit,
                query_filter=qdrant_filter,
                with_payload=True,
            )
            points = resp.points
        else:
            points = self._client.search(
                collection_name=self._collection_name,
                query_vector=query_vector,
                limit=limit,
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
    s = get_settings()
    client = QdrantClient(
        url=s.qdrant_url,
        api_key=s.qdrant_api_key or None,
        check_compatibility=False,
    )
    return QdrantVectorStore(
        client=client,
        collection_name=s.qdrant_collection,
        embedding=get_embeddings(),
    )
