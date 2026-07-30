from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.db.qdrant import QdrantVectorStore


def test_vector_store_uses_existing_alias_without_creating_collection() -> None:
    client = MagicMock()
    client.get_collections.return_value = SimpleNamespace(collections=[])
    client.get_aliases.return_value = SimpleNamespace(
        aliases=[
            SimpleNamespace(
                alias_name="paperrag-active",
                collection_name="paperrag-bge-m3-20260710",
            )
        ]
    )
    embedding = MagicMock()

    QdrantVectorStore(
        client=client,
        collection_name="paperrag-active",
        embedding=embedding,
    )

    client.create_collection.assert_not_called()
    embedding.embed_query.assert_not_called()


def test_add_texts_rejects_partial_embedding_response() -> None:
    client = MagicMock()
    client.get_collections.return_value = SimpleNamespace(
        collections=[SimpleNamespace(name="paperrag")]
    )
    embedding = MagicMock()
    embedding.embed_documents.return_value = [[0.1, 0.2]]
    store = QdrantVectorStore(
        client=client,
        collection_name="paperrag",
        embedding=embedding,
    )

    with pytest.raises(RuntimeError, match="Embedding count mismatch"):
        store.add_texts(
            texts=["first", "second"],
            metadatas=[{}, {}],
            ids=["first", "second"],
        )

    client.upsert.assert_not_called()
