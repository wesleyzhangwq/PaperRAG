from types import SimpleNamespace
from unittest.mock import MagicMock

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
