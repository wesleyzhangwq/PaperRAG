"""Contract tests for embedding batching (prevents silent chunk drops in Qdrant)."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from app.db.qdrant import AlibabaEmbeddingClient


def _client() -> AlibabaEmbeddingClient:
    return AlibabaEmbeddingClient(
        model="text-embedding-v4",
        api_base="https://dashscope.aliyuncs.com",
        api_key="test-key",
        query_cache_max=0,
    )


class TestEmbedBatchContract(unittest.TestCase):
    def test_partial_batch_vectors_raise(self) -> None:
        """Fewer vectors than input strings must not silently truncate."""
        client = _client()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "output": {
                "embeddings": [
                    {"embedding": [0.0, 1.0]},
                ]
            }
        }
        with patch.object(client, "_post_embed_chunk", return_value=mock_resp):
            with self.assertRaises(RuntimeError) as ctx:
                client.embed_documents(["a", "b"])
        self.assertIn("1 vectors for 2 input", str(ctx.exception))

    def test_matching_batch_succeeds(self) -> None:
        client = _client()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "output": {
                "embeddings": [
                    {"embedding": [0.0, 1.0]},
                    {"embedding": [0.1, 0.9]},
                ]
            }
        }
        with patch.object(client, "_post_embed_chunk", return_value=mock_resp):
            vecs = client.embed_documents(["a", "b"])
        self.assertEqual(len(vecs), 2)
        self.assertEqual(len(vecs[0]), 2)
