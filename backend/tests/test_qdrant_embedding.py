"""Regression tests for embedding batch invariants (no silent truncation)."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from app.db.qdrant import AlibabaEmbeddingClient


class TestEmbedBatchValidation(unittest.TestCase):
    def _client(self) -> AlibabaEmbeddingClient:
        return AlibabaEmbeddingClient(
            model="m",
            api_base="https://example.com",
            api_key="k",
            query_cache_max=0,
        )

    @patch("app.db.qdrant.requests.post")
    @patch("app.db.qdrant.get_settings")
    def test_raises_when_fewer_vectors_than_inputs(
        self, mock_settings: MagicMock, mock_post: MagicMock
    ) -> None:
        mock_settings.return_value = MagicMock(
            http_retry_max_attempts=1,
            http_retry_backoff_base_sec=0.01,
        )
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "output": {
                    "embeddings": [
                        {"embedding": [0.0, 1.0]},
                        {"embedding": [0.0, 1.0]},
                    ]
                }
            },
        )
        emb = self._client()
        with self.assertRaises(RuntimeError) as ctx:
            emb._embed_batch(["a", "b", "c"])
        self.assertIn("different number of vectors", str(ctx.exception))

    @patch("app.db.qdrant.requests.post")
    @patch("app.db.qdrant.get_settings")
    def test_ok_when_counts_match(self, mock_settings: MagicMock, mock_post: MagicMock) -> None:
        mock_settings.return_value = MagicMock(
            http_retry_max_attempts=1,
            http_retry_backoff_base_sec=0.01,
        )
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "output": {
                    "embeddings": [
                        {"embedding": [0.0]},
                        {"embedding": [1.0]},
                    ]
                }
            },
        )
        emb = self._client()
        out = emb._embed_batch(["x", "y"])
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0], [0.0])
        self.assertEqual(out[1], [1.0])


if __name__ == "__main__":
    unittest.main()
