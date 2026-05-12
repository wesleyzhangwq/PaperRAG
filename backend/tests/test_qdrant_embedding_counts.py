"""Regression: embedding API must return one vector per input text."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from app.db.qdrant import AlibabaEmbeddingClient


class TestEmbeddingBatchLengths(unittest.TestCase):
    def test_fewer_vectors_than_inputs_raises(self) -> None:
        client = AlibabaEmbeddingClient(
            model="m",
            api_base="https://example.com",
            api_key="k",
            query_cache_max=0,
        )
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "output": {"embeddings": [{"embedding": [0.0, 1.0]}]},
        }
        with patch.object(client, "_post_embed_chunk", return_value=mock_resp):
            with self.assertRaises(RuntimeError) as ctx:
                client._embed_batch(["a", "b"])
        self.assertIn("2 inputs", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
