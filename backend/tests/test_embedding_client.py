from unittest.mock import MagicMock, patch

from app.db.qdrant import EmbeddingClient


def test_openai_compatible_embedding_endpoint_and_response_shape():
    client = EmbeddingClient(
        model="BAAI/bge-m3",
        api_base="https://api.siliconflow.cn/v1",
        api_key="test-key",
    )
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "data": [
            {"embedding": [0.1, 0.2], "index": 0},
            {"embedding": [0.3, 0.4], "index": 1},
        ]
    }

    with patch("app.db.qdrant.requests.post", return_value=response) as post:
        vectors = client.embed_documents(["a", "b"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    post.assert_called_once()
    assert post.call_args.args[0] == "https://api.siliconflow.cn/v1/embeddings"
    assert post.call_args.kwargs["json"] == {"model": "BAAI/bge-m3", "input": ["a", "b"]}


def test_dashscope_embedding_endpoint_and_response_shape_still_works():
    client = EmbeddingClient(
        model="text-embedding-v4",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="test-key",
    )
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "output": {
            "embeddings": [
                {"embedding": [0.1, 0.2]},
                {"embedding": [0.3, 0.4]},
            ]
        }
    }

    with patch("app.db.qdrant.requests.post", return_value=response) as post:
        vectors = client.embed_documents(["a", "b"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    post.assert_called_once()
    assert post.call_args.args[0] == (
        "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
    )
    assert post.call_args.kwargs["json"] == {
        "model": "text-embedding-v4",
        "input": {"texts": ["a", "b"]},
    }
