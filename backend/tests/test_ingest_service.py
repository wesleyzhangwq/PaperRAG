from unittest.mock import MagicMock, patch

import pytest

from app.models.paper import Chunk, Paper
from app.services.ingest import _ingest_one
from app.utils.document import DocumentBlock, ParsedDocument


def _mock_db(existing: Paper | None, old_chunks: list[Chunk]):
    db = MagicMock()
    paper_query = MagicMock()
    paper_query.filter.return_value = paper_query
    paper_query.one_or_none.return_value = existing
    chunk_query = MagicMock()
    chunk_query.filter.return_value = chunk_query
    chunk_query.all.return_value = old_chunks

    def query(model):
        return paper_query if model is Paper else chunk_query

    db.query.side_effect = query
    return db, chunk_query


def _parsed() -> ParsedDocument:
    return ParsedDocument(
        blocks=[
            DocumentBlock(
                text=(
                    "Dense retrieval produces candidates. BM25 fusion retains "
                    "lexical evidence and improves the final evidence set."
                ),
                modality="table",
                source_locator={"sheet": "Eval"},
            )
        ],
        title="Evaluation",
        metadata={"modalities": ["table"]},
    )


def test_ingest_persists_provenance_after_vector_upsert(tmp_path):
    source = tmp_path / "eval.txt"
    source.write_text("placeholder", encoding="utf-8")
    db, chunk_query = _mock_db(None, [])
    vector_store = MagicMock()

    with (
        patch("app.services.ingest.parse_document", return_value=_parsed()),
        patch("app.services.ingest.get_qdrant_vector_store", return_value=vector_store),
    ):
        paper_id, status = _ingest_one(
            db,
            {
                "paper_id": "local-test",
                "title": "Evaluation",
                "year": 2026,
                "primary_category": "local",
                "source_path": str(source),
                "source_kind": "upload",
                "media_type": "text/plain",
            },
        )

    assert (paper_id, status) == ("local-test", "ok")
    metadata = vector_store.add_texts.call_args.kwargs["metadatas"][0]
    assert metadata["modality"] == "table"
    assert metadata["source_locator"] == {"sheet": "Eval"}
    persisted_chunk = db.add_all.call_args.args[0][0]
    assert persisted_chunk.modality == "table"
    assert persisted_chunk.source_locator == {"sheet": "Eval"}
    chunk_query.delete.assert_called_once()


def test_ingest_does_not_replace_mysql_chunks_when_stale_vector_delete_fails(tmp_path):
    source = tmp_path / "eval.txt"
    source.write_text("placeholder", encoding="utf-8")
    existing = Paper(
        paper_id="local-test",
        title="Existing",
        authors=[],
        year=2026,
        primary_category="local",
        ingest_status="ok",
        num_chunks=2,
    )
    old_chunks = [
        Chunk(
            chunk_id="local-test::0",
            paper_id="local-test",
            chunk_index=0,
            chunk_text="old first",
        ),
        Chunk(
            chunk_id="local-test::1",
            paper_id="local-test",
            chunk_index=1,
            chunk_text="old second",
        ),
    ]
    db, chunk_query = _mock_db(existing, old_chunks)
    vector_store = MagicMock()
    vector_store.delete.side_effect = RuntimeError("qdrant unavailable")

    with (
        patch("app.services.ingest.parse_document", return_value=_parsed()),
        patch("app.services.ingest.get_qdrant_vector_store", return_value=vector_store),
        pytest.raises(RuntimeError, match="qdrant unavailable"),
    ):
        _ingest_one(
            db,
            {
                "paper_id": "local-test",
                "title": "Replacement",
                "year": 2026,
                "primary_category": "local",
                "source_path": str(source),
            },
            force=True,
        )

    chunk_query.delete.assert_not_called()
    assert existing.title == "Existing"
    assert existing.ingest_status == "ok"
