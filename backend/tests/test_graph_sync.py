from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.db.neo4j import GraphUnavailable
from app.services.semantic_scholar import CitationSnapshot, RemotePaper


def _paper(**overrides):
    defaults = {
        "paper_id": "2401.00001",
        "doi": None,
        "title": "Local",
        "year": 2024,
        "authors": ["Ada"],
        "categories": ["cs.CL"],
        "ingest_status": "ok",
        "num_chunks": 3,
        "graph_sync_status": "pending",
        "graph_sync_error": None,
        "graph_synced_at": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_sync_paper_projects_local_and_external_nodes_then_marks_ok() -> None:
    from app.services.graph_sync import sync_paper

    paper = _paper()
    with patch("app.services.graph_sync.fetch_citation_snapshot") as mock_fetch, patch(
        "app.services.graph_sync.get_neo4j_repository"
    ) as mock_repo:
        mock_fetch.return_value = CitationSnapshot(
            source=RemotePaper("s2-a", "2401.00001", None, "Local", 2024, (("author-a", "Ada"),)),
            references=(RemotePaper("s2-b", None, None, "Bridge", 2020, ()),),
            citations=(),
        )

        status = sync_paper(MagicMock(), paper, local_papers={paper.paper_id: paper})

    assert status == "ok"
    assert paper.graph_sync_status == "ok"
    assert paper.graph_sync_error is None
    kwargs = mock_repo.return_value.replace_source_projection.call_args.kwargs
    assert kwargs["source_paper_id"] == "2401.00001"
    assert {item["graph_key"] for item in kwargs["papers"]} == {"arxiv:2401.00001", "s2:s2-b"}
    assert kwargs["citation_edges"] == [{"source_key": "arxiv:2401.00001", "target_key": "s2:s2-b"}]


def test_sync_paper_marks_failure_without_raising() -> None:
    from app.services.graph_sync import sync_paper

    paper = _paper()
    with patch(
        "app.services.graph_sync.fetch_citation_snapshot",
        side_effect=GraphUnavailable("offline"),
    ):
        status = sync_paper(MagicMock(), paper, local_papers={paper.paper_id: paper})

    assert status == "failed"
    assert paper.graph_sync_status == "failed"
    assert "offline" in paper.graph_sync_error
