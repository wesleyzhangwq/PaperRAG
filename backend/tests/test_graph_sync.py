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
            references=(RemotePaper("s2-b", None, None, "Bridge", 2020, (("author-b", "Bob"),)),),
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
    assert kwargs["authors"] == [
        {"paper_key": "arxiv:2401.00001", "author_id": "author-a", "name": "Ada"}
    ]


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


def test_run_graph_sync_skips_completed_papers_and_commits_each_attempt() -> None:
    from app.services.graph_sync import run_graph_sync

    completed = _paper(paper_id="2401.00001", graph_sync_status="ok")
    pending = _paper(paper_id="2401.00002", graph_sync_status="pending")
    db = MagicMock()
    with patch("app.services.graph_sync.init_db"), patch(
        "app.services.graph_sync.SessionLocal", return_value=db
    ), patch(
        "app.services.graph_sync._successful_papers", return_value=[completed, pending]
    ), patch("app.services.graph_sync.sync_paper", return_value="ok") as mock_sync:
        stats = run_graph_sync()

    mock_sync.assert_called_once_with(
        db,
        pending,
        local_papers={completed.paper_id: completed, pending.paper_id: pending},
    )
    assert db.commit.call_count == 1
    assert stats == {
        "ok": 1,
        "unresolved": 0,
        "failed": 0,
        "pending": 0,
        "skipped_ok": 1,
        "total": 2,
    }
