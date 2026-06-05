from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.db.mysql import get_db
from app.main import app
from app.models.paper import Paper


def _mock_db(existing_paper=None):
    db = MagicMock()
    query = MagicMock()
    db.query.return_value = query
    query.filter.return_value = query
    query.one_or_none.return_value = existing_paper
    yield db


client = TestClient(app)


def _set_db_override(existing_paper=None):
    previous = app.dependency_overrides.get(get_db)
    app.dependency_overrides[get_db] = lambda: next(_mock_db(existing_paper))
    return previous


def _restore_db_override(previous) -> None:
    if previous is None:
        app.dependency_overrides.pop(get_db, None)
    else:
        app.dependency_overrides[get_db] = previous


def test_normalize_arxiv_id_accepts_ids_and_urls():
    from app.services.arxiv_import import normalize_arxiv_id

    assert normalize_arxiv_id("2511.16043") == "2511.16043"
    assert normalize_arxiv_id("arXiv:2511.16043v2") == "2511.16043"
    assert normalize_arxiv_id("https://arxiv.org/abs/2511.16043v3") == "2511.16043"
    assert normalize_arxiv_id("https://arxiv.org/pdf/2511.16043.pdf") == "2511.16043"


def test_build_arxiv_record_preserves_metadata_and_adds_topic_bucket():
    from app.services.arxiv_import import build_arxiv_record

    result = SimpleNamespace(
        get_short_id=lambda: "2511.16043v1",
        title="Agent0: Unleashing Self-Evolving Agents from Zero Data",
        authors=[SimpleNamespace(name="Ada Lovelace"), SimpleNamespace(name="Alan Turing")],
        published=datetime(2025, 11, 20, tzinfo=timezone.utc),
        updated=datetime(2025, 11, 21, tzinfo=timezone.utc),
        primary_category="cs.AI",
        categories=["cs.AI", "cs.LG"],
        doi="10.0000/example",
        summary="Tool-integrated reasoning enables self-evolving AI agents.",
        pdf_url="https://arxiv.org/pdf/2511.16043",
        entry_id="https://arxiv.org/abs/2511.16043v1",
    )

    record = build_arxiv_record(result)

    assert record["paper_id"] == "2511.16043"
    assert record["title"] == "Agent0: Unleashing Self-Evolving Agents from Zero Data"
    assert record["authors"] == ["Ada Lovelace", "Alan Turing"]
    assert record["year"] == 2025
    assert record["primary_category"] == "cs.AI"
    assert record["categories"] == ["cs.AI", "cs.LG", "agents_reasoning"]
    assert record["abstract"] == "Tool-integrated reasoning enables self-evolving AI agents."
    assert record["pdf_path"] is None


def test_fetch_arxiv_record_prefers_abs_page_metadata_over_export_api():
    from app.services.arxiv_import import fetch_arxiv_record

    html = """
    <html><head>
      <meta name="citation_title" content="Agent0: Unleashing Self-Evolving Agents from Zero Data" />
      <meta name="citation_author" content="Ada Lovelace" />
      <meta name="citation_author" content="Alan Turing" />
      <meta name="citation_date" content="2025/11/20" />
      <meta name="citation_pdf_url" content="https://arxiv.org/pdf/2511.16043" />
      <meta name="citation_arxiv_id" content="2511.16043" />
      <meta name="citation_abstract" content="Tool-integrated reasoning enables self-evolving AI agents." />
    </head><body>
      <td class="tablecell subjects">
        Artificial Intelligence (cs.AI); Machine Learning (cs.LG)
      </td>
    </body></html>
    """
    response = MagicMock()
    response.text = html
    response.raise_for_status.return_value = None

    with patch("app.services.arxiv_import.requests.get", return_value=response) as get, \
         patch("app.services.arxiv_import.arxiv.Client") as arxiv_client:
        record = fetch_arxiv_record("2511.16043")

    get.assert_called_once()
    assert get.call_args.args[0] == "https://arxiv.org/abs/2511.16043"
    arxiv_client.assert_not_called()
    assert record["paper_id"] == "2511.16043"
    assert record["title"] == "Agent0: Unleashing Self-Evolving Agents from Zero Data"
    assert record["authors"] == ["Ada Lovelace", "Alan Turing"]
    assert record["year"] == 2025
    assert record["primary_category"] == "cs.AI"
    assert record["categories"] == ["cs.AI", "cs.LG", "agents_reasoning"]
    assert record["pdf_url"] == "https://arxiv.org/pdf/2511.16043"


def test_arxiv_import_skips_existing_ingested_paper():
    existing = Paper(
        paper_id="2511.16043",
        title="Existing Paper",
        authors=[],
        year=2025,
        primary_category="cs.AI",
        categories=["agents_reasoning"],
        ingest_status="ok",
        num_chunks=42,
    )

    previous_override = _set_db_override(existing)
    try:
        with patch("app.routers.upload._run_arxiv_import_job") as run_job:
            resp = client.post("/upload/arxiv", json={"arxiv_ids": ["https://arxiv.org/abs/2511.16043v2"]})
    finally:
        _restore_db_override(previous_override)

    assert resp.status_code == 200
    item = resp.json()["items"][0]
    assert item["paper_id"] == "2511.16043"
    assert item["status"] == "skipped"
    assert item["num_chunks"] == 42
    assert item["message"] == "already_exists"
    run_job.assert_not_called()


def test_arxiv_import_queues_new_paper_without_inline_ingest():
    previous_override = _set_db_override(None)
    try:
        with patch("app.routers.upload._run_arxiv_import_job") as run_job, \
             patch("app.routers.upload._ingest_one") as ingest_one:
            resp = client.post("/upload/arxiv", json={"arxiv_ids": ["2511.16043"]})
    finally:
        _restore_db_override(previous_override)

    assert resp.status_code == 200
    item = resp.json()["items"][0]
    assert item["paper_id"] == "2511.16043"
    assert item["status"] == "queued"
    assert item["message"] == "queued"
    ingest_one.assert_not_called()
    run_job.assert_called_once()
    assert run_job.call_args.args[1] == "2511.16043"


def test_arxiv_import_rejects_invalid_id():
    previous_override = _set_db_override(None)
    try:
        resp = client.post("/upload/arxiv", json={"arxiv_ids": ["not-a-paper"]})
    finally:
        _restore_db_override(previous_override)

    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "invalid_arxiv_id"
