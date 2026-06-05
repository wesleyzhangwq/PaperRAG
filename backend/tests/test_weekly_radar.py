from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.services import weekly_radar
from app.services.weekly_radar import (
    PaperCandidate,
    RadarConfig,
    WeeklyRadarResult,
    dedupe_candidates,
    rank_candidates,
    to_ingest_record,
    write_reports,
)


def _candidate(
    paper_id: str,
    title: str,
    *,
    abstract: str = "",
    categories: list[str] | None = None,
    citation_count: int = 0,
    published: datetime | None = None,
) -> PaperCandidate:
    return PaperCandidate(
        paper_id=paper_id,
        title=title,
        authors=["A. Author"],
        year=published.year if published else 2026,
        published=published,
        updated=None,
        primary_category=(categories or ["cs.CL"])[0],
        categories=categories or ["cs.CL"],
        abstract=abstract,
        pdf_url=f"https://arxiv.org/pdf/{paper_id}.pdf",
        entry_id=f"https://arxiv.org/abs/{paper_id}",
        citation_count=citation_count,
    )


def test_dedupe_candidates_prefers_richer_record() -> None:
    sparse = _candidate("2606.00001v1", "Agentic RAG for Scientific Discovery")
    rich = _candidate(
        "2606.00001v2",
        "Agentic RAG for Scientific Discovery",
        abstract="A richer abstract",
        citation_count=3,
    )
    same_title = _candidate("2606.99999", "  Agentic   RAG for Scientific Discovery!!!  ")

    deduped = dedupe_candidates([sparse, rich, same_title])

    assert len(deduped) == 1
    assert deduped[0].paper_id == "2606.00001"
    assert deduped[0].abstract == "A richer abstract"
    assert deduped[0].citation_count == 3


def test_rank_candidates_prefers_vertical_relevance_and_impact() -> None:
    now = datetime(2026, 6, 5, tzinfo=timezone.utc)
    relevant = _candidate(
        "2606.00002",
        "Agentic Retrieval-Augmented Generation for Scientific Question Answering",
        abstract="A RAG agent with tool use, citation attribution, and factuality evaluation.",
        citation_count=10,
        published=now,
    )
    generic = _candidate(
        "2606.00003",
        "A Fast Optimizer for Image Classification",
        abstract="A generic training optimizer.",
        categories=["cs.LG"],
        citation_count=2,
        published=now,
    )

    ranked = rank_candidates([generic, relevant], RadarConfig(), now=now)

    assert ranked[0].candidate.paper_id == "2606.00002"
    assert ranked[0].score_breakdown["topic_relevance"] > ranked[1].score_breakdown["topic_relevance"]
    assert ranked[0].total_score > ranked[1].total_score


def test_rank_candidates_prioritizes_rag_over_generic_llm_agent() -> None:
    now = datetime(2026, 6, 5, tzinfo=timezone.utc)
    rag = _candidate(
        "2606.00006",
        "Efficient Retrieval-Augmented Generation for Scientific Question Answering",
        abstract="A RAG system with citation attribution.",
        published=now,
    )
    generic_agent = _candidate(
        "2606.00007",
        "Planning Capabilities in LLM Agents",
        abstract="A benchmark for generic LLM agents.",
        published=now,
    )

    ranked = rank_candidates([generic_agent, rag], RadarConfig(), now=now)

    assert ranked[0].candidate.paper_id == "2606.00006"
    assert ranked[0].score_breakdown["topic_relevance"] > ranked[1].score_breakdown["topic_relevance"]


def test_topic_matching_does_not_count_substrings() -> None:
    now = datetime(2026, 6, 5, tzinfo=timezone.utc)
    fragment = _candidate(
        "2606.00008",
        "How Agents Agree, Fragment, or Settle When Forming Conventions",
        abstract="Consensus dynamics among agents.",
        published=now,
    )
    explicit_rag = _candidate(
        "2606.00009",
        "RAG for Scientific Agents",
        abstract="Retrieval augmented generation for scientific question answering.",
        published=now,
    )

    ranked = rank_candidates([fragment, explicit_rag], RadarConfig(topic_terms=("rag",)), now=now)
    by_id = {item.candidate.paper_id: item for item in ranked}

    assert by_id["2606.00008"].score_breakdown["topic_relevance"] == 0
    assert by_id["2606.00009"].score_breakdown["topic_relevance"] > 0


def test_topic_matching_deduplicates_overlapping_terms() -> None:
    now = datetime(2026, 6, 5, tzinfo=timezone.utc)
    candidate = _candidate(
        "2606.00011",
        "Planning Capabilities in LLM Agents",
        abstract="A benchmark for LLM agents.",
        published=now,
    )

    ranked = rank_candidates(
        [candidate],
        RadarConfig(topic_terms=("llm agent", "llm agents", "agent", "agents")),
        now=now,
    )

    assert ranked[0].score_breakdown["topic_relevance"] == pytest.approx(7.68)
    assert ranked[0].reasons[0] == "topic: llm agent"


def test_to_ingest_record_preserves_arxiv_identity_and_pdf_path(tmp_path: Path) -> None:
    candidate = _candidate(
        "2606.00004v3",
        "RAG Agents for Science",
        abstract="About scientific RAG agents.",
        categories=["cs.AI", "cs.CL"],
    )
    pdf_path = tmp_path / "2606.00004.pdf"

    record = to_ingest_record(candidate, pdf_path)

    assert record["paper_id"] == "2606.00004"
    assert record["entry_id"] == "https://arxiv.org/abs/2606.00004"
    assert record["pdf_url"] == "https://arxiv.org/pdf/2606.00004.pdf"
    assert record["pdf_path"] == str(pdf_path)
    assert record["corpus_bucket"] == "weekly_agentic_rag_scientific_ai"


def test_write_reports_creates_json_and_markdown(tmp_path: Path) -> None:
    now = datetime(2026, 6, 5, tzinfo=timezone.utc)
    result = WeeklyRadarResult(
        generated_at=now,
        window_start=now,
        window_end=now,
        topic_name="agentic_rag_scientific_ai",
        candidate_count=1,
        selected=rank_candidates([_candidate("2606.00005", "RAG Evaluation for Scientific Agents")], RadarConfig(), now=now),
        ingest_stats={"ok": 0, "skipped": 0, "failed": 0, "total": 0},
        downloaded_pdfs=[],
        skipped_downloads=[],
    )

    json_path, md_path = write_reports(result, tmp_path)

    assert json_path.exists()
    assert md_path.exists()
    assert "RAG Evaluation for Scientific Agents" in md_path.read_text()
    assert '"candidate_count": 1' in json_path.read_text()


def test_run_weekly_radar_dry_run_writes_reports_without_ingest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    candidates = [
        _candidate(
            f"2606.{idx:05d}",
            f"Agentic RAG Scientific AI Paper {idx}",
            abstract="retrieval augmented generation agent evaluation citation",
            citation_count=idx,
        )
        for idx in range(12)
    ]
    config = RadarConfig(output_dir=tmp_path, pdf_dir=tmp_path / "pdfs", top_k=10)

    monkeypatch.setattr(weekly_radar, "fetch_recent_arxiv_candidates", lambda *_args, **_kwargs: candidates)
    monkeypatch.setattr(weekly_radar, "enrich_openalex_citations", lambda *_args, **_kwargs: None)

    def fail_ingest(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("dry-run must not call ingest")

    monkeypatch.setattr(weekly_radar, "run_ingest", fail_ingest)

    result = weekly_radar.run_weekly_radar(config=config, dry_run=True)

    assert len(result.selected) == 10
    assert result.ingest_stats["total"] == 0
    assert result.report_json is not None and result.report_json.exists()
    assert result.report_markdown is not None and result.report_markdown.exists()


def test_fetch_recent_arxiv_candidates_falls_back_to_recent_html(monkeypatch: pytest.MonkeyPatch) -> None:
    html = """
    <dt>
      <a href="/abs/2606.01234" title="Abstract">arXiv:2606.01234</a>
    </dt>
    <dd>
      <div class="meta">
        <div class="list-title mathjax">Title: Agentic RAG for Scientific AI</div>
        <div class="list-authors">Authors: Ada Lovelace, Alan Turing</div>
        <div class="list-subjects">Subjects: Computation and Language (cs.CL); Artificial Intelligence (cs.AI)</div>
      </div>
    </dd>
    """

    def fail_api(*_args: object, **_kwargs: object) -> list[PaperCandidate]:
        raise RuntimeError("arxiv api 429")

    monkeypatch.setattr(weekly_radar, "_fetch_arxiv_api_candidates", fail_api)
    monkeypatch.setattr(weekly_radar, "_fetch_recent_html_page", lambda *_args, **_kwargs: html)

    result = weekly_radar.fetch_recent_arxiv_candidates(
        RadarConfig(categories=("cs.CL",), max_candidates=10),
        datetime(2026, 6, 1, tzinfo=timezone.utc),
        datetime(2026, 6, 5, tzinfo=timezone.utc),
    )

    assert len(result) == 1
    assert result[0].paper_id == "2606.01234"
    assert result[0].title == "Agentic RAG for Scientific AI"
    assert result[0].authors == ["Ada Lovelace", "Alan Turing"]
    assert result[0].categories == ["cs.CL", "cs.AI"]


def test_recent_html_page_prefers_export_arxiv_host(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class FakeResponse:
        text = "<html></html>"

        def raise_for_status(self) -> None:
            return None

    def fake_get(url: str, **_kwargs: object) -> FakeResponse:
        calls.append(url)
        return FakeResponse()

    monkeypatch.setattr(weekly_radar.requests, "get", fake_get)

    html = weekly_radar._fetch_recent_html_page("cs.CL", RadarConfig(max_candidates=80))

    assert html == "<html></html>"
    assert calls[0].startswith("https://export.arxiv.org/list/cs.CL/recent")
    assert calls[0].endswith("show=100")


def test_openalex_enrichment_is_skipped_without_mailto(monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = _candidate("2606.00010", "RAG Agents for Science")
    calls = {"count": 0}

    class FakeSession:
        def get(self, *_args: object, **_kwargs: object) -> object:
            calls["count"] += 1
            return object()

    monkeypatch.setattr(weekly_radar.requests, "Session", FakeSession)

    weekly_radar.enrich_openalex_citations([candidate], RadarConfig(openalex_mailto=None))

    assert calls["count"] == 0
    assert candidate.citation_count == 0
