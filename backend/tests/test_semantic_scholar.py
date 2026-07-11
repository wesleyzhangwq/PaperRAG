from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest


def _response(payload: dict) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = payload
    return response


def test_fetch_snapshot_maps_arxiv_identifiers_and_both_citation_directions() -> None:
    from app.services.semantic_scholar import fetch_citation_snapshot

    with patch("app.services.semantic_scholar.requests.get") as mock_get, patch(
        "app.services.semantic_scholar._wait_for_rate_limit"
    ):
        mock_get.side_effect = [
        _response({
            "paperId": "s2-source",
            "externalIds": {"ArXiv": "2401.00001"},
            "title": "Source",
            "year": 2024,
            "authors": [],
        }),
        _response({
            "data": [
                {"citedPaper": {"paperId": None, "title": "Unresolved"}},
                {"citedPaper": {
                    "paperId": "s2-ref",
                    "externalIds": {"ArXiv": "2301.00001"},
                    "title": "Reference",
                    "year": 2023,
                    "authors": [],
                }},
            ],
            "next": None,
        }),
        _response({
            "data": [{
                "citingPaper": {
                    "paperId": "s2-citing",
                    "externalIds": {"ArXiv": "2501.00001"},
                    "title": "Citing",
                    "year": 2025,
                    "authors": [],
                },
            }],
            "next": None,
        }),
        ]

        snapshot = fetch_citation_snapshot(arxiv_id="2401.00001", doi=None)

    assert snapshot.source.s2_paper_id == "s2-source"
    assert [paper.arxiv_id for paper in snapshot.references] == ["2301.00001"]
    assert [paper.arxiv_id for paper in snapshot.citations] == ["2501.00001"]
    assert "ARXIV:2401.00001" in mock_get.call_args_list[0].args[0]
    assert mock_get.call_args_list[1].kwargs["params"]["fields"].endswith(
        ",year,authors"
    )


def test_request_retries_rate_limit_with_api_key_header() -> None:
    from app.services.semantic_scholar import _request_json

    limited = MagicMock(status_code=429, text="slow down", headers={"Retry-After": "2"})
    ok = _response({"paperId": "s2-source"})
    settings = SimpleNamespace(
        semantic_scholar_api_key="secret",
        semantic_scholar_max_retries=3,
        semantic_scholar_retry_backoff_sec=0.25,
        semantic_scholar_min_interval_sec=1.1,
    )
    with patch("app.services.semantic_scholar.get_settings", return_value=settings), patch(
        "app.services.semantic_scholar.requests.get", side_effect=[limited, ok]
    ) as mock_get, patch(
        "app.services.semantic_scholar._wait_for_rate_limit"
    ) as mock_wait, patch("app.services.semantic_scholar.time.sleep") as mock_sleep:
        payload = _request_json("/paper/ARXIV:2401.00001")

    assert payload == {"paperId": "s2-source"}
    assert mock_wait.call_count == 2
    assert mock_sleep.call_args_list == [call(2.0)]
    assert mock_get.call_args_list[0].kwargs["headers"] == {"x-api-key": "secret"}


def test_rate_limiter_keeps_request_starts_below_one_per_second() -> None:
    import app.services.semantic_scholar as service

    settings = SimpleNamespace(semantic_scholar_min_interval_sec=1.1)
    with patch.object(service, "get_settings", return_value=settings), patch.object(
        service.time, "monotonic", side_effect=[10.0, 10.2]
    ), patch.object(service.time, "sleep") as mock_sleep:
        service._last_request_started_at = 0.0
        service._wait_for_rate_limit()
        service._wait_for_rate_limit()

    mock_sleep.assert_called_once()
    assert mock_sleep.call_args.args[0] == pytest.approx(0.9)


def test_neighbor_fetch_stops_at_configured_limit_even_when_api_has_next_page() -> None:
    from app.services.semantic_scholar import _fetch_paged_papers

    payload = {
        "data": [
            {"citedPaper": {"paperId": "s2-a", "title": "A", "authors": []}},
            {"citedPaper": {"paperId": "s2-b", "title": "B", "authors": []}},
        ],
        "next": 1000,
    }
    with patch(
        "app.services.semantic_scholar._request_json", return_value=payload
    ) as mock_request:
        papers = _fetch_paged_papers(
            "ARXIV:2401.00001",
            "references",
            "citedPaper",
            max_papers=1,
        )

    assert [paper.s2_paper_id for paper in papers] == ["s2-a"]
    mock_request.assert_called_once()
