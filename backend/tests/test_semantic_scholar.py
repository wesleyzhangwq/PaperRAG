from unittest.mock import MagicMock, patch


def _response(payload: dict) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = payload
    return response


def test_fetch_snapshot_maps_arxiv_identifiers_and_both_citation_directions() -> None:
    from app.services.semantic_scholar import fetch_citation_snapshot

    with patch("app.services.semantic_scholar.requests.get") as mock_get:
        mock_get.side_effect = [
        _response({
            "paperId": "s2-source",
            "externalIds": {"ArXiv": "2401.00001"},
            "title": "Source",
            "year": 2024,
            "authors": [],
        }),
        _response({
            "data": [{
                "citedPaper": {
                    "paperId": "s2-ref",
                    "externalIds": {"ArXiv": "2301.00001"},
                    "title": "Reference",
                    "year": 2023,
                    "authors": [],
                },
            }],
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
