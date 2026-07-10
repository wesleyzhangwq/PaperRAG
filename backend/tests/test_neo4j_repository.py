from unittest.mock import MagicMock

def test_expand_local_papers_returns_only_distinct_local_candidates() -> None:
    from app.db.neo4j import Neo4jGraphRepository

    driver = MagicMock()
    session = driver.session.return_value.__enter__.return_value
    session.run.return_value = [
        {
            "paper_id": "B",
            "seed_paper_id": "A",
            "hops": 1,
            "relations": ["CITES"],
        },
        {
            "paper_id": "B",
            "seed_paper_id": "A",
            "hops": 2,
            "relations": ["CITES", "IN_CATEGORY"],
        },
    ]
    repo = Neo4jGraphRepository(driver=driver, database="neo4j", timeout_ms=800)

    candidates = repo.expand_local_papers(
        seed_paper_ids=["A"],
        seed_scores={"A": 0.9},
        max_hops=2,
        limit=12,
    )

    assert [candidate.paper_id for candidate in candidates] == ["B"]
    assert candidates[0].graph_score == 0.9
    assert candidates[0].paths[0]["relations"] == ["CITES"]
    assert "MATCH path" in session.run.call_args.args[0]
