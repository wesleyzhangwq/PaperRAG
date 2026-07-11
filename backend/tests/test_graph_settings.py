from app.core.config import Settings


def test_graph_settings_default_to_a_disabled_bounded_feature() -> None:
    settings = Settings(_env_file=None)

    assert settings.graph_rag_enabled is False
    assert settings.graph_seed_papers == 4
    assert settings.graph_max_hops == 2
    assert settings.graph_candidate_limit == 12
    assert settings.graph_query_timeout_ms == 800
    assert settings.semantic_scholar_min_interval_sec == 1.1
    assert settings.semantic_scholar_max_retries == 5
    assert settings.semantic_scholar_neighbor_limit == 1000
    assert settings.neo4j_uri is None


def test_graph_settings_accept_explicit_connection_values(monkeypatch) -> None:
    monkeypatch.setenv("GRAPH_RAG_ENABLED", "true")
    monkeypatch.setenv("NEO4J_URI", "bolt://neo4j:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "local-graph-password")

    settings = Settings(_env_file=None)

    assert settings.graph_rag_enabled is True
    assert settings.neo4j_uri == "bolt://neo4j:7687"
    assert settings.neo4j_user == "neo4j"
