from app.core.config import Settings


def test_rag_defaults_match_frozen_eval_winner() -> None:
    fields = Settings.model_fields

    assert fields["retrieval_k"].default == 20
    assert fields["final_context_k"].default == 5
    assert fields["hybrid_oversample"].default == 4.0
    assert fields["hybrid_alpha"].default == 0.5
    assert fields["hybrid_max_fetch"].default == 96
    assert fields["agent_external_retrieval_enabled"].default is True
    assert fields["agent_routing_mode"].default == "full_agentic"
