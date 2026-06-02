from __future__ import annotations

from app.evals.fixtures import build_eval_fixture
from app.evals.retrieval_quality import evaluate_retrieval_quality


def test_retrieval_quality_eval_catches_recall_isolation_ranking_and_visibility() -> None:
    fixture = build_eval_fixture()

    result = evaluate_retrieval_quality(fixture)

    assert result.passed is True
    assert result.checks == {
        "canon_lore_recalled": True,
        "session_memory_recalled": True,
        "persona_memory_recalled": True,
        "wrong_world_excluded": True,
        "wrong_session_excluded": True,
        "wrong_persona_excluded": True,
        "gm_only_excluded": True,
        "character_private_excluded": True,
        "relevant_memory_ranked_above_irrelevant": True,
    }

    retrieved_ids = [chunk.id for chunk in fixture.retrieve_actor_chunks(top_k=5)]
    assert result.checks["gm_only_excluded"] is True
    assert result.checks["character_private_excluded"] is True
    assert fixture.canon_lore_chunk_id in retrieved_ids
    assert fixture.session_memory_chunk_id in retrieved_ids
    assert fixture.persona_memory_chunk_id in retrieved_ids
    assert fixture.wrong_world_chunk_id not in retrieved_ids
    assert fixture.wrong_session_chunk_id not in retrieved_ids
    assert fixture.wrong_persona_chunk_id not in retrieved_ids
    assert fixture.gm_only_chunk_id not in retrieved_ids
    assert fixture.character_private_chunk_id not in retrieved_ids
