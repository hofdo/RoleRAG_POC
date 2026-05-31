from __future__ import annotations

from app.evals.fixtures import build_eval_fixture
from app.evals.retrieval_quality import evaluate_retrieval_quality


def test_retrieval_quality_eval_catches_ranking_and_visibility_expectations() -> None:
    fixture = build_eval_fixture()

    result = evaluate_retrieval_quality(fixture, top_k=3)

    assert result.passed is True
    assert result.checks["public_lore_in_top_k"] is True
    assert result.checks["relevant_memory_in_top_k"] is True
    assert result.checks["irrelevant_memory_ranked_lower"] is True
    assert result.checks["gm_only_excluded"] is True
    assert result.checks["character_private_excluded"] is True
    assert fixture.public_lore_chunk_id in result.retrieved_ids
    assert fixture.relevant_memory_chunk_id in result.retrieved_ids
    assert fixture.gm_only_chunk_id not in result.retrieved_ids
    assert fixture.character_private_chunk_id not in result.retrieved_ids
