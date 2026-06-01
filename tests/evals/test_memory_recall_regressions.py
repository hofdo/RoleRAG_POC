from __future__ import annotations

from app.evals.fixtures import build_eval_fixture
from app.evals.memory_recall import evaluate_memory_recall


def test_memory_recall_eval_proves_newly_indexed_memories_are_retrievable() -> None:
    fixture = build_eval_fixture()

    result = evaluate_memory_recall(fixture)

    assert result.passed is True
    assert result.checks["indexed_player_memory_retrievable"] is True
    assert result.checks["indexed_gm_memory_excluded"] is True
    assert result.checks["indexed_character_private_memory_excluded"] is True
