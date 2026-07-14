from __future__ import annotations

from app.evals.memory_write_lifecycle import evaluate_memory_write_lifecycle


def test_memory_write_lifecycle_eval_proves_extraction_dedup_fold_and_auditable_drops() -> None:
    result = evaluate_memory_write_lifecycle()

    assert result.passed is True
    assert result.checks == {
        "promise_turn_writes_memory": True,
        "durable_fact_survives_extraction_and_dedup": True,
        "dawn_promise_seed_persisted": True,
        "distinct_compass_candidate_survives_write_dedup": True,
        "near_verbatim_restatement_dropped_by_write_dedup": True,
        "dropped_restatement_is_auditable": True,
        "deterministic_tag_and_importance_fold_onto_curated_summary": True,
        "fold_is_auditable": True,
        "promise_memory_pinned_regardless_of_flag": True,
        "fold_target_memory_pinned_regardless_of_flag": True,
        "compass_memory_pinned_only_with_flag_on": True,
    }
