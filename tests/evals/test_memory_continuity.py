from __future__ import annotations

from app.evals.memory_continuity import evaluate_memory_continuity


def test_memory_continuity_eval_proves_bounded_history_and_scoped_reindex_recovery() -> None:
    result = evaluate_memory_continuity()

    assert result.passed is True
    assert result.checks == {
        "sixteen_turns_complete": True,
        "actor_history_never_exceeds_four_prior_turns": True,
        "early_dialogue_markers_leave_recent_history": True,
        "actor_prompt_message_count_plateaus": True,
        "actor_prompt_chars_plateau": True,
        "old_promise_recalled_from_durable_memory": True,
        "hidden_memories_excluded_from_actor_prompts": True,
        "greeting_only_turns_skip_memory": True,
        "invalid_visibility_skipped": True,
        "fresh_reindex_recovers_sqlite_promise": True,
        "reindexed_recovery_excludes_scoped_distractors": True,
    }
