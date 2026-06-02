from __future__ import annotations

from app.evals.regression_runner import run_regressions


def test_regression_runner_reports_all_eval_categories() -> None:
    report = run_regressions()

    assert report.passed is True
    assert report.total_checks >= 5
    assert {result.name for result in report.results} == {
        "retrieval",
        "memory_recall",
        "visibility",
        "role_consistency",
        "memory",
        "cloud_routing",
    }
    assert all(result.passed for result in report.results)
    retrieval = next(result for result in report.results if result.name == "retrieval")
    assert set(retrieval.checks) == {
        "canon_lore_recalled",
        "session_memory_recalled",
        "persona_memory_recalled",
        "wrong_world_excluded",
        "wrong_session_excluded",
        "wrong_persona_excluded",
        "gm_only_excluded",
        "character_private_excluded",
        "relevant_memory_ranked_above_irrelevant",
    }
