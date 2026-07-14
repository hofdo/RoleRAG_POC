from __future__ import annotations

import subprocess
import sys

from app.evals.regression_runner import run_regressions


def test_regression_runner_reports_all_eval_categories() -> None:
    report = run_regressions()

    assert report.passed is True
    assert report.total_checks >= 5
    assert {result.name for result in report.results} == {
        "retrieval",
        "event_key_retrieval",
        "memory_recall",
        "visibility",
        "role_consistency",
        "memory",
        "memory_continuity",
        "memory_write_lifecycle",
        "draft_validation",
        "provider_binding",
        "containment",
        "structured_resilience",
        "retrieval_miss",
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


def test_regression_runner_module_prints_memory_continuity_without_runpy_warning() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "app.evals.regression_runner"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "PASS memory_continuity:" in completed.stdout
    assert "RuntimeWarning" not in completed.stderr
    assert "found in sys.modules" not in completed.stderr
