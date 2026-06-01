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
