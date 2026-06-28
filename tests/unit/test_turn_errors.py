from __future__ import annotations

from app.orchestration.turn_errors import classify_warning, classify_warnings


def test_known_prefixes_map_to_stage_and_category() -> None:
    assert classify_warning("retrieval skipped: qdrant offline").stage == "retrieval"
    assert classify_warning("draft withheld: critic unavailable").category == "fail_closed"
    assert classify_warning("memory curation skipped: boom (parse)").stage == "memory"
    assert classify_warning("containment risk: maybe paraphrased").category == "security"
    assert classify_warning("cloud actor skipped: cloud mode is off").stage == "routing"


def test_specific_prefix_wins_over_generic() -> None:
    assert classify_warning("memory dedup skipped: boom").category == "dedup_skipped"
    assert classify_warning("memory dedup dropped 2 duplicate candidate(s)").category == "dedup"


def test_unknown_warning_falls_back_to_general() -> None:
    error = classify_warning("something unexpected happened")
    assert error.stage == "general"
    assert error.category == "warning"
    assert error.suggestion is None
    assert error.message == "something unexpected happened"  # original text preserved


def test_classify_warnings_preserves_order_and_count() -> None:
    errors = classify_warnings(["retrieval skipped: a", "totally unknown"])
    assert [error.stage for error in errors] == ["retrieval", "general"]
