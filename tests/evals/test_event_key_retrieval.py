from __future__ import annotations

from app.evals.event_key_retrieval import SEEDED_EVENTS, evaluate_event_key_retrieval
from app.evals.fixtures import build_eval_fixture


def test_event_key_retrieval_covers_all_five_study_events() -> None:
    assert [event.key for event in SEEDED_EVENTS] == [
        "before_dawn_promise",
        "silver_compass",
        "blue_seal_trust_rule",
        "key_hiding_place",
        "three_tap_signal",
    ]


def test_event_key_retrieval_selects_each_seeded_memory_at_callback() -> None:
    result = evaluate_event_key_retrieval(build_eval_fixture())

    assert result.passed is True
    for event in SEEDED_EVENTS:
        assert result.checks[f"{event.key}_selected"] is True
    assert result.checks["diagnostics_record_rejected_candidates"] is True


def test_event_key_retrieval_reports_selected_ids_per_event() -> None:
    result = evaluate_event_key_retrieval(build_eval_fixture())

    assert set(result.selected_ids) == {event.key for event in SEEDED_EVENTS}
    assert all(ids for ids in result.selected_ids.values())
