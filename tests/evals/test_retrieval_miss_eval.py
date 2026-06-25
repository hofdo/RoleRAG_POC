from __future__ import annotations

import pytest

import app.diagnostics.retrieval_miss as retrieval_miss_module
from app.evals.event_key_retrieval import SEEDED_EVENTS
from app.evals.fixtures import build_eval_fixture
from app.evals.retrieval_miss import evaluate_retrieval_miss


def test_retrieval_miss_passes() -> None:
    result = evaluate_retrieval_miss(build_eval_fixture())

    assert result.passed is True
    for event in SEEDED_EVENTS:
        assert result.checks[f"{event.key}_present"] is True
        assert result.checks[f"{event.key}_selected"] is True
        assert result.checks[f"{event.key}_within_rank_floor"] is True
        assert result.checks[f"{event.key}_above_score_floor"] is True


def test_retrieval_miss_enforces_score_floor() -> None:
    # A floor above every baseline adjusted_score must trip the score check,
    # proving the floor is load-bearing rather than always-satisfied.
    result = evaluate_retrieval_miss(build_eval_fixture(), min_adjusted_score=2.0)

    assert result.passed is False
    for event in SEEDED_EVENTS:
        assert result.checks[f"{event.key}_above_score_floor"] is False


def test_retrieval_miss_enforces_rank_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    # Reverse the candidate ranking so durable memories fall to the tail of the
    # candidate set. With a tight rank floor this must fail, proving the rank
    # check would catch a future tradeoff that demotes a wanted memory.
    original = retrieval_miss_module._ordered_candidates

    def reversed_ranking(selected, rejected):  # type: ignore[no-untyped-def]
        return list(reversed(original(selected, rejected)))

    monkeypatch.setattr(retrieval_miss_module, "_ordered_candidates", reversed_ranking)

    result = evaluate_retrieval_miss(build_eval_fixture(), max_rank=1)

    assert result.passed is False
    assert any(
        result.checks[f"{event.key}_within_rank_floor"] is False for event in SEEDED_EVENTS
    )
