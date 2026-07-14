"""Direct MemoryDeduplicator coverage: dropped candidates must be auditable.

docs/22 P2.2 live finding (2026-07-12): a lexical write-dedup false-drop was
undiagnosable from the persisted warning alone ("dropped 1 duplicate
candidate(s)") -- the dropped text existed nowhere. The warning now carries a
snippet of each dropped summary.
"""

from __future__ import annotations

from app.domain import MemoryCandidate, Visibility
from app.memory.deterministic_extractor import is_covered_by_summaries
from app.orchestration.stages.memory_dedup import (
    CoverageMatch,
    MemoryDeduplicator,
    best_covering_summary,
    ordered_union,
)
from app.orchestration.stages.session_summary_cache import SessionSummaryCache


class _StubStore:
    def __init__(self, summaries: list[str]) -> None:
        self._summaries = summaries

    def list_memories_for_session(self, session_id: str) -> list[object]:
        class _Episode:
            def __init__(self, summary: str) -> None:
                self.summary = summary

        return [_Episode(s) for s in self._summaries]


def _candidate(summary: str) -> MemoryCandidate:
    return MemoryCandidate(summary=summary, visibility=Visibility.PLAYER, importance=2)


def test_lexical_drop_warning_names_the_dropped_summary() -> None:
    existing = "The player promised to guard the ledger until dawn."
    dedup = MemoryDeduplicator(
        cache=SessionSummaryCache(),
        embedding_provider=None,
        write_dedup_cosine_threshold=1.0,
    )
    warnings: list[str] = []
    kept = dedup.drop_duplicates(
        session_id="s1",
        candidates=[_candidate(existing)],  # identical -> covered -> dropped
        warnings=warnings,
        store=_StubStore([existing]),  # type: ignore[arg-type]
    )
    assert kept == []
    assert len(warnings) == 1
    assert warnings[0].startswith("memory dedup dropped 1 duplicate candidate(s)")
    assert "guard the ledger" in warnings[0]


def test_distinct_candidate_kept_with_no_warning() -> None:
    dedup = MemoryDeduplicator(
        cache=SessionSummaryCache(),
        embedding_provider=None,
        write_dedup_cosine_threshold=1.0,
    )
    warnings: list[str] = []
    kept = dedup.drop_duplicates(
        session_id="s1",
        candidates=[_candidate("The player gave Iria Vale a silver compass to keep.")],
        warnings=warnings,
        store=_StubStore(
            ["The player promised to return to the archive before dawn."]
        ),  # type: ignore[arg-type]
    )
    assert len(kept) == 1
    assert warnings == []


# --- best_covering_summary / ordered_union (docs/26 §3.2, #77) --------------------
# Pure-function coverage of the argmax-not-first-match helper the coverage-drop fold
# (stages/memory.py) relies on. Small, hand-verifiable strings here; the
# TurnMemoryStage-level fixtures in test_core_stages.py exercise the same helper
# through the real deterministic extractor + a scripted curator.


def test_best_covering_summary_picks_argmax_not_first_match() -> None:
    # "low" clears the 0.5 default threshold (2/4 == 0.5) and is listed FIRST; "high"
    # clears it by more (3/4 == 0.75) and is listed second. A first-match
    # implementation would stop at "low" (index 0); best-match must not.
    candidate = "wolf fox bear meadow"
    low = "wolf fox trail"
    high = "wolf fox bear trail"

    match = best_covering_summary(candidate, [low, high])

    assert match == CoverageMatch(index=1, score=0.75)


def test_best_covering_summary_tie_breaks_to_lowest_index() -> None:
    candidate = "wolf fox bear meadow"
    first = "wolf fox bear plain"
    second = "wolf fox bear valley"

    match = best_covering_summary(candidate, [first, second])

    assert match is not None
    assert match.index == 0
    assert match.score == 0.75


def test_best_covering_summary_returns_none_when_nothing_clears_threshold() -> None:
    candidate = "wolf fox bear meadow"
    unrelated = "completely unrelated forest story"

    assert best_covering_summary(candidate, [unrelated]) is None
    assert is_covered_by_summaries(candidate, [unrelated]) is False


def test_best_covering_summary_agrees_with_is_covered_by_summaries_when_covered() -> None:
    # For every input reachable in practice, best_covering_summary returning non-None
    # must agree exactly with is_covered_by_summaries returning True -- the fold must
    # not silently diverge from the (unmodified) coverage-drop decision.
    candidate = "The player promised to guard the ledger"
    summaries = ["The player promised to guard the ledger"]

    assert is_covered_by_summaries(candidate, summaries) is True
    assert best_covering_summary(candidate, summaries) is not None


def test_best_covering_summary_respects_reversal_marker_escape() -> None:
    # Pinned pair from test_deterministic_extractor.py's reversal test: term overlap
    # alone is 0.71 (well above the 0.5 default threshold), but the candidate's
    # reversal marker ("no longer", "revoked") is absent from the summary, so this is
    # a state CHANGE, not a duplicate -- must not be treated as covered.
    candidate = "Mira no longer trusts the player and revoked the safehouse offer"
    summary = "Mira trusts the player and offered the player the safehouse"

    assert is_covered_by_summaries(candidate, [summary]) is False
    assert best_covering_summary(candidate, [summary]) is None


def test_ordered_union_preserves_order_and_dedupes() -> None:
    assert ordered_union(["promise"], ["promise", "deadline"]) == ["promise", "deadline"]
    assert ordered_union([], ["a", "b"]) == ["a", "b"]
    assert ordered_union(["a", "b"], []) == ["a", "b"]
    assert ordered_union(["a", "b"], ["c", "a"]) == ["a", "b", "c"]
