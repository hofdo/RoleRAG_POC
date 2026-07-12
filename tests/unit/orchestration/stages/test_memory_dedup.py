"""Direct MemoryDeduplicator coverage: dropped candidates must be auditable.

docs/22 P2.2 live finding (2026-07-12): a lexical write-dedup false-drop was
undiagnosable from the persisted warning alone ("dropped 1 duplicate
candidate(s)") -- the dropped text existed nowhere. The warning now carries a
snippet of each dropped summary.
"""

from __future__ import annotations

from app.domain import MemoryCandidate, Visibility
from app.orchestration.stages.memory_dedup import MemoryDeduplicator
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
