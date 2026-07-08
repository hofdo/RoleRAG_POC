"""Direct coverage for SessionSummaryCache copy/invalidate semantics (#51).

Write-dedup compares new candidates against this per-session mirror; consolidation
invalidates it. If ``load()`` ever returned the live list instead of a copy, or
``invalidate()`` stopped forcing a reload, dedup would compare against a stale/mutated
summary set and silently drop legitimate memories or keep duplicates — a recall path
that was only covered indirectly before.
"""

from __future__ import annotations

from datetime import datetime

from app.domain import MemoryCandidate, MemoryEpisode, Visibility
from app.memory import MemoryEpisodeStore
from app.orchestration.stages.session_summary_cache import SessionSummaryCache


class _FakeMemoryRepository:
    """In-memory MemoryRepository (Protocol-structural) with a list-call counter."""

    def __init__(self) -> None:
        self.episodes: list[MemoryEpisode] = []
        self.list_calls = 0

    def append_memories(
        self, *, session_id: str, memories: list[MemoryCandidate]
    ) -> list[MemoryEpisode]:
        created = [
            MemoryEpisode(
                id=f"m{len(self.episodes) + i}",
                session_id=session_id,
                scene_id=candidate.scene_id or "scene",
                actor_id=candidate.actor_id,
                summary=candidate.summary,
                importance=candidate.importance,
                visibility=candidate.visibility,
                tags=list(candidate.tags),
            )
            for i, candidate in enumerate(memories)
        ]
        self.episodes.extend(created)
        return created

    def list_memories_for_session(
        self, session_id: str, limit: int | None = None
    ) -> list[MemoryEpisode]:
        self.list_calls += 1
        matches = [e for e in self.episodes if e.session_id == session_id]
        return matches if limit is None else matches[:limit]

    def add_tag_to_memories(self, memory_ids: list[str], tag: str) -> None:  # pragma: no cover
        for episode in self.episodes:
            if episode.id in memory_ids:
                episode.tags.append(tag)

    def delete_memories_since(
        self, session_id: str, created_at: datetime
    ) -> list[str]:  # pragma: no cover
        return []


def _candidate(summary: str) -> MemoryCandidate:
    return MemoryCandidate(summary=summary, visibility=Visibility.PLAYER, importance=3)


def _store_with(summaries: list[str]) -> tuple[MemoryEpisodeStore, _FakeMemoryRepository]:
    repo = _FakeMemoryRepository()
    store = MemoryEpisodeStore(memory_repository=repo)
    if summaries:
        store.persist_memories(
            session_id="s1", memories=[_candidate(text) for text in summaries]
        )
    return store, repo


def test_load_populates_from_store_once_then_caches() -> None:
    store, repo = _store_with(["a", "b"])
    cache = SessionSummaryCache()

    assert cache.load("s1", store) == ["a", "b"]
    assert cache.load("s1", store) == ["a", "b"]
    assert repo.list_calls == 1  # second load served from the in-process mirror


def test_load_returns_a_copy_so_caller_mutation_cannot_corrupt_the_mirror() -> None:
    store, _ = _store_with(["a", "b"])
    cache = SessionSummaryCache()

    handed_out = cache.load("s1", store)
    handed_out.append("mutated-by-caller")  # the dedup loop mutates its working list

    assert cache.load("s1", store) == ["a", "b"]  # mirror unchanged


def test_append_grows_the_cached_mirror() -> None:
    store, _ = _store_with(["a"])
    cache = SessionSummaryCache()
    cache.load("s1", store)  # populate the mirror

    cache.append("s1", ["b"])

    assert cache.load("s1", store) == ["a", "b"]


def test_append_is_a_noop_when_session_not_cached() -> None:
    store, _ = _store_with(["a"])
    cache = SessionSummaryCache()

    cache.append("s1", ["b"])  # never loaded -> nothing to grow

    assert cache.load("s1", store) == ["a"]


def test_invalidate_forces_reload_reflecting_new_store_state() -> None:
    # The consolidation -> invalidate -> reload sequence: after consolidation rewrites the
    # persisted summaries, the next load must re-read the store, not serve the stale mirror.
    store, repo = _store_with(["a", "b"])
    cache = SessionSummaryCache()
    assert cache.load("s1", store) == ["a", "b"]

    store.persist_memories(session_id="s1", memories=[_candidate("c")])
    assert cache.load("s1", store) == ["a", "b"]  # still stale (mirror not invalidated)

    cache.invalidate("s1")

    assert cache.load("s1", store) == ["a", "b", "c"]  # reloaded from the store
    assert repo.list_calls == 2  # first populate + post-invalidate reload
