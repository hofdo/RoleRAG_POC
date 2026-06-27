from __future__ import annotations

from app.memory import MemoryEpisodeStore


class SessionSummaryCache:
    """Per-session, in-process mirror of persisted memory summaries for write-dedup.

    ponytail: single-user ceiling. Keyed per session, in-process only; it mirrors persisted
    summaries so dedup stops reloading the whole session every turn. Concurrent out-of-process
    writes to the same session are NOT reflected until the entry is invalidated (consolidation
    drops it) or the process restarts. Fine for one-writer-per-session POC scale; revisit if
    sessions go shared.
    """

    def __init__(self) -> None:
        self._cache: dict[str, list[str]] = {}

    def load(self, session_id: str, store: MemoryEpisodeStore) -> list[str]:
        """Return a COPY of the session's summaries, populating from the store on first miss.

        Callers mutate the returned list during the dedup loop; the cached mirror must only grow
        with actually-persisted summaries (via append), so a copy is handed out here.
        """
        cached = self._cache.get(session_id)
        if cached is None:
            cached = [
                episode.summary for episode in store.list_memories_for_session(session_id)
            ]
            self._cache[session_id] = cached
        return list(cached)

    def append(self, session_id: str, summaries: list[str]) -> None:
        cached = self._cache.get(session_id)
        if cached is not None:
            cached.extend(summaries)

    def invalidate(self, session_id: str) -> None:
        self._cache.pop(session_id, None)
