from __future__ import annotations

from collections.abc import Sequence

from app.domain import MemoryEpisode, Visibility

# Tags that mark a memory as durable session canon worth pinning verbatim into
# the actor prompt. These mirror the durable-event vocabulary the deterministic
# extractor and curator already attach (see app/memory/deterministic_extractor.py).
CANON_TAGS: frozenset[str] = frozenset(
    {
        "promise",
        "entrusted",
        "deadline",
        "rule",
        "agreement",
        "signal",
        "code",
        "protocol",
        "oath",
        "vow",
        "pledge",
    }
)

def build_standing_facts(
    memories: Sequence[MemoryEpisode],
    *,
    importance_floor: int,
    max_items: int,
    max_chars: int,
) -> tuple[str, ...]:
    """Deterministically derive pinned canon lines from session memories.

    Pure projection of the memories already loaded for the session: no
    persistence, PLAYER-visibility only (GM / character-private memories never
    enter the actor prompt). Selection is ordered by importance then recency so
    the most load-bearing commitments survive the item/char caps.
    """
    eligible = [
        memory
        for memory in memories
        if memory.visibility == Visibility.PLAYER
        and memory.importance >= importance_floor
        and CANON_TAGS.intersection(memory.tags)
    ]
    eligible.sort(key=lambda memory: (-memory.importance, -_created_ordinal(memory), memory.id))

    selected: list[str] = []
    seen: set[str] = set()
    total_chars = 0
    for memory in eligible:
        line = memory.summary.strip()
        normalized = line.lower()
        if not line or normalized in seen:
            continue
        if len(selected) >= max_items:
            break
        if total_chars + len(line) > max_chars:
            continue
        selected.append(line)
        seen.add(normalized)
        total_chars += len(line)
    return tuple(selected)


def _created_ordinal(memory: MemoryEpisode) -> float:
    """Sort key for recency. Missing timestamps sort oldest (lowest)."""
    if memory.created_at is None:
        return float("-inf")
    return memory.created_at.timestamp()
