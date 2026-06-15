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
    pinned: Sequence[str] = (),
    importance_floor: int,
    max_items: int,
    max_chars: int,
) -> tuple[str, ...]:
    """Build the actor prompt's Standing-facts block.

    Author-pinned canon facts (`pinned`) come first verbatim, then facts derived
    from session memories (PLAYER-visibility only, durable-tagged, ordered by
    importance then recency). The combined list is deduplicated by text and bounded
    by the item/char caps so the most load-bearing facts survive.
    """
    eligible = [
        memory
        for memory in memories
        if memory.visibility == Visibility.PLAYER
        and memory.importance >= importance_floor
        and CANON_TAGS.intersection(memory.tags)
    ]
    eligible.sort(key=lambda memory: (-memory.importance, -_created_ordinal(memory), memory.id))

    candidate_lines = [*pinned, *(memory.summary for memory in eligible)]

    selected: list[str] = []
    seen: set[str] = set()
    total_chars = 0
    for candidate in candidate_lines:
        line = candidate.strip()
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
