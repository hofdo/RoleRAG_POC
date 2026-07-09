"""Memory consolidation ("sleep cycle") helpers.

When low-value episodic memories accumulate, they crowd the retrievable index
with sparse, redundant entries. Consolidation rolls a batch of old, low-importance,
non-durable memories into a single dense summary, marks the originals consolidated
(so they are never re-indexed), and indexes only the summary. It is non-destructive:
originals stay in SQLite, so a cold reindex still recovers everything.

The summary text is produced by the curator (LLM) with a deterministic fallback;
this module holds the deterministic pieces and the selection policy.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.domain import MemoryEpisode, Visibility

CONSOLIDATED_TAG = "consolidated"
SUMMARY_TAG = "consolidation_summary"

# Tags that mark a memory as durable session canon — never consolidated. Mirrors
# app/orchestration/canon_builder.py CANON_TAGS; the importance ceiling below is
# the primary guard, this is belt-and-suspenders for low-importance durable events.
_PRESERVE_TAGS: frozenset[str] = frozenset(
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

CONSOLIDATION_PROMPT = (
    "You compress past roleplay memories. Given a list of earlier minor events, "
    "write a single concise summary (2-4 sentences, third person) that preserves "
    "any concrete facts, names, and commitments. Do not invent details. Output the "
    "summary text only, with no preamble."
)


def select_consolidatable(
    memories: Sequence[MemoryEpisode],
    *,
    importance_ceiling: int,
    min_age: int = 0,
    batch_cap: int = 0,
) -> list[MemoryEpisode]:
    """Old, low-importance, non-durable, not-already-consolidated PLAYER memories,
    ordered oldest first (the natural batch to roll up).

    Age is rank-based, derived from the oldest-first ordering already computed here
    (created_at, then id as a deterministic tie-break) -- no wall-clock dependency.
    ``min_age`` keeps the ``min_age`` newest eligible memories out of the result (a
    rolling recent window), so a low-importance memory written this turn always gets
    at least ``min_age`` turns of standalone retrievability before it can be folded.
    ``batch_cap`` caps the result to the oldest N after the age floor is applied.
    Both default to 0, which is a no-op: 0 excludes nothing and 0 means unlimited,
    reproducing the pre-C2 selection byte-for-byte.
    """
    eligible = [
        memory
        for memory in memories
        if memory.visibility == Visibility.PLAYER
        and memory.importance <= importance_ceiling
        and not _PRESERVE_TAGS.intersection(memory.tags)
        and CONSOLIDATED_TAG not in memory.tags
        and SUMMARY_TAG not in memory.tags
    ]
    eligible.sort(key=_age_sort_key)
    if min_age > 0:
        eligible = eligible[:-min_age] if min_age < len(eligible) else []
    if batch_cap > 0:
        eligible = eligible[:batch_cap]
    return eligible


def deterministic_consolidated_summary(memories: Sequence[MemoryEpisode]) -> str:
    """Fallback summary when the LLM call fails: a bounded, ordered roll-up of the
    original summaries. Reliable and lossless in content (if verbose)."""
    parts = [memory.summary.strip() for memory in memories if memory.summary.strip()]
    joined = " ".join(parts)
    return f"Earlier in this session: {joined}"


def _age_sort_key(memory: MemoryEpisode) -> tuple[float, str]:
    created = memory.created_at.timestamp() if memory.created_at is not None else float("-inf")
    return (created, memory.id)
