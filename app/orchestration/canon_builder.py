from __future__ import annotations

from collections.abc import Sequence

from app.domain import MemoryEpisode, Visibility
from app.rag.ranking import content_terms

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

# German aliases for the durable-commitment tag family (docs/26 §3.3, #78), listed
# exactly per spec. Kept as a SEPARATE frozenset rather than merged into CANON_TAGS
# itself so the flag-off default tag set -- and every byte-identical-at-default
# contract downstream of it -- is untouched by this addition; effective_canon_tags()
# below only widens with these when canon_tag_pinning is on.
CANON_TAGS_DE: frozenset[str] = frozenset(
    {"regel", "versprechen", "abmachung", "frist", "schwur", "eid", "anvertraut"}
)


def effective_canon_tags(*, canon_tag_pinning: bool) -> frozenset[str]:
    """CANON_TAGS, widened with the German aliases when canon_tag_pinning is on.

    Flag off returns CANON_TAGS unchanged (same object) -- byte-identical default.
    """
    if not canon_tag_pinning:
        return CANON_TAGS
    return CANON_TAGS | CANON_TAGS_DE


def build_standing_facts(
    memories: Sequence[MemoryEpisode],
    *,
    pinned: Sequence[str] = (),
    importance_floor: int,
    max_items: int,
    max_chars: int,
    canon_tag_pinning: bool = False,
    warnings: list[str] | None = None,
) -> tuple[str, ...]:
    """Build the actor prompt's Standing-facts block.

    Author-pinned canon facts (`pinned`) come first verbatim, then facts derived
    from session memories (PLAYER-visibility only, durable-tagged, ordered by
    importance then recency). The combined list is deduplicated by text and bounded
    by the item/char caps so the most load-bearing facts survive.

    ``canon_tag_pinning`` (docs/26 §3.3, #78, default False = byte-identical) widens
    eligibility to CANON_TAGS-carrying memories below ``importance_floor`` (the
    blue-seal case: curator-tagged but never reaching the floor) and adds the German
    tag aliases to the matched-tag set. It also activates the narrow stale-fact
    tag-family-supersession safeguard (§3.3.1, see ``_drop_superseded_facts``): with
    the flag off, no supersession logic runs at all, regardless of fixture.

    ``warnings``, when provided, is appended to (mutated in place, mirroring
    ``MemoryDeduplicator.drop_duplicates``'s convention) with one audit message per
    safeguard drop, so the drop is visible in turn diagnostics.
    """
    canon_tags = effective_canon_tags(canon_tag_pinning=canon_tag_pinning)
    eligible = [
        memory
        for memory in memories
        if memory.visibility == Visibility.PLAYER
        and canon_tags.intersection(memory.tags)
        and (memory.importance >= importance_floor or canon_tag_pinning)
    ]
    if canon_tag_pinning:
        eligible = _drop_superseded_facts(eligible, canon_tags=canon_tags, warnings=warnings)
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


def _drop_superseded_facts(
    eligible: list[MemoryEpisode],
    *,
    canon_tags: frozenset[str],
    warnings: list[str] | None,
) -> list[MemoryEpisode]:
    """docs/26 §3.3.1 stale-fact safeguard. Caller-gated on ``canon_tag_pinning``.

    Drops an OLDER canon-eligible entry when a NEWER entry in the SAME canon-tag
    family has content terms that are a STRICT superset of the older entry's --
    the amendment case ("the rule is blue seal only" superseded by "the rule is now
    blue seal plus a knock"). Deliberately narrow (Q3: keep-both is the fallback):
    ambiguous cases keep BOTH entries pinned --

    * non-superset term overlap (partial overlap or disjoint)
    * equal term sets (no strict superset either way)
    * different/no shared tag family
    * either side missing ``created_at``, or not strictly newer

    Every drop appends an audit message to ``warnings`` (when provided), mirroring
    every other silent-drop-turned-audited-warning in this codebase (write-dedup,
    the coverage-drop fold in ``app.orchestration.stages.memory_dedup``).
    """
    terms_by_id = {memory.id: content_terms(memory.summary) for memory in eligible}
    drop_ids: set[str] = set()
    for older in eligible:
        if older.created_at is None:
            continue  # missing created_at: ambiguous, never drop (keep both)
        older_family = canon_tags.intersection(older.tags)
        older_terms = terms_by_id[older.id]
        for newer in eligible:
            if newer.id == older.id or newer.created_at is None:
                continue
            if newer.created_at <= older.created_at:
                continue  # not strictly newer: ambiguous, keep both
            if not (canon_tags.intersection(newer.tags) & older_family):
                continue  # different/no shared tag family: keep both
            if not (terms_by_id[newer.id] > older_terms):
                continue  # not a strict superset (incl. equal sets): keep both
            drop_ids.add(older.id)
            if warnings is not None:
                warnings.append(
                    "stale canon fact superseded (tag family="
                    f"{sorted(older_family)}): dropping older pinned fact "
                    f'"{_snippet(older.summary)}" superseded by newer '
                    f'"{_snippet(newer.summary)}"'
                )
            break
    return [memory for memory in eligible if memory.id not in drop_ids]


def _snippet(text: str, limit: int = 80) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _created_ordinal(memory: MemoryEpisode) -> float:
    """Sort key for recency. Missing timestamps sort oldest (lowest)."""
    if memory.created_at is None:
        return float("-inf")
    return memory.created_at.timestamp()
