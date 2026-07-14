"""Offline D3-artifact replay (docs/26 Stage 0 + Stage 4, backlog #75/#79).

Two read-only replays over a preserved live-run SQLite artifact, no model / no
Qdrant / no embeddings:

* ``replay_selection`` (Stage 0, #75) -- each PLAYER-visible memory's canon/
  standing-facts eligibility under the CURRENT ``build_standing_facts`` predicate
  (importance, tags, ``CANON_TAGS`` intersection, eligible yes/no) plus a summary.
* ``replay_lexical`` (Stage 4, #79) -- the Lane B session-pool-IDF lexical scorer
  (``app.rag.lexical.score_memories_lexical``) run against a query, reporting each
  hit's summed-IDF score and matched terms and the rank of a target memory. Used
  to falsify the #73 blue-seal callback offline before any live run.

Opens the artifact strictly read-only. ``mode=ro`` alone is NOT sufficient: a
WAL-mode database still gets ``-shm``/``-wal`` sidecar files created next to it on
open even in read-only mode (verified empirically against
``docs/artifacts/live-validation-D3-2026-07-12.db``) -- ``immutable=1`` is required
to skip that machinery entirely and touch nothing on disk.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from app.domain import MemoryEpisode, Visibility
from app.orchestration.canon_builder import CANON_TAGS
from app.persistence.sqlite import parse_datetime
from app.rag.lexical import LexicalHit, score_memories_lexical

# Mirrors Settings.canon_importance_floor's default (app/config.py:127) -- not read
# from Settings so this script stays free of any env-derived configuration, matching
# its "no model, no Qdrant, no embedding" offline contract.
DEFAULT_IMPORTANCE_FLOOR = 4

# The docs/26 §3.4 / #73 blue-seal callback repro: this query's content terms hit
# only a handful of the D3 pool's memories, so the target rule memory is expected to
# land within the lexical quota. Defaults for the CLI lexical replay.
DEFAULT_LEXICAL_QUERY = "I ask Iria what rule we agreed to use before trusting any message"
DEFAULT_TARGET_MEMORY_PREFIX = "354b8d98"


@dataclass(frozen=True)
class MemoryEligibility:
    memory_id: str
    importance: int
    tags: tuple[str, ...]
    matched_canon_tags: tuple[str, ...]
    eligible: bool  # today's build_standing_facts predicate: tag AND importance floor


def replay_selection(
    *,
    database_path: Path,
    session_id: str,
    importance_floor: int = DEFAULT_IMPORTANCE_FLOOR,
) -> list[MemoryEligibility]:
    """Read-only replay of ``build_standing_facts``'s eligibility predicate
    (visibility == PLAYER, importance >= floor, CANON_TAGS intersects tags) against
    a preserved live-run artifact, oldest-first."""
    uri = f"{Path(database_path).resolve().as_uri()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        rows = connection.execute(
            "SELECT id, importance, tags_json FROM memory_episodes "
            "WHERE session_id = ? AND visibility = ? ORDER BY created_at ASC",
            (session_id, Visibility.PLAYER.value),
        ).fetchall()
    finally:
        connection.close()
    results: list[MemoryEligibility] = []
    for memory_id, importance, tags_json in rows:
        tags = tuple(json.loads(tags_json))
        matched = tuple(sorted(CANON_TAGS.intersection(tags)))
        results.append(
            MemoryEligibility(
                memory_id=memory_id,
                importance=importance,
                tags=tags,
                matched_canon_tags=matched,
                eligible=bool(matched) and importance >= importance_floor,
            )
        )
    return results


def summarize(memories: list[MemoryEligibility], *, importance_floor: int) -> str:
    distribution = Counter(memory.importance for memory in memories)
    dist = ", ".join(f"{count}x{importance}" for importance, count in sorted(distribution.items()))
    tag_eligible = sum(1 for memory in memories if memory.matched_canon_tags)
    eligible = sum(1 for memory in memories if memory.eligible)
    return (
        f"player_memories={len(memories)} importance_floor={importance_floor} "
        f"importance_distribution=[{dist}] tag_eligible={tag_eligible} eligible={eligible}"
    )


def load_memories(*, database_path: Path, session_id: str) -> list[MemoryEpisode]:
    """Read every memory episode for ``session_id`` from a preserved artifact,
    strictly read-only (``mode=ro&immutable=1`` -- see the module docstring), as
    ``MemoryEpisode`` objects the Lane B scorer can consume directly."""
    uri = f"{Path(database_path).resolve().as_uri()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        rows = connection.execute(
            "SELECT id, scene_id, actor_id, summary, importance, visibility, tags_json, "
            "created_at FROM memory_episodes WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        ).fetchall()
    finally:
        connection.close()
    memories: list[MemoryEpisode] = []
    for row in rows:
        memory_id, scene_id, actor_id, summary, importance, visibility, tags_json, created_at = row
        memories.append(
            MemoryEpisode(
                id=memory_id,
                session_id=session_id,
                scene_id=scene_id or "",
                actor_id=actor_id,
                summary=summary,
                importance=importance,
                visibility=Visibility(visibility),
                tags=json.loads(tags_json),
                created_at=parse_datetime(created_at) if created_at else None,
            )
        )
    return memories


def replay_lexical(
    *,
    database_path: Path,
    session_id: str,
    query_text: str,
) -> list[LexicalHit]:
    """Score the session's memory pool with the Lane B lexical scorer (docs/26
    §3.4) against ``query_text``, read-only. Returns hits in the scorer's
    deterministic descending order."""
    memories = load_memories(database_path=database_path, session_id=session_id)
    return score_memories_lexical(query_text=query_text, memories=memories)


def target_rank(hits: list[LexicalHit], *, target_prefix: str) -> int | None:
    """1-based rank of the first hit whose memory id starts with ``target_prefix``,
    or ``None`` if it did not match any query term (absent from the ranked hits)."""
    for rank, hit in enumerate(hits, start=1):
        if hit.memory_id.startswith(target_prefix):
            return rank
    return None


def _run_lexical(args: argparse.Namespace) -> None:
    hits = replay_lexical(
        database_path=args.database_path,
        session_id=args.session_id,
        query_text=args.lexical_query,
    )
    for rank, hit in enumerate(hits, start=1):
        print(
            f"#{rank} {hit.memory_id[:8]} score={hit.score:.4f} "
            f"importance={hit.importance} matched={list(hit.matched)}"
        )
    position = target_rank(hits, target_prefix=args.target_id)
    location = (
        f"rank {position}/{len(hits)}"
        if position is not None
        else "not in ranked hits (no term match)"
    )
    print(
        f"lexical_hits={len(hits)} query={args.lexical_query!r} "
        f"target={args.target_id} -> {location}"
    )


def _run_eligibility(args: argparse.Namespace) -> None:
    memories = replay_selection(
        database_path=args.database_path,
        session_id=args.session_id,
        importance_floor=args.importance_floor,
    )
    for memory in memories:
        print(
            f"{memory.memory_id[:8]} importance={memory.importance} tags={list(memory.tags)} "
            f"canon_tags={list(memory.matched_canon_tags)} "
            f"eligible={'yes' if memory.eligible else 'no'}"
        )
    print(summarize(memories, importance_floor=args.importance_floor))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Offline read-only replay against a preserved artifact: canon/standing-facts "
            "eligibility (default) or the Lane B lexical scorer (--lexical-query)."
        )
    )
    parser.add_argument("--database-path", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--importance-floor", type=int, default=DEFAULT_IMPORTANCE_FLOOR)
    parser.add_argument(
        "--lexical-query",
        nargs="?",
        const=DEFAULT_LEXICAL_QUERY,
        default=None,
        help=(
            "Run the Lane B lexical scorer against this query instead of the eligibility "
            f"replay. With no value, uses the #73 blue-seal repro query {DEFAULT_LEXICAL_QUERY!r}."
        ),
    )
    parser.add_argument("--target-id", default=DEFAULT_TARGET_MEMORY_PREFIX)
    args = parser.parse_args()

    if args.lexical_query is not None:
        _run_lexical(args)
    else:
        _run_eligibility(args)


if __name__ == "__main__":
    main()
