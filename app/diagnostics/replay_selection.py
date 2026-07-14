"""Offline D3-artifact replay (docs/26 Stage 0, backlog #75).

Reports each PLAYER-visible memory's canon/standing-facts eligibility under the
CURRENT ``build_standing_facts`` predicate (``app/orchestration/canon_builder.py``)
for a preserved live-run SQLite artifact and a given session: importance, tags,
``CANON_TAGS`` intersection, and eligible yes/no, plus a summary line. No model, no
Qdrant, no embeddings -- a pure SQLite read.

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

from app.domain import Visibility
from app.orchestration.canon_builder import CANON_TAGS

# Mirrors Settings.canon_importance_floor's default (app/config.py:127) -- not read
# from Settings so this script stays free of any env-derived configuration, matching
# its "no model, no Qdrant, no embedding" offline contract.
DEFAULT_IMPORTANCE_FLOOR = 4


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline canon/standing-facts eligibility replay against a read-only artifact."
    )
    parser.add_argument("--database-path", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--importance-floor", type=int, default=DEFAULT_IMPORTANCE_FLOOR)
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
