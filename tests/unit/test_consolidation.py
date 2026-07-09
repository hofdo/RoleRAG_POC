from __future__ import annotations

from datetime import UTC, datetime

from app.domain import MemoryEpisode, Visibility
from app.memory.consolidation import (
    CONSOLIDATED_TAG,
    SUMMARY_TAG,
    deterministic_consolidated_summary,
    select_consolidatable,
)


def _episode(
    memory_id: str,
    *,
    importance: int,
    tags: list[str] | None = None,
    visibility: Visibility = Visibility.PLAYER,
    created_at: datetime | None = None,
) -> MemoryEpisode:
    return MemoryEpisode(
        id=memory_id,
        session_id="session-1",
        scene_id="rose-gallery",
        actor_id="archivist",
        summary=f"Event {memory_id}",
        importance=importance,
        visibility=visibility,
        tags=tags if tags is not None else ["mood"],
        created_at=created_at,
    )


def test_select_consolidatable_filters_durable_high_and_already_processed() -> None:
    memories = [
        _episode("low", importance=2),
        _episode("high", importance=4),  # above ceiling -> preserved
        _episode("durable", importance=1, tags=["promise"]),  # canon tag -> preserved
        _episode("gm", importance=1, visibility=Visibility.GM),  # not player -> preserved
        _episode("already", importance=1, tags=[CONSOLIDATED_TAG]),  # already consolidated
        _episode("summary", importance=1, tags=[SUMMARY_TAG]),  # a prior summary
    ]

    selected = select_consolidatable(memories, importance_ceiling=3)

    assert [memory.id for memory in selected] == ["low"]


def test_select_consolidatable_orders_oldest_first() -> None:
    memories = [
        _episode("newer", importance=1, created_at=datetime(2026, 6, 2, tzinfo=UTC)),
        _episode("older", importance=1, created_at=datetime(2026, 6, 1, tzinfo=UTC)),
    ]

    selected = select_consolidatable(memories, importance_ceiling=3)

    assert [memory.id for memory in selected] == ["older", "newer"]


def _aged_memories(count: int) -> list[MemoryEpisode]:
    """`count` eligible memories, oldest (id 0) to newest (id count-1)."""
    return [
        _episode(str(i), importance=1, created_at=datetime(2026, 6, 1 + i, tzinfo=UTC))
        for i in range(count)
    ]


def test_select_consolidatable_defaults_reproduce_old_selection_byte_for_byte() -> None:
    """min_age=0 and batch_cap=0 (the defaults) must be exact no-ops: same input,
    same output, with or without explicitly passing the new keyword args."""
    memories = [
        _episode("low", importance=2),
        _episode("high", importance=4),
        _episode("durable", importance=1, tags=["promise"]),
        _episode("gm", importance=1, visibility=Visibility.GM),
        _episode("already", importance=1, tags=[CONSOLIDATED_TAG]),
        _episode("summary", importance=1, tags=[SUMMARY_TAG]),
        *_aged_memories(5),
    ]

    old_call = select_consolidatable(memories, importance_ceiling=3)
    new_call_explicit_defaults = select_consolidatable(
        memories, importance_ceiling=3, min_age=0, batch_cap=0
    )

    assert old_call == new_call_explicit_defaults
    assert [memory.id for memory in old_call] == ["low", "0", "1", "2", "3", "4"]


def test_select_consolidatable_min_age_excludes_too_young_memories() -> None:
    # 5 eligible memories, oldest (0) to newest (4). min_age=2 keeps the 2 newest
    # (3, 4) out of the foldable pool -- a rolling recent window.
    memories = _aged_memories(5)

    selected = select_consolidatable(memories, importance_ceiling=3, min_age=2)

    assert [memory.id for memory in selected] == ["0", "1", "2"]


def test_select_consolidatable_min_age_larger_than_pool_excludes_everything() -> None:
    memories = _aged_memories(3)

    selected = select_consolidatable(memories, importance_ceiling=3, min_age=10)

    assert selected == []


def test_select_consolidatable_batch_cap_folds_only_oldest_n() -> None:
    memories = _aged_memories(6)

    selected = select_consolidatable(memories, importance_ceiling=3, batch_cap=2)

    assert [memory.id for memory in selected] == ["0", "1"]


def test_select_consolidatable_batch_cap_and_min_age_combine() -> None:
    # 6 eligible memories (0 oldest .. 5 newest). min_age=2 drops the 2 newest (4, 5)
    # from the foldable pool, leaving [0, 1, 2, 3]; batch_cap=2 then folds only the
    # oldest 2 of that pool, keeping the rolling recent window (2, 3, 4, 5) intact.
    memories = _aged_memories(6)

    selected = select_consolidatable(memories, importance_ceiling=3, min_age=2, batch_cap=2)

    assert [memory.id for memory in selected] == ["0", "1"]


def test_select_consolidatable_batch_cap_larger_than_pool_is_a_no_op() -> None:
    memories = _aged_memories(3)

    selected = select_consolidatable(memories, importance_ceiling=3, batch_cap=100)

    assert [memory.id for memory in selected] == ["0", "1", "2"]


def test_deterministic_consolidated_summary_combines_text() -> None:
    memories = [_episode("a", importance=1), _episode("b", importance=1)]

    summary = deterministic_consolidated_summary(memories)

    assert "Event a" in summary
    assert "Event b" in summary
