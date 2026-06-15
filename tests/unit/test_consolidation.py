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


def test_deterministic_consolidated_summary_combines_text() -> None:
    memories = [_episode("a", importance=1), _episode("b", importance=1)]

    summary = deterministic_consolidated_summary(memories)

    assert "Event a" in summary
    assert "Event b" in summary
