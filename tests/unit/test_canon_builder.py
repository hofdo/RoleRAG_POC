from __future__ import annotations

from datetime import datetime, timezone

from app.domain import MemoryEpisode, Visibility
from app.orchestration.canon_builder import build_standing_facts


def _memory(
    memory_id: str,
    *,
    summary: str,
    importance: int = 4,
    visibility: Visibility = Visibility.PLAYER,
    tags: list[str] | None = None,
    created_at: datetime | None = None,
) -> MemoryEpisode:
    return MemoryEpisode(
        id=memory_id,
        session_id="session-1",
        scene_id="rose-gallery",
        actor_id="archivist",
        summary=summary,
        importance=importance,
        visibility=visibility,
        tags=tags if tags is not None else ["promise"],
        created_at=created_at,
    )


def test_filters_to_player_visible_canon_tags_above_floor() -> None:
    memories = [
        _memory("keep", summary="The player promised to return before dawn.", tags=["promise"]),
        _memory("gm", summary="Hidden GM fact.", visibility=Visibility.GM, tags=["rule"]),
        _memory("no-tag", summary="The player admired the roses.", tags=["mood"]),
        _memory("low", summary="A minor agreement.", importance=2, tags=["agreement"]),
    ]

    facts = build_standing_facts(
        memories, importance_floor=3, max_items=8, max_chars=900
    )

    assert facts == ("The player promised to return before dawn.",)


def test_pinned_facts_lead_and_dedupe_against_derived() -> None:
    memories = [
        _memory("m", summary="The player promised to return before dawn.", tags=["promise"]),
    ]

    facts = build_standing_facts(
        memories,
        pinned=[
            "Author rule: the east gate stays shut.",
            "The player promised to return before dawn.",  # duplicate of the derived fact
        ],
        importance_floor=1,
        max_items=8,
        max_chars=900,
    )

    assert facts[0] == "Author rule: the east gate stays shut."
    assert facts.count("The player promised to return before dawn.") == 1


def test_orders_by_importance_then_recency() -> None:
    memories = [
        _memory(
            "old-high",
            summary="Rule: never open the west door.",
            importance=5,
            tags=["rule"],
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
        _memory(
            "new-low",
            summary="The player agreed to a three-tap signal.",
            importance=4,
            tags=["signal"],
            created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        ),
        _memory(
            "new-high",
            summary="The player vowed to protect the archive.",
            importance=5,
            tags=["vow"],
            created_at=datetime(2026, 6, 5, tzinfo=timezone.utc),
        ),
    ]

    facts = build_standing_facts(
        memories, importance_floor=1, max_items=8, max_chars=900
    )

    assert facts == (
        "The player vowed to protect the archive.",
        "Rule: never open the west door.",
        "The player agreed to a three-tap signal.",
    )


def test_dedups_by_summary() -> None:
    memories = [
        _memory("a", summary="The player promised to return before dawn."),
        _memory("b", summary="the player promised to return before dawn."),
    ]

    facts = build_standing_facts(
        memories, importance_floor=1, max_items=8, max_chars=900
    )

    assert facts == ("The player promised to return before dawn.",)


def test_caps_items_and_chars() -> None:
    memories = [
        _memory(f"m{i}", summary=f"Durable commitment number {i}.", importance=4)
        for i in range(5)
    ]

    capped_items = build_standing_facts(
        memories, importance_floor=1, max_items=2, max_chars=900
    )
    assert len(capped_items) == 2

    capped_chars = build_standing_facts(
        memories, importance_floor=1, max_items=8, max_chars=30
    )
    assert sum(len(fact) for fact in capped_chars) <= 30


def test_handles_none_created_at_without_crashing() -> None:
    memories = [
        _memory(
            "dated",
            summary="Dated commitment.",
            created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        ),
        _memory("undated", summary="Undated commitment.", created_at=None),
    ]

    facts = build_standing_facts(
        memories, importance_floor=1, max_items=8, max_chars=900
    )

    # Dated memory outranks the undated one (missing timestamp sorts oldest).
    assert facts == ("Dated commitment.", "Undated commitment.")
