from __future__ import annotations

from datetime import datetime, timezone

from app.domain import MemoryEpisode, Visibility
from app.orchestration.canon_builder import (
    CANON_TAGS,
    CANON_TAGS_DE,
    build_standing_facts,
    effective_canon_tags,
)


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


# --- docs/26 §3.3 Lane A tag-eligible canon pinning (#78) ------------------------


def test_flag_off_is_byte_identical_golden_two_promise_facts() -> None:
    """docs/26 §6 Stage 3 golden test (fix 2's regression case). Fixture pool has
    TWO distinct, same-tag-family (promise) deterministic-extractor-shaped facts,
    both already floor-eligible at importance=4 under TODAY's unchanged predicate.
    With CANON_TAG_PINNING=false (the default, not passed here), build_standing_facts
    must behave exactly as it did before this stage: no eligibility change, no
    supersession logic runs at all, same order, same text, no warnings."""
    memories = [
        _memory(
            "vault",
            summary="The player promised to guard the eastern vault before midnight.",
            importance=4,
            tags=["promise"],
            created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        ),
        _memory(
            "bridge",
            summary="The player promised to meet at the northern bridge before sunrise.",
            importance=4,
            tags=["promise"],
            created_at=datetime(2026, 6, 2, tzinfo=timezone.utc),
        ),
    ]
    warnings: list[str] = []

    facts = build_standing_facts(
        memories, importance_floor=4, max_items=8, max_chars=900, warnings=warnings
    )

    assert facts == (
        "The player promised to meet at the northern bridge before sunrise.",
        "The player promised to guard the eastern vault before midnight.",
    )
    assert warnings == []


def test_safeguard_never_fires_when_flag_is_off_even_on_amendment_fixture() -> None:
    """The safeguard must NOT fire regardless of fixture when canon_tag_pinning is
    False -- proven here with a fixture that WOULD trigger it if the flag were on
    (see test_safeguard_drops_older_entry_on_strict_term_superset_same_family_newer).
    importance_floor=1 (not the real default of 4) isolates this test to "does the
    safeguard fire," separate from "does the widened eligibility predicate apply."
    """
    memories = [
        _memory(
            "seal-old",
            summary="The trust rule is a blue wax seal.",
            importance=2,
            tags=["rule"],
            created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        ),
        _memory(
            "seal-new",
            summary="The trust rule is now a blue wax seal plus a knock at the door.",
            importance=2,
            tags=["rule"],
            created_at=datetime(2026, 6, 2, tzinfo=timezone.utc),
        ),
    ]
    warnings: list[str] = []

    facts = build_standing_facts(
        memories, importance_floor=1, max_items=8, max_chars=900, warnings=warnings
    )

    assert facts == (
        "The trust rule is now a blue wax seal plus a knock at the door.",
        "The trust rule is a blue wax seal.",
    )
    assert warnings == []


def test_flag_on_widens_eligibility_to_sub_floor_tagged_memory() -> None:
    """The blue-seal shape (docs/26 §3.3): tag-eligible but sub-floor importance."""
    memories = [
        _memory(
            "blue-seal",
            summary="The player will only trust messages with a blue wax seal.",
            importance=2,
            tags=["rule"],
        ),
    ]

    facts_flag_off = build_standing_facts(memories, importance_floor=4, max_items=8, max_chars=900)
    facts_flag_on = build_standing_facts(
        memories, importance_floor=4, max_items=8, max_chars=900, canon_tag_pinning=True
    )

    assert facts_flag_off == ()
    assert facts_flag_on == ("The player will only trust messages with a blue wax seal.",)


def test_flag_on_does_not_pin_untagged_sub_floor_memory() -> None:
    memories = [
        _memory("mood", summary="The gallery smelled of roses.", importance=2, tags=["mood"]),
    ]

    facts = build_standing_facts(
        memories, importance_floor=4, max_items=8, max_chars=900, canon_tag_pinning=True
    )

    assert facts == ()


def test_flag_on_does_not_pin_gm_visibility_memory() -> None:
    memories = [
        _memory(
            "gm",
            summary="Hidden GM fact about the rule.",
            importance=2,
            visibility=Visibility.GM,
            tags=["rule"],
        ),
    ]

    facts = build_standing_facts(
        memories, importance_floor=4, max_items=8, max_chars=900, canon_tag_pinning=True
    )

    assert facts == ()


def test_effective_canon_tags_widens_only_when_flag_on() -> None:
    assert effective_canon_tags(canon_tag_pinning=False) == CANON_TAGS
    assert effective_canon_tags(canon_tag_pinning=True) == CANON_TAGS | CANON_TAGS_DE
    assert "versprechen" in CANON_TAGS_DE
    assert "versprechen" not in CANON_TAGS


def test_flag_on_makes_german_tagged_sub_floor_memory_eligible() -> None:
    memories = [
        _memory(
            "versprechen",
            summary="Der Spieler versprach, vor Sonnenaufgang zurueckzukehren.",
            importance=2,
            tags=["versprechen"],
        ),
    ]

    facts_flag_off = build_standing_facts(memories, importance_floor=4, max_items=8, max_chars=900)
    facts_flag_on = build_standing_facts(
        memories, importance_floor=4, max_items=8, max_chars=900, canon_tag_pinning=True
    )

    assert facts_flag_off == ()
    assert facts_flag_on == ("Der Spieler versprach, vor Sonnenaufgang zurueckzukehren.",)


# --- docs/26 §3.3.1 stale-fact safeguard (flag-gated) -----------------------------


def test_safeguard_drops_older_entry_on_strict_term_superset_same_family_newer() -> None:
    """The constructed amendment case: docs/26's own "blue seal only" superseded by
    "blue seal plus a knock" example. Must fire (only) with the flag on, and the
    drop must be auditable via the warnings out-param."""
    memories = [
        _memory(
            "seal-old",
            summary="The trust rule is a blue wax seal.",
            importance=2,
            tags=["rule"],
            created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        ),
        _memory(
            "seal-new",
            summary="The trust rule is now a blue wax seal plus a knock at the door.",
            importance=2,
            tags=["rule"],
            created_at=datetime(2026, 6, 2, tzinfo=timezone.utc),
        ),
    ]
    warnings: list[str] = []

    facts = build_standing_facts(
        memories,
        importance_floor=4,
        max_items=8,
        max_chars=900,
        canon_tag_pinning=True,
        warnings=warnings,
    )

    assert facts == ("The trust rule is now a blue wax seal plus a knock at the door.",)
    assert len(warnings) == 1
    assert "stale canon fact superseded" in warnings[0]
    assert "rule" in warnings[0]
    assert "blue wax seal" in warnings[0]


def test_safeguard_keeps_both_on_partial_overlap_not_a_superset() -> None:
    memories = [
        _memory(
            "a",
            summary="The trust rule is a blue wax seal.",
            importance=2,
            tags=["rule"],
            created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        ),
        _memory(
            "b",
            summary="The trust rule is a crimson ribbon instead.",
            importance=2,
            tags=["rule"],
            created_at=datetime(2026, 6, 2, tzinfo=timezone.utc),
        ),
    ]
    warnings: list[str] = []

    facts = build_standing_facts(
        memories,
        importance_floor=4,
        max_items=8,
        max_chars=900,
        canon_tag_pinning=True,
        warnings=warnings,
    )

    assert len(facts) == 2
    assert warnings == []


def test_safeguard_keeps_both_on_equal_term_sets() -> None:
    memories = [
        _memory(
            "a",
            summary="The trust rule is a blue wax seal.",
            importance=2,
            tags=["rule"],
            created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        ),
        _memory(
            "b",
            summary="A blue wax seal is the trust rule.",  # same content terms, reordered
            importance=2,
            tags=["rule"],
            created_at=datetime(2026, 6, 2, tzinfo=timezone.utc),
        ),
    ]
    warnings: list[str] = []

    facts = build_standing_facts(
        memories,
        importance_floor=4,
        max_items=8,
        max_chars=900,
        canon_tag_pinning=True,
        warnings=warnings,
    )

    assert len(facts) == 2
    assert warnings == []


def test_safeguard_keeps_both_on_cross_family_tags() -> None:
    memories = [
        _memory(
            "a",
            summary="The trust rule is a blue wax seal.",
            importance=2,
            tags=["rule"],
            created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        ),
        _memory(
            "b",
            # Strict term superset of "a" AND newer -- but a DIFFERENT tag family,
            # so the safeguard must not treat it as an amendment of "a".
            summary="The trust rule is a blue wax seal plus a knock at the door.",
            importance=2,
            tags=["promise"],
            created_at=datetime(2026, 6, 2, tzinfo=timezone.utc),
        ),
    ]
    warnings: list[str] = []

    facts = build_standing_facts(
        memories,
        importance_floor=4,
        max_items=8,
        max_chars=900,
        canon_tag_pinning=True,
        warnings=warnings,
    )

    assert len(facts) == 2
    assert warnings == []


def test_safeguard_keeps_both_when_either_side_missing_created_at() -> None:
    memories = [
        _memory(
            "a",
            summary="The trust rule is a blue wax seal.",
            importance=2,
            tags=["rule"],
            created_at=None,
        ),
        _memory(
            "b",
            summary="The trust rule is a blue wax seal plus a knock at the door.",
            importance=2,
            tags=["rule"],
            created_at=datetime(2026, 6, 2, tzinfo=timezone.utc),
        ),
    ]
    warnings: list[str] = []

    facts = build_standing_facts(
        memories,
        importance_floor=4,
        max_items=8,
        max_chars=900,
        canon_tag_pinning=True,
        warnings=warnings,
    )

    assert len(facts) == 2
    assert warnings == []
