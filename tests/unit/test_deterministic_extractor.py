from __future__ import annotations

from app.domain import Visibility
from app.memory.deterministic_extractor import (
    extract_explicit_durable_events,
    is_covered_by_summaries,
)


def test_extracts_first_person_promise_with_deadline() -> None:
    candidates = extract_explicit_durable_events(
        user_message=(
            "I promise to return before dawn if you keep the archive door unbarred."
        ),
        scene_id="rose-gallery",
        actor_id="archivist",
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert "return before dawn" in candidate.summary
    assert "archive door unbarred" in candidate.summary
    assert candidate.visibility == Visibility.PLAYER
    assert candidate.importance == 4
    assert "promise" in candidate.tags
    assert candidate.scene_id == "rose-gallery"
    assert candidate.actor_id == "archivist"


def test_extracts_entrusted_item() -> None:
    candidates = extract_explicit_durable_events(
        user_message="I entrust my silver compass to you until I return.",
        scene_id="rose-gallery",
        actor_id="archivist",
    )

    assert len(candidates) == 1
    assert "silver compass" in candidates[0].summary
    assert "entrusted" in candidates[0].tags


def test_ignores_greetings_and_filler() -> None:
    assert (
        extract_explicit_durable_events(
            user_message="Good evening, archivist. The roses look lovely tonight.",
            scene_id="rose-gallery",
            actor_id="archivist",
        )
        == []
    )


def test_ignores_third_person_promise_mentions() -> None:
    assert (
        extract_explicit_durable_events(
            user_message="She promised the regent nothing, or so the courtiers say.",
            scene_id="rose-gallery",
            actor_id="archivist",
        )
        == []
    )


def test_extracts_only_the_trigger_sentence_from_longer_messages() -> None:
    candidates = extract_explicit_durable_events(
        user_message=(
            "The gallery feels colder tonight. I promise to bring the cipher ledger "
            "tomorrow. Tell me about the regent."
        ),
        scene_id="rose-gallery",
        actor_id="archivist",
    )

    assert len(candidates) == 1
    assert "cipher ledger" in candidates[0].summary
    assert "regent" not in candidates[0].summary
    assert "colder" not in candidates[0].summary


def test_coverage_detects_paraphrased_duplicate() -> None:
    assert is_covered_by_summaries(
        "Player commitment: I promise to return before dawn.",
        ["The player promised to return before dawn."],
    )


def test_coverage_rejects_unrelated_summary() -> None:
    assert not is_covered_by_summaries(
        "Player commitment: I promise to return before dawn.",
        ["The player admired the rose arrangements in the gallery."],
    )
