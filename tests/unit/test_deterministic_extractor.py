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


def test_extracts_mutual_agreement_signal() -> None:
    candidates = extract_explicit_durable_events(
        user_message="We agreed on a three-tap signal at the side door.",
        scene_id="rose-gallery",
        actor_id="archivist",
    )

    assert len(candidates) == 1
    assert "three-tap signal" in candidates[0].summary
    assert "agreement" in candidates[0].tags


def test_extracts_established_password_rule() -> None:
    candidates = extract_explicit_durable_events(
        user_message="The password is ravenwood.",
        scene_id="rose-gallery",
        actor_id="archivist",
    )

    assert len(candidates) == 1
    assert "ravenwood" in candidates[0].summary
    assert "agreement" in candidates[0].tags


def test_extracts_negative_commitment() -> None:
    candidates = extract_explicit_durable_events(
        user_message="I will never tell anyone what you hide in the vault.",
        scene_id="rose-gallery",
        actor_id="archivist",
    )

    assert len(candidates) == 1
    assert "vault" in candidates[0].summary
    assert "promise" in candidates[0].tags


def test_extracts_you_have_my_word() -> None:
    candidates = extract_explicit_durable_events(
        user_message="You have my word on the matter of the ledger.",
        scene_id="rose-gallery",
        actor_id="archivist",
    )

    assert len(candidates) == 1
    assert "promise" in candidates[0].tags


def test_ignores_plain_future_actions() -> None:
    # "I will/I'll <verb>" without a commitment verb or negation is normal
    # narration, not a durable event.
    assert (
        extract_explicit_durable_events(
            user_message="I will tell you about the garden. I'll look around the room.",
            scene_id="rose-gallery",
            actor_id="archivist",
        )
        == []
    )


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


def test_reversal_candidate_is_not_covered_by_the_fact_it_contradicts() -> None:
    assert not is_covered_by_summaries(
        "Mira no longer trusts the player and revoked the safehouse offer",
        ["Mira trusts the player and offered the player the safehouse"],
    )


def test_identical_summary_is_still_covered() -> None:
    assert is_covered_by_summaries(
        "The player promised to guard the ledger",
        ["The player promised to guard the ledger"],
    )


def test_reversal_present_in_both_summaries_is_still_covered() -> None:
    assert is_covered_by_summaries(
        "Mira no longer trusts the player",
        ["Mira no longer trusts the player and revoked the safehouse offer"],
    )
