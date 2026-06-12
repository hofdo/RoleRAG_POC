from __future__ import annotations

from app.orchestration.draft_validator import DraftValidationResult, validate_draft

VISIBLE_TEXTS = [
    "The Rose Gallery sits inside the Winter Palace.",
    "Iria the archivist guards the winter ledgers.",
    "The regent fears open daylight.",
]


def test_known_entities_and_plain_description_pass() -> None:
    result = validate_draft(
        draft="Iria gestures at the Rose Gallery. The regent is not spoken of here.",
        player_message="I look around the gallery.",
        visible_texts=VISIBLE_TEXTS,
    )

    assert result.unsupported_entities == ()
    assert result.evaded_action is False
    assert result.flags == []


def test_invented_named_entity_is_flagged() -> None:
    result = validate_draft(
        draft="Duke Erran left a silver map with Iria before vanishing.",
        player_message="What happened here?",
        visible_texts=VISIBLE_TEXTS,
    )

    assert "Duke Erran" in result.unsupported_entities
    assert any("unsupported entity" in flag for flag in result.flags)


def test_sentence_initial_capitalised_words_are_not_entities() -> None:
    result = validate_draft(
        draft="Quiet falls over the gallery. Nothing moves in the corridor.",
        player_message="I listen carefully.",
        visible_texts=VISIBLE_TEXTS,
    )

    assert result.unsupported_entities == ()


def test_entity_known_through_possessive_in_context_passes() -> None:
    result = validate_draft(
        draft="Iria keeps the key.",
        player_message="Who keeps the key?",
        visible_texts=["Iria's key opens the archive."],
    )

    assert result.unsupported_entities == ()


def test_direct_question_with_unrelated_draft_is_flagged_as_evaded() -> None:
    result = validate_draft(
        draft="The chandeliers glitter softly above the marble floor.",
        player_message="What did the regent promise you before dawn?",
        visible_texts=VISIBLE_TEXTS,
    )

    assert result.evaded_action is True
    assert any("unaddressed" in flag for flag in result.flags)


def test_direct_question_with_responsive_draft_passes() -> None:
    result = validate_draft(
        draft="The regent promised nothing; promises mean little before dawn.",
        player_message="What did the regent promise you before dawn?",
        visible_texts=VISIBLE_TEXTS,
    )

    assert result.evaded_action is False


def test_player_action_statement_requires_acknowledgement() -> None:
    result = validate_draft(
        draft="A draft of cold air slips through the gallery.",
        player_message="I hand Iria the sealed cipher letter.",
        visible_texts=VISIBLE_TEXTS,
    )

    assert result.evaded_action is True


def test_terse_single_term_question_is_not_evaded() -> None:
    result = validate_draft(
        draft="The gallery rests in silence beneath the mirrors.",
        player_message="What news?",
        visible_texts=VISIBLE_TEXTS,
    )

    assert result.evaded_action is False


def test_smalltalk_without_action_is_never_evaded() -> None:
    result = validate_draft(
        draft="The gallery rests in silence.",
        player_message="Hm.",
        visible_texts=VISIBLE_TEXTS,
    )

    assert result.evaded_action is False


def test_result_flags_combine_both_checks() -> None:
    result = validate_draft(
        draft="Lord Maren shrugs at the painted ceiling.",
        player_message="Where is the cipher key hidden?",
        visible_texts=VISIBLE_TEXTS,
    )

    assert isinstance(result, DraftValidationResult)
    assert "Lord Maren" in result.unsupported_entities
    assert result.evaded_action is True
    assert len(result.flags) == 2
