from __future__ import annotations

from app.domain import RetrievedChunk, Visibility
from app.llm.provider import LlmMessage
from app.orchestration.context_budget import (
    ContextBudget,
    _truncate_text,
    context_preflight_warning,
    context_usage_warning,
    estimate_prompt_tokens,
    select_retrieved_chunks_for_prompt,
)


def _chunk(
    chunk_id: str,
    *,
    text: str,
    visibility: Visibility = Visibility.PLAYER,
) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        source="demo.md",
        source_type="lore",
        text=text,
        score=0.9,
        visibility=visibility,
    )


def test_select_retrieved_chunks_filters_deduplicates_truncates_and_limits() -> None:
    chunks = [
        _chunk("player-1", text="A" * 20),
        _chunk("gm-1", text="hidden", visibility=Visibility.GM),
        _chunk("player-1", text="duplicate"),
        _chunk("private-1", text="private", visibility=Visibility.CHARACTER_PRIVATE),
        _chunk("player-2", text="short"),
        _chunk("player-3", text="excluded by limit"),
    ]

    selected = select_retrieved_chunks_for_prompt(
        chunks,
        budget=ContextBudget(retrieved_chunks=2, max_retrieved_chunk_chars=10),
    )

    assert [chunk.id for chunk in selected] == ["player-1", "player-2"]
    assert selected[0].text == "AAAAAAA..."
    assert selected[1].text == "short"


def test_select_excludes_standing_fact_duplicates_and_recovers_the_slot() -> None:
    # docs/22 C1: a durable fact pinned into Standing facts also wins rerank, so it
    # appears in the retrieved set too. Excluding it must not shrink the block below
    # budget -- the caller over-fetches so the freed slot is filled by a distinct fact.
    chunks = [
        _chunk("mem-promise", text="The player promised to guard the northern gate."),
        _chunk("mem-distinct", text="The regent distrusts the chancellor."),
        _chunk("mem-extra", text="A storm is expected by nightfall."),
    ]

    selected = select_retrieved_chunks_for_prompt(
        chunks,
        budget=ContextBudget(retrieved_chunks=2, max_retrieved_chunk_chars=800),
        # Normalized match: differing case + whitespace still excludes the duplicate.
        exclude_texts=["the player promised to  guard the  NORTHERN gate."],
    )

    assert [chunk.id for chunk in selected] == ["mem-distinct", "mem-extra"]


def test_select_dedupes_duplicate_text_distinct_ids_and_recovers_the_slot() -> None:
    # docs/22 N3: with semantic write-dedup off, near-identical memories can
    # accumulate under distinct ids. Duplicate normalized text must not co-fill
    # slots; first occurrence (highest rank) wins and the freed slot goes to the
    # next distinct chunk.
    chunks = [
        _chunk("mem-a", text="The regent distrusts the chancellor."),
        _chunk("mem-b", text="  The   regent distrusts THE chancellor.  "),
        _chunk("mem-c", text="A storm is expected by nightfall."),
    ]

    selected = select_retrieved_chunks_for_prompt(
        chunks,
        budget=ContextBudget(retrieved_chunks=2, max_retrieved_chunk_chars=800),
    )

    assert [chunk.id for chunk in selected] == ["mem-a", "mem-c"]


def test_select_leaves_legitimately_distinct_chunks_untouched() -> None:
    chunks = [
        _chunk("mem-a", text="The regent distrusts the chancellor."),
        _chunk("mem-b", text="The chancellor distrusts the regent."),
        _chunk("mem-c", text="A storm is expected by nightfall."),
    ]

    selected = select_retrieved_chunks_for_prompt(
        chunks,
        budget=ContextBudget(retrieved_chunks=3, max_retrieved_chunk_chars=800),
    )

    assert [chunk.id for chunk in selected] == ["mem-a", "mem-b", "mem-c"]


def test_select_text_dedup_interacts_with_standing_facts_exclusion() -> None:
    # A chunk excluded by the C1 standing-facts match must not block a later,
    # genuinely distinct duplicate-text pair from also being deduped by N3.
    chunks = [
        _chunk("mem-promise", text="The player promised to guard the northern gate."),
        _chunk("mem-distinct", text="The regent distrusts the chancellor."),
        _chunk("mem-distinct-dup", text="the regent distrusts THE chancellor."),
        _chunk("mem-extra", text="A storm is expected by nightfall."),
    ]

    selected = select_retrieved_chunks_for_prompt(
        chunks,
        budget=ContextBudget(retrieved_chunks=3, max_retrieved_chunk_chars=800),
        exclude_texts=["the player promised to  guard the  NORTHERN gate."],
    )

    assert [chunk.id for chunk in selected] == ["mem-distinct", "mem-extra"]


def test_select_dedupes_on_post_truncation_text_when_pre_truncation_text_differs() -> None:
    # docs/22 P5 (fixed 2026-07-11): two chunks sharing an 850-char prefix but
    # diverging after it are distinct pre-truncation, so the N3 pre-truncation check
    # alone lets both through -- but at an 800-char cap, _truncate_text renders them
    # byte-identical prompt blocks. The post-truncation check must catch this.
    shared_prefix = "The regent distrusts the chancellor and plans to act soon. " * 14
    shared_prefix = shared_prefix[:850]
    text_a = shared_prefix + "A" * 150  # 1000 chars total
    text_b = shared_prefix + "B" * 150  # 1000 chars total, diverges only after 850
    assert text_a != text_b  # pre-truncation texts are genuinely distinct

    chunks = [
        _chunk("mem-a", text=text_a),
        _chunk("mem-b", text=text_b),
        _chunk("mem-c", text="A storm is expected by nightfall."),
    ]

    selected = select_retrieved_chunks_for_prompt(
        chunks,
        budget=ContextBudget(retrieved_chunks=2, max_retrieved_chunk_chars=800),
    )

    # mem-b renders identically to the already-selected mem-a once truncated to 800
    # chars, so it must be dropped and the freed slot filled by the next distinct
    # chunk (mem-c) -- not left duplicated in the prompt.
    assert [chunk.id for chunk in selected] == ["mem-a", "mem-c"]
    assert selected[0].text != text_a  # confirms truncation actually happened
    assert len(selected[0].text) <= 800


def test_select_leaves_distinct_post_truncation_text_untouched() -> None:
    # Chunks that remain distinct after truncation (diverge within the cap) must not
    # be affected by the post-truncation dedup check.
    chunks = [
        _chunk("mem-a", text="The regent distrusts the chancellor."),
        _chunk("mem-b", text="The chancellor distrusts the regent."),
    ]

    selected = select_retrieved_chunks_for_prompt(
        chunks,
        budget=ContextBudget(retrieved_chunks=2, max_retrieved_chunk_chars=800),
    )

    assert [chunk.id for chunk in selected] == ["mem-a", "mem-b"]


def test_select_exclude_texts_empty_is_byte_identical() -> None:
    chunks = [_chunk("player-1", text="only fact")]
    baseline = select_retrieved_chunks_for_prompt(
        chunks, budget=ContextBudget(retrieved_chunks=5, max_retrieved_chunk_chars=800)
    )
    with_empty = select_retrieved_chunks_for_prompt(
        chunks,
        budget=ContextBudget(retrieved_chunks=5, max_retrieved_chunk_chars=800),
        exclude_texts=(),
    )
    assert [c.model_dump() for c in baseline] == [c.model_dump() for c in with_empty]


# -- docs/22 P0.3: sentence-boundary chunk trimming ------------------------------------


def test_truncate_text_under_cap_is_byte_identical() -> None:
    text = "The regent distrusts the chancellor."
    assert _truncate_text(text, 800) == text
    assert _truncate_text(text, len(text)) == text


def test_truncate_text_trims_at_last_sentence_boundary() -> None:
    text = "The regent distrusts the chancellor. He plans to expose the forged ledger soon."
    # Cap lands inside the second sentence; the trim should back up to the sentence
    # boundary after "chancellor." rather than hard-cutting mid-clause.
    result = _truncate_text(text, 60)
    assert result == "The regent distrusts the chancellor...."
    assert result.startswith("The regent distrusts the chancellor.")
    assert len(result) <= 60


def test_truncate_text_falls_back_to_word_boundary_without_sentence_punctuation() -> None:
    text = "the quick brown fox jumps over the lazy dog near the riverbank"
    result = _truncate_text(text, 30)
    assert result == "the quick brown fox jumps..."
    assert len(result) <= 30
    # No sentence punctuation anywhere, so the fallback must be a clean word boundary.
    assert not result[:-3].endswith(" ")


def test_truncate_text_hard_cuts_pathological_text_with_no_boundary() -> None:
    # No spaces and no sentence punctuation at all within the budget window.
    text = "A" * 20
    assert _truncate_text(text, 10) == "AAAAAAA..."


def test_truncate_text_keeps_omission_marker_when_trimmed() -> None:
    text = "word " * 200
    result = _truncate_text(text, 50)
    assert len(text) > 50
    assert result.endswith("...")
    assert len(result) <= 50


def test_truncate_text_tiny_cap_returns_dots_only() -> None:
    text = "The regent distrusts the chancellor."
    assert _truncate_text(text, 3) == "..."
    assert _truncate_text(text, 1) == "."


# --- Boundary-aware trimming (#69) -------------------------------------------------


def test_truncate_text_unchanged_when_within_budget() -> None:
    assert _truncate_text("short text", 800) == "short text"


def test_truncate_text_cuts_at_the_exact_fit_boundary() -> None:
    text = "exactly ten"  # len == 11
    assert _truncate_text(text, len(text)) == text


def test_truncate_text_cuts_at_last_word_boundary_not_mid_word() -> None:
    text = "The regent fears open daylight and hidden passageways"
    result = _truncate_text(text, 30)
    assert len(result) <= 30
    assert result.endswith("...")
    # Cutting at a word boundary means the retained text (sans "...") is a
    # whitespace-delimited prefix of the original -- never a sliced-off word.
    retained = result[: -len("...")]
    assert text.startswith(retained)
    assert not retained.endswith(" ")
    next_char_index = len(retained)
    assert next_char_index == len(text) or text[next_char_index] == " "


def test_truncate_text_falls_back_to_hard_cut_when_no_space_in_budget() -> None:
    # A single unbroken run of characters longer than the budget: no word boundary
    # exists within budget, so this must degrade to the old hard-cut behavior
    # instead of collapsing to an (almost) empty string.
    text = "A" * 50
    result = _truncate_text(text, 10)
    assert result == "AAAAAAA..."
    assert len(result) == 10


# --- Sentence-trim retention floor (cross-review P1, 2026-07-11) -------------------


def test_truncate_text_early_terminator_does_not_collapse_the_chunk() -> None:
    # A leading abbreviation ("Mr.") is a sentence boundary a few characters in.
    # Winning unconditionally would collapse an entire 960-char chunk to "Mr...."
    # while still consuming one of the prompt's scarce retrieved-chunk slots. The
    # word-boundary fallback would fail identically here (the only space in the
    # text is the one right after "Mr."), so the fix must fall all the way through
    # to a hard cut at the full budget rather than stopping at the first tier that
    # merely isn't the sentence boundary.
    text = "Mr. " + "x" * 960
    result = _truncate_text(text, 200)
    assert result != "Mr...."
    assert len(result) == 200
    assert result.startswith("Mr. xxx")
    assert result.endswith("...")


def test_truncate_text_sentence_boundary_wins_at_exactly_half_budget() -> None:
    # budget = max_chars - 3 = 20; a sentence boundary retaining exactly half
    # (10 chars) must still win the tie (">=", not ">").
    text = "AAAAAAAAA. " + "b" * 40
    result = _truncate_text(text, 23)
    assert result == "AAAAAAAAA...."
    assert len(result) <= 23


def test_truncate_text_sentence_boundary_below_half_budget_falls_through() -> None:
    # Same shape as the exact-half case but the boundary now retains one char
    # less than half the budget, so it must lose to a later, larger word/hard-cut
    # tier instead of winning on a stale ">" comparison.
    text = "AAAAAAAA. " + "b" * 40
    result = _truncate_text(text, 23)
    assert result != "AAAAAAAA...."
    assert len(result) <= 23


# --- Context-ceiling preflight estimator (#69) --------------------------------------


def test_estimate_prompt_tokens_uses_chars_over_four_plus_max_tokens() -> None:
    messages = [
        LlmMessage(role="system", content="x" * 40),
        LlmMessage(role="user", content="y" * 20),
    ]
    assert estimate_prompt_tokens(messages, max_tokens=100) == (60 // 4) + 100


def test_context_preflight_warning_disabled_when_window_is_zero() -> None:
    messages = [LlmMessage(role="user", content="x" * 100_000)]
    assert (
        context_preflight_warning(
            messages, max_tokens=500, context_window_tokens=0, warn_ratio=0.85
        )
        is None
    )


def test_context_preflight_warning_fires_above_threshold() -> None:
    messages = [LlmMessage(role="user", content="x" * 4000)]  # ~1000 estimated tokens
    warning = context_preflight_warning(
        messages, max_tokens=0, context_window_tokens=1000, warn_ratio=0.85
    )
    assert warning is not None
    assert "context preflight: estimated 1000 tokens vs window 1000" in warning


def test_context_preflight_warning_silent_below_threshold() -> None:
    messages = [LlmMessage(role="user", content="x" * 40)]  # ~10 estimated tokens
    assert (
        context_preflight_warning(
            messages, max_tokens=0, context_window_tokens=1000, warn_ratio=0.85
        )
        is None
    )


def test_context_usage_warning_disabled_when_window_is_zero() -> None:
    assert (
        context_usage_warning(
            {"prompt_tokens": 10_000}, context_window_tokens=0, warn_ratio=0.85
        )
        is None
    )


def test_context_usage_warning_ignores_missing_or_zero_prompt_tokens() -> None:
    assert (
        context_usage_warning({}, context_window_tokens=1000, warn_ratio=0.85) is None
    )
    assert (
        context_usage_warning(
            {"prompt_tokens": 0}, context_window_tokens=1000, warn_ratio=0.85
        )
        is None
    )


def test_context_usage_warning_fires_when_actual_prompt_tokens_exceed_ratio() -> None:
    warning = context_usage_warning(
        {"prompt_tokens": 900}, context_window_tokens=1000, warn_ratio=0.85
    )
    assert warning is not None
    assert "actual prompt_tokens 900" in warning


def test_context_usage_warning_silent_below_ratio() -> None:
    assert (
        context_usage_warning(
            {"prompt_tokens": 100}, context_window_tokens=1000, warn_ratio=0.85
        )
        is None
    )

