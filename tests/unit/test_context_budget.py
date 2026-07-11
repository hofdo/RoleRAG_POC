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

