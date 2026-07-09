from __future__ import annotations

from app.domain import RetrievedChunk, Visibility
from app.orchestration.context_budget import (
    ContextBudget,
    _truncate_text,
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

