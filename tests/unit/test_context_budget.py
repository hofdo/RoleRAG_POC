from __future__ import annotations

from app.domain import RetrievedChunk, Visibility
from app.orchestration.context_budget import ContextBudget, select_retrieved_chunks_for_prompt


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

