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

