from __future__ import annotations

from collections.abc import Collection, Sequence

from pydantic import BaseModel, Field

from app.domain import RetrievedChunk, Visibility


class ContextBudget(BaseModel):
    retrieved_chunks: int = Field(default=5, ge=0)
    max_retrieved_chunk_chars: int = Field(default=800, ge=1)


def select_retrieved_chunks_for_prompt(
    chunks: Sequence[RetrievedChunk],
    *,
    budget: ContextBudget,
    exclude_texts: Collection[str] = (),
) -> list[RetrievedChunk]:
    """Pick the PLAYER-visible retrieved chunks for the actor prompt.

    ``exclude_texts`` drops chunks whose (normalized) text already appears in the
    Standing-facts block: a high-importance durable memory is pinned there verbatim
    *and* wins rerank, so without this it double-spends one of only
    ``budget.retrieved_chunks`` slots, evicting a distinct fact (docs/22 C1). The
    caller over-fetches so the freed slot is filled by the next distinct chunk.
    """
    excluded = {_normalize_for_match(text) for text in exclude_texts}
    selected: list[RetrievedChunk] = []
    seen_ids: set[str] = set()
    for chunk in chunks:
        if chunk.visibility != Visibility.PLAYER or chunk.id in seen_ids:
            continue
        if excluded and _normalize_for_match(chunk.text) in excluded:
            continue
        seen_ids.add(chunk.id)
        selected.append(
            chunk.model_copy(
                update={"text": _truncate_text(chunk.text, budget.max_retrieved_chunk_chars)}
            )
        )
        if len(selected) >= budget.retrieved_chunks:
            break
    return selected


def _normalize_for_match(text: str) -> str:
    return " ".join(text.split()).lower()


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return "." * max_chars
    return f"{text[: max_chars - 3]}..."
