from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from app.domain import RetrievedChunk, Visibility


class ContextBudget(BaseModel):
    retrieved_chunks: int = Field(default=5, ge=0)
    max_retrieved_chunk_chars: int = Field(default=800, ge=1)


def select_retrieved_chunks_for_prompt(
    chunks: Sequence[RetrievedChunk],
    *,
    budget: ContextBudget,
) -> list[RetrievedChunk]:
    selected: list[RetrievedChunk] = []
    seen_ids: set[str] = set()
    for chunk in chunks:
        if chunk.visibility != Visibility.PLAYER or chunk.id in seen_ids:
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


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return "." * max_chars
    return f"{text[: max_chars - 3]}..."
