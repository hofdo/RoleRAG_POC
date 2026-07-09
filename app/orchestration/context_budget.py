from __future__ import annotations

import re
from collections.abc import Collection, Sequence

from pydantic import BaseModel, Field

from app.domain import RetrievedChunk, Visibility

_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?](?:\s|$)")


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

    Distinct-id chunks whose normalized text duplicates a chunk already selected are
    also dropped (docs/22 N3): with semantic write-dedup off, near-identical memories
    can accumulate under different ids and would otherwise co-fill retrieved slots.
    First occurrence (highest rank) wins, so ranking order and determinism are
    preserved; the freed slot goes to the next distinct chunk, same as the C1 path --
    but only if the caller's fetched ``chunks`` window contains one. Unlike the C1
    exclusion count, the number of N3 collisions isn't known ahead of the fetch, so
    ``TurnRetrievalStage.run`` over-fetches a bounded worst-case margin
    (``retrieved_chunks - 1``) rather than an exact count; see its docstring.
    """
    excluded = {_normalize_for_match(text) for text in exclude_texts}
    selected: list[RetrievedChunk] = []
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    for chunk in chunks:
        if chunk.visibility != Visibility.PLAYER or chunk.id in seen_ids:
            continue
        normalized_text = _normalize_for_match(chunk.text)
        if excluded and normalized_text in excluded:
            continue
        if normalized_text in seen_texts:
            continue
        seen_ids.add(chunk.id)
        seen_texts.add(normalized_text)
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
    """Trim ``text`` to at most ``max_chars`` (docs/22 P0.3).

    Prefers cutting at the last sentence boundary before the cap so a retrieved
    chunk doesn't lose a fact mid-clause; falls back to the last word boundary,
    then a hard cut, for text with no punctuation/whitespace to anchor on. The
    explicit ``"..."`` omission marker is always kept when trimming occurs.
    """
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return "." * max_chars
    budget = max_chars - 3
    window = text[:budget]

    last_sentence_end = -1
    for match in _SENTENCE_BOUNDARY_RE.finditer(window):
        last_sentence_end = match.start() + 1
    if last_sentence_end > 0:
        return f"{window[:last_sentence_end]}..."

    last_space = window.rfind(" ")
    if last_space > 0:
        return f"{window[:last_space]}..."

    return f"{window}..."
