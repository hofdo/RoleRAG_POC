from __future__ import annotations

from collections.abc import Collection, Sequence

from pydantic import BaseModel, Field

from app.domain import RetrievedChunk, Visibility
from app.llm.provider import LlmMessage

# Preflight prompt-size heuristic, not an exact tokenizer count: ~4 characters per
# token holds up reasonably for English prose. It exists only to warn before a
# small local context window silently context-shifts (#69); real usage (when the
# provider reports it) always supersedes this estimate.
CHARS_PER_TOKEN_ESTIMATE = 4


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
    """Cut at the last word boundary within budget rather than mid-word, then
    append "...". Falls back to a hard cut when no boundary is found (a single
    run of non-space characters at least as long as the budget) so the result
    never exceeds max_chars either way."""
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return "." * max_chars
    budget = max_chars - 3
    truncated = text[:budget]
    boundary = truncated.rfind(" ")
    if boundary > 0:
        truncated = truncated[:boundary]
    return f"{truncated}..."


def estimate_prompt_tokens(
    messages: Sequence[LlmMessage],
    *,
    max_tokens: int,
) -> int:
    """Cheap preflight estimate of prompt + completion tokens for one generation
    call, using CHARS_PER_TOKEN_ESTIMATE. Not an exact tokenizer count -- see that
    constant's docstring."""
    total_chars = sum(len(message.content) for message in messages)
    return (total_chars // CHARS_PER_TOKEN_ESTIMATE) + max_tokens


def context_preflight_warning(
    messages: Sequence[LlmMessage],
    *,
    max_tokens: int,
    context_window_tokens: int,
    warn_ratio: float,
) -> str | None:
    """Warn (never block) when the estimated prompt+completion token count for an
    about-to-run generation call would cross warn_ratio * context_window_tokens.
    Disabled when context_window_tokens <= 0 (the default -- byte-identical, no
    warning ever fires)."""
    if context_window_tokens <= 0:
        return None
    estimated = estimate_prompt_tokens(messages, max_tokens=max_tokens)
    threshold = warn_ratio * context_window_tokens
    if estimated <= threshold:
        return None
    return (
        f"context preflight: estimated {estimated} tokens vs window {context_window_tokens} "
        f"(warn ratio {warn_ratio})"
    )


def context_usage_warning(
    usage: dict[str, int],
    *,
    context_window_tokens: int,
    warn_ratio: float,
) -> str | None:
    """Post-hoc counterpart to context_preflight_warning: once a real usage report
    is available, actual prompt_tokens beats the chars/4 estimate. Disabled when
    context_window_tokens <= 0 or the provider reported no prompt_tokens."""
    if context_window_tokens <= 0:
        return None
    prompt_tokens = usage.get("prompt_tokens")
    if not prompt_tokens:
        return None
    threshold = warn_ratio * context_window_tokens
    if prompt_tokens <= threshold:
        return None
    return (
        f"context preflight: actual prompt_tokens {prompt_tokens} exceeded "
        f"{warn_ratio} * window {context_window_tokens}"
    )
