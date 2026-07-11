from __future__ import annotations

import re
from collections.abc import Collection, Sequence

from pydantic import BaseModel, Field

from app.domain import RetrievedChunk, Visibility
from app.llm.provider import LlmMessage

# Preflight prompt-size heuristic, not an exact tokenizer count: ~4 characters per
# token holds up reasonably for English prose. It exists only to warn before a
# small local context window silently context-shifts (#69); real usage (when the
# provider reports it) always supersedes this estimate.
CHARS_PER_TOKEN_ESTIMATE = 4

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
    (``retrieved_chunks - 1``) rather than an exact count; see its docstring for what
    that margin does and does not cover (docs/22 P6).

    The N3 text-dedup key is checked twice -- once on the raw chunk text (cheap, and
    catches the common exact-duplicate case before truncating), and again on the
    *post-truncation* text (docs/22 P5, fixed 2026-07-11): two chunks that only
    diverge past ``budget.max_retrieved_chunk_chars`` (e.g. sharing an 850-char
    prefix with a 800-char cap) are distinct pre-truncation and so pass the first
    check, but ``_truncate_text`` renders them byte-identical -- without the second
    check both copies would still co-fill retrieved slots as duplicate prompt blocks.
    Same first-occurrence-wins, freed-slot-goes-to-the-next-distinct-chunk semantics
    as the pre-truncation check.
    """
    excluded = {_normalize_for_match(text) for text in exclude_texts}
    selected: list[RetrievedChunk] = []
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    seen_truncated_texts: set[str] = set()
    for chunk in chunks:
        if chunk.visibility != Visibility.PLAYER or chunk.id in seen_ids:
            continue
        normalized_text = _normalize_for_match(chunk.text)
        if excluded and normalized_text in excluded:
            continue
        if normalized_text in seen_texts:
            continue
        truncated_text = _truncate_text(chunk.text, budget.max_retrieved_chunk_chars)
        normalized_truncated_text = _normalize_for_match(truncated_text)
        if normalized_truncated_text in seen_truncated_texts:
            continue
        seen_ids.add(chunk.id)
        seen_texts.add(normalized_text)
        seen_truncated_texts.add(normalized_truncated_text)
        selected.append(chunk.model_copy(update={"text": truncated_text}))
        if len(selected) >= budget.retrieved_chunks:
            break
    return selected


def _normalize_for_match(text: str) -> str:
    return " ".join(text.split()).lower()


def _truncate_text(text: str, max_chars: int) -> str:
    """Trim ``text`` to at most ``max_chars`` (docs/22 P0.3, retention-floor fix
    2026-07-11).

    Prefers cutting at the last sentence boundary before the cap so a retrieved
    chunk doesn't lose a fact mid-clause -- but only when that boundary retains at
    least half of the available budget. An early terminator (an abbreviation like
    "Mr." within the first few characters, or a stray "?"/"!" in dialogue) would
    otherwise win unconditionally and collapse the whole chunk to a few characters
    of noise while still consuming one of the prompt's scarce retrieved-chunk
    slots. The word-boundary fallback is held to the same floor for the same
    reason: a solitary early space (e.g. right after "Mr. ") is just as
    noise-collapsing as a solitary early sentence terminator. When neither
    boundary clears the floor, hard-cut at the full budget instead -- always
    keeping the explicit ``"..."`` omission marker when trimming occurs.
    """
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return "." * max_chars
    budget = max_chars - 3
    window = text[:budget]

    # A boundary only wins when it retains at least half of ``budget``;
    # compared via cross-multiplication (retained * 2 >= budget) to avoid float
    # rounding at exact-half boundaries.
    last_sentence_end = -1
    for match in _SENTENCE_BOUNDARY_RE.finditer(window):
        last_sentence_end = match.start() + 1
    if last_sentence_end > 0 and last_sentence_end * 2 >= budget:
        return f"{window[:last_sentence_end]}..."

    last_space = window.rfind(" ")
    if last_space > 0 and last_space * 2 >= budget:
        return f"{window[:last_space]}..."

    return f"{window}..."


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
