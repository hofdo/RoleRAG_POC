from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.domain import MemoryCandidate
from app.memory import MemoryEpisodeStore
from app.memory.deterministic_extractor import (
    COVERAGE_THRESHOLD,
    WRITE_DEDUP_COVERAGE_THRESHOLD,
    _reversal_markers,
    _strip_framing,
    is_covered_by_summaries,
)
from app.memory.semantic_dedup import is_semantic_duplicate
from app.orchestration.stages.session_summary_cache import SessionSummaryCache
from app.rag.embeddings import EmbeddingProvider
from app.rag.ranking import content_terms


class MemoryDeduplicator:
    """Lexical + semantic write-dedup, backed by the per-session summary cache."""

    def __init__(
        self,
        *,
        cache: SessionSummaryCache,
        embedding_provider: EmbeddingProvider | None,
        write_dedup_cosine_threshold: float,
    ) -> None:
        self._cache = cache
        self.embedding_provider = embedding_provider
        self.write_dedup_cosine_threshold = write_dedup_cosine_threshold

    def drop_duplicates(
        self,
        *,
        session_id: str,
        candidates: list[MemoryCandidate],
        warnings: list[str],
        store: MemoryEpisodeStore,
    ) -> list[MemoryCandidate]:
        """Skip candidates already covered by persisted session memories.

        Always-on curation writes ~1.7 memories per turn; without this cap the store fills with
        near-duplicates that crowd real events out of retrieval in long sessions.
        """
        try:
            # Per-session summary cache: reload the full session from the store only on first
            # miss (see SessionSummaryCache); kept current via the writer (append) and
            # consolidation (invalidate).
            existing = self._cache.load(session_id, store)
        except Exception as exc:
            warnings.append(f"memory dedup skipped: {exc}")
            return candidates
        if not existing:
            return candidates
        kept: list[MemoryCandidate] = []
        dropped_summaries: list[str] = []
        for candidate in candidates:
            if is_covered_by_summaries(
                candidate.summary, existing, threshold=WRITE_DEDUP_COVERAGE_THRESHOLD
            ):
                dropped_summaries.append(candidate.summary)
                continue
            kept.append(candidate)
            existing.append(candidate.summary)
        if dropped_summaries:
            warnings.append(
                f"memory dedup dropped {len(dropped_summaries)} duplicate candidate(s): "
                + "; ".join(_snippet(s) for s in dropped_summaries)
            )
        return self._drop_semantic_duplicates(
            candidates=kept,
            existing_summaries=existing[: len(existing) - len(kept)],
            warnings=warnings,
        )

    def _drop_semantic_duplicates(
        self,
        *,
        candidates: list[MemoryCandidate],
        existing_summaries: list[str],
        warnings: list[str],
    ) -> list[MemoryCandidate]:
        """Drop candidates whose embedding is near-identical to an existing or already-kept
        memory. Inert unless the cosine threshold is below 1.0."""
        if (
            self.embedding_provider is None
            or self.write_dedup_cosine_threshold >= 1.0
            or not candidates
        ):
            return candidates
        try:
            reference_vectors = list(self.embedding_provider.embed_batch(existing_summaries))
            kept: list[MemoryCandidate] = []
            dropped_summaries: list[str] = []
            for candidate in candidates:
                vector = self.embedding_provider.embed_text(candidate.summary)
                if is_semantic_duplicate(
                    vector,
                    reference_vectors,
                    threshold=self.write_dedup_cosine_threshold,
                ):
                    dropped_summaries.append(candidate.summary)
                    continue
                kept.append(candidate)
                reference_vectors.append(vector)
        except Exception as exc:
            warnings.append(f"semantic memory dedup skipped: {exc}")
            return candidates
        if dropped_summaries:
            warnings.append(
                f"semantic memory dedup dropped {len(dropped_summaries)} near-duplicate "
                "candidate(s): " + "; ".join(_snippet(s) for s in dropped_summaries)
            )
        return kept


def _snippet(summary: str, limit: int = 80) -> str:
    # The dropped text used to be unrecoverable — a live false-drop (docs/22 P2.2,
    # 2026-07-12) took a DB diff against a prior run to diagnose. Warnings persist
    # into turn diagnostics, so the snippet makes drops auditable after the fact.
    text = summary if len(summary) <= limit else summary[: limit - 1] + "…"
    return f'"{text}"'


@dataclass(frozen=True)
class CoverageMatch:
    """The argmax result of `best_covering_summary`: which curated summary covers a
    candidate best, and by how much."""

    index: int
    score: float


def best_covering_summary(
    candidate_summary: str,
    summaries: Sequence[str],
    *,
    threshold: float = COVERAGE_THRESHOLD,
) -> CoverageMatch | None:
    """Argmax-coverage variant of `is_covered_by_summaries` (docs/26 §3.2): instead of
    a bool for the first summary that clears `threshold`, return the index + score of
    the summary with the HIGHEST coverage among those that clear it.

    Reuses `is_covered_by_summaries`'s own term-overlap + reversal-marker primitives
    unmodified, so for every input reachable from the real deterministic extractor
    (candidate summaries always carry real content terms -- see the guard below),
    `is_covered_by_summaries(candidate_summary, summaries, threshold=threshold)` is
    True if and only if this returns non-None. `is_covered_by_summaries` itself is
    untouched.

    Deterministic tie-break: the LOWEST index wins an exact-score tie (first
    encountered), per docs/26 §3.2.
    """
    candidate_terms = content_terms(_strip_framing(candidate_summary))
    if not candidate_terms:
        # Cannot happen via extract_explicit_durable_events (every extracted
        # candidate matches a trigger pattern with real content words); guarded here
        # only to avoid a ZeroDivisionError below for any other caller.
        return None
    candidate_markers = _reversal_markers(candidate_summary)
    best: CoverageMatch | None = None
    for index, summary in enumerate(summaries):
        overlap = len(candidate_terms & content_terms(_strip_framing(summary)))
        score = overlap / len(candidate_terms)
        if score < threshold:
            continue
        # Mirrors is_covered_by_summaries: a reversal marker present in the candidate
        # but absent from this summary means a state CHANGE, not a duplicate -- this
        # summary does not cover the candidate, no matter how high its term overlap.
        if candidate_markers - _reversal_markers(summary):
            continue
        if best is None or score > best.score:
            best = CoverageMatch(index=index, score=score)
    return best


def ordered_union(base: Sequence[str], additional: Sequence[str]) -> list[str]:
    """`base`'s items in order, then any `additional` items not already present (in
    their own relative order). No duplicates either way."""
    result = list(base)
    seen = set(result)
    for item in additional:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result
