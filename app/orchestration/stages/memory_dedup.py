from __future__ import annotations

from app.domain import MemoryCandidate
from app.memory import MemoryEpisodeStore
from app.memory.deterministic_extractor import (
    WRITE_DEDUP_COVERAGE_THRESHOLD,
    is_covered_by_summaries,
)
from app.memory.semantic_dedup import is_semantic_duplicate
from app.orchestration.stages.session_summary_cache import SessionSummaryCache
from app.rag.embeddings import EmbeddingProvider


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
