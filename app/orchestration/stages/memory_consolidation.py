from __future__ import annotations

from app.domain import MemoryCandidate, MemoryEpisode, Visibility
from app.llm.provider import LlmProvider, resolve_provider
from app.llm.router import ModelProviderName
from app.memory import MemoryEpisodeStore
from app.memory.consolidation import (
    SUMMARY_TAG,
    deterministic_consolidated_summary,
    select_consolidatable,
)
from app.orchestration.stages.memory_protocols import MemoryCuratingAgent, MemoryIndexing
from app.orchestration.stages.routing import TurnRoutingStage
from app.orchestration.stages.session_summary_cache import SessionSummaryCache


class MemoryConsolidator:
    """Sleep-cycle roll-up: fold old low-value episodic memories into one dense summary."""

    def __init__(
        self,
        *,
        provider: LlmProvider,
        routing_stage: TurnRoutingStage,
        memory_store: MemoryEpisodeStore | None,
        memory_indexer: MemoryIndexing | None,
        memory_curator: MemoryCuratingAgent | None,
        cache: SessionSummaryCache,
        consolidation_threshold: int,
        consolidation_importance_ceiling: int,
        consolidation_min_age: int = 0,
        consolidation_batch_cap: int = 0,
        cloud_provider: LlmProvider | None = None,
        canon_tag_pinning: bool = False,
    ) -> None:
        self.provider = provider
        self.cloud_provider = cloud_provider
        self.routing_stage = routing_stage
        self.memory_store = memory_store
        self.memory_indexer = memory_indexer
        self.memory_curator = memory_curator
        self._cache = cache
        self.consolidation_threshold = consolidation_threshold
        self.consolidation_importance_ceiling = consolidation_importance_ceiling
        self.consolidation_min_age = consolidation_min_age
        self.consolidation_batch_cap = consolidation_batch_cap
        # docs/26 §3.3, #78: gates the German preserve-tag aliases in
        # select_consolidatable (byte-identical when False).
        self.canon_tag_pinning = canon_tag_pinning

    async def consolidate_if_needed(
        self,
        *,
        session_id: str,
        retrieval_confidence: float | None,
        scene_complexity: int,
        route_provider: ModelProviderName,
    ) -> tuple[str, ...]:
        """Roll up old low-value episodic memories into one dense summary once the
        consolidatable backlog reaches the threshold. Inert when threshold is 0."""
        if (
            self.consolidation_threshold <= 0
            or self.memory_store is None
            or self.memory_indexer is None
            or self.memory_curator is None
        ):
            return ()
        try:
            # Threshold gate uses the age-floored pool (min_age excluded), so the
            # rolling recent window can't be bypassed by triggering early; batch_cap
            # is applied separately below and only bounds how many of that pool are
            # actually folded in this pass, never whether consolidation triggers.
            eligible = select_consolidatable(
                self.memory_store.list_memories_for_session(session_id),
                importance_ceiling=self.consolidation_importance_ceiling,
                min_age=self.consolidation_min_age,
                canon_tag_pinning=self.canon_tag_pinning,
            )
        except Exception as exc:
            return (f"memory consolidation skipped: {exc}",)
        if len(eligible) < self.consolidation_threshold:
            return ()
        candidates = (
            eligible[: self.consolidation_batch_cap]
            if self.consolidation_batch_cap > 0
            else eligible
        )

        warnings: list[str] = []
        # Consolidation reuses the memory route: same provider as regular memory
        # extraction for this session.
        route = self.routing_stage.memory(provider=route_provider)
        try:
            summary_text = await self.memory_curator.consolidate(
                provider=resolve_provider(route, local=self.provider, cloud=self.cloud_provider),
                route=route,
                summaries=[memory.summary for memory in candidates],
            )
        except Exception as exc:
            summary_text = deterministic_consolidated_summary(candidates)
            warnings.append(f"memory consolidation fell back to deterministic roll-up: {exc}")
        try:
            index_warnings = self._persist_consolidation(
                session_id=session_id,
                candidates=candidates,
                summary_text=summary_text,
            )
        except Exception as exc:
            warnings.append(f"memory consolidation skipped: {exc}")
            return tuple(warnings)
        warnings.extend(index_warnings)
        warnings.append(
            f"memory consolidation: rolled up {len(candidates)} memories into 1 summary"
        )
        return tuple(warnings)

    def _persist_consolidation(
        self,
        *,
        session_id: str,
        candidates: list[MemoryEpisode],
        summary_text: str,
    ) -> tuple[str, ...]:
        """Persist the roll-up summary and mark the originals consolidated in SQLite,
        then best-effort mirror that into Qdrant (docs/22 P3, reordered 2026-07-11).

        SQLite-first: the summary write, the mark-consolidated tag, and the mirror-cache
        invalidation are the authoritative-state changes and must all land before any
        Qdrant work runs. The previous order indexed the summary *before* marking the
        originals consolidated, so an index failure (e.g. a P1.4 embedding-model
        fingerprint mismatch) raised out of this method with the summary already
        committed but the originals never marked -- the same oldest-N backlog stayed
        eligible, so ``consolidate_if_needed`` re-triggered on the very next turn and
        minted a second summary of the same memories, one orphan summary row per turn.
        Indexing/unindexing is now best-effort *after* the SQLite triplet commits,
        mirroring ``TurnMemoryStage._persist_and_index``'s existing fail-open pattern: a
        failure degrades to a warning instead of raising, so the SQLite state (which is
        authoritative -- Qdrant is a rebuildable derived index, invariant #1) is never
        left inconsistent with what ``select_consolidatable`` will see next turn. A
        stale Qdrant mirror recovers via ``reindex-memories``.
        """
        assert self.memory_store is not None and self.memory_indexer is not None
        original_ids = [memory.id for memory in candidates]
        # Provenance carry-forward (docs/26 §3.1.1, #76): the summary's source_turn_id
        # is the EARLIEST (min) of the folded originals' non-null source_turn_id
        # values -- "when was this fact first established." min() is taken over the
        # non-null subset only: a legacy/unprovenanced original (source_turn_id is
        # None) must never make min() see None, and an all-legacy fold must leave the
        # summary's source_turn_id None rather than coercing to a sentinel turn id.
        non_null_source_turn_ids = [
            memory.source_turn_id for memory in candidates if memory.source_turn_id is not None
        ]
        source_turn_id = min(non_null_source_turn_ids) if non_null_source_turn_ids else None
        summary = MemoryCandidate(
            summary=summary_text,
            visibility=Visibility.PLAYER,
            importance=min(5, self.consolidation_importance_ceiling + 1),
            # source_ids audit trail (docs/26 §3.1.1): the folded originals' own
            # episode ids, comma-joined, reusing the existing tags_json column --
            # no new schema. Independent of source_turn_id: recorded even when every
            # original was unprovenanced (source_turn_id stays None above).
            tags=[SUMMARY_TAG, f"source_ids:{','.join(original_ids)}"],
            scene_id=candidates[0].scene_id,
            actor_id=candidates[0].actor_id,
            source_turn_id=source_turn_id,
        )
        persisted = self.memory_store.persist_memories(session_id=session_id, memories=[summary])
        self.memory_store.mark_memories_consolidated(original_ids)
        # Consolidation removed originals and added one summary; rather than surgically patch
        # the mirror, drop the entry so the next turn rebuilds it from the store. Consolidation
        # is infrequent, so the one extra load is cheap and keeps the cache consistent.
        self._cache.invalidate(session_id)
        try:
            self.memory_indexer.index_memories(persisted)
            self.memory_indexer.unindex(original_ids)
        except Exception as exc:
            return (f"memory consolidation indexing degraded: {exc}",)
        return ()
