from __future__ import annotations

from app.domain import MemoryCandidate, MemoryEpisode, Visibility
from app.llm.provider import LlmProvider
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
    ) -> None:
        self.provider = provider
        self.routing_stage = routing_stage
        self.memory_store = memory_store
        self.memory_indexer = memory_indexer
        self.memory_curator = memory_curator
        self._cache = cache
        self.consolidation_threshold = consolidation_threshold
        self.consolidation_importance_ceiling = consolidation_importance_ceiling

    async def consolidate_if_needed(
        self,
        *,
        session_id: str,
        retrieval_confidence: float | None,
        scene_complexity: int,
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
            candidates = select_consolidatable(
                self.memory_store.list_memories_for_session(session_id),
                importance_ceiling=self.consolidation_importance_ceiling,
            )
        except Exception as exc:
            return (f"memory consolidation skipped: {exc}",)
        if len(candidates) < self.consolidation_threshold:
            return ()

        warnings: list[str] = []
        route = self.routing_stage.memory(
            retrieval_confidence=retrieval_confidence,
            scene_complexity=scene_complexity,
        )
        try:
            summary_text = await self.memory_curator.consolidate(
                provider=self.provider,
                route=route,
                summaries=[memory.summary for memory in candidates],
            )
        except Exception as exc:
            summary_text = deterministic_consolidated_summary(candidates)
            warnings.append(f"memory consolidation fell back to deterministic roll-up: {exc}")
        try:
            self._persist_consolidation(
                session_id=session_id,
                candidates=candidates,
                summary_text=summary_text,
            )
        except Exception as exc:
            warnings.append(f"memory consolidation skipped: {exc}")
            return tuple(warnings)
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
    ) -> None:
        assert self.memory_store is not None and self.memory_indexer is not None
        summary = MemoryCandidate(
            summary=summary_text,
            visibility=Visibility.PLAYER,
            importance=min(5, self.consolidation_importance_ceiling + 1),
            tags=[SUMMARY_TAG],
            scene_id=candidates[0].scene_id,
            actor_id=candidates[0].actor_id,
        )
        persisted = self.memory_store.persist_memories(session_id=session_id, memories=[summary])
        self.memory_indexer.index_memories(persisted)
        original_ids = [memory.id for memory in candidates]
        self.memory_store.mark_memories_consolidated(original_ids)
        self.memory_indexer.unindex(original_ids)
        # Consolidation removed originals and added one summary; rather than surgically patch
        # the mirror, drop the entry so the next turn rebuilds it from the store. Consolidation
        # is infrequent, so the one extra load is cheap and keeps the cache consistent.
        self._cache.invalidate(session_id)
