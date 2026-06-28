from __future__ import annotations

from app.agents.secret_guard import collect_hidden_facts
from app.domain import (
    MemoryCandidate,
    PersonaCard,
    SceneState,
    SessionState,
)
from app.llm.provider import LlmProvider
from app.llm.structured_output import StructuredOutputError
from app.memory import MemoryEpisodeStore
from app.memory.deterministic_extractor import (
    contains_durable_event_terms,
    extract_explicit_durable_events,
    is_covered_by_summaries,
)
from app.orchestration.stages.critique import record_structured_failure, validate_gating
from app.orchestration.stages.failure_log import StructuredFailureRecording
from app.orchestration.stages.memory_consolidation import MemoryConsolidator
from app.orchestration.stages.memory_dedup import MemoryDeduplicator
from app.orchestration.stages.memory_protocols import (
    MemoryCuratingAgent,
    MemoryIndexing,
    MemoryStageResult,
)
from app.orchestration.stages.routing import TurnRoutingStage
from app.orchestration.stages.session_summary_cache import SessionSummaryCache
from app.rag.embeddings import EmbeddingProvider

__all__ = [
    "MemoryCuratingAgent",
    "MemoryIndexing",
    "MemoryStageResult",
    "TurnMemoryStage",
]


class TurnMemoryStage:
    """Facade over memory extraction + write (dedup/index) + consolidation collaborators.

    Public __init__/run() are unchanged from the pre-split monolith; the heavy concerns now live
    in MemoryDeduplicator and MemoryConsolidator, sharing one SessionSummaryCache.
    """

    def __init__(
        self,
        *,
        provider: LlmProvider,
        memory_store: MemoryEpisodeStore | None,
        memory_curator: MemoryCuratingAgent | None,
        memory_indexer: MemoryIndexing | None,
        routing_stage: TurnRoutingStage,
        failure_sink: StructuredFailureRecording | None = None,
        gating: str = "always",
        embedding_provider: EmbeddingProvider | None = None,
        write_dedup_cosine_threshold: float = 1.0,
        consolidation_threshold: int = 0,
        consolidation_importance_ceiling: int = 3,
    ) -> None:
        self.provider = provider
        self.memory_store = memory_store
        self.memory_curator = memory_curator
        self.memory_indexer = memory_indexer
        self.routing_stage = routing_stage
        self.failure_sink = failure_sink
        self.gating = validate_gating(gating)
        self._summary_cache = SessionSummaryCache()
        self._deduplicator = MemoryDeduplicator(
            cache=self._summary_cache,
            embedding_provider=embedding_provider,
            write_dedup_cosine_threshold=write_dedup_cosine_threshold,
        )
        self._consolidator = MemoryConsolidator(
            provider=provider,
            routing_stage=routing_stage,
            memory_store=memory_store,
            memory_indexer=memory_indexer,
            memory_curator=memory_curator,
            cache=self._summary_cache,
            consolidation_threshold=consolidation_threshold,
            consolidation_importance_ceiling=consolidation_importance_ceiling,
        )

    async def run(
        self,
        *,
        session: SessionState,
        scene: SceneState,
        persona: PersonaCard,
        user_message: str,
        assistant_message: str,
        retrieval_confidence: float | None,
        scene_complexity: int,
    ) -> MemoryStageResult:
        result = await self._run_extraction(
            session=session,
            scene=scene,
            persona=persona,
            user_message=user_message,
            assistant_message=assistant_message,
            retrieval_confidence=retrieval_confidence,
            scene_complexity=scene_complexity,
        )
        consolidation = await self._consolidator.consolidate_if_needed(
            session_id=session.id,
            retrieval_confidence=retrieval_confidence,
            scene_complexity=scene_complexity,
        )
        if not consolidation:
            return result
        return MemoryStageResult(
            memory_written=result.memory_written,
            warnings=(*result.warnings, *consolidation),
        )

    async def _run_extraction(
        self,
        *,
        session: SessionState,
        scene: SceneState,
        persona: PersonaCard,
        user_message: str,
        assistant_message: str,
        retrieval_confidence: float | None,
        scene_complexity: int,
    ) -> MemoryStageResult:
        if self.memory_curator is None or self.memory_store is None:
            return MemoryStageResult(memory_written=False, warnings=())

        warnings: list[str] = []
        fallback_candidates = extract_explicit_durable_events(
            user_message=user_message,
            scene_id=scene.id,
            actor_id=persona.id,
        )
        if (
            self.gating == "auto"
            and not fallback_candidates
            and not contains_durable_event_terms(user_message)
            and not contains_durable_event_terms(assistant_message)
        ):
            return MemoryStageResult(
                memory_written=False,
                warnings=("memory curation gated: no durable-event signals",),
            )
        route = self.routing_stage.memory(
            retrieval_confidence=retrieval_confidence,
            scene_complexity=scene_complexity,
        )
        try:
            memory_result = await self.memory_curator.curate(
                provider=self.provider,
                route=route,
                session=session,
                scene=scene,
                persona=persona,
                user_message=user_message,
                assistant_message=assistant_message,
            )
            curated = list(memory_result.memories) if memory_result.write_memory else []
            curated_summaries = [candidate.summary for candidate in curated]
            extras = [
                candidate
                for candidate in fallback_candidates
                if not is_covered_by_summaries(candidate.summary, curated_summaries)
            ]
            if extras:
                warnings.append(
                    f"deterministic memory fallback added {len(extras)} explicit durable event(s)"
                )
            memory_written = self._persist_and_index(
                session_id=session.id,
                candidates=[*curated, *extras],
                warnings=warnings,
            )
            return MemoryStageResult(
                memory_written=memory_written,
                warnings=tuple(warnings),
            )
        except StructuredOutputError as exc:
            failure_warnings = [f"memory curation skipped: {exc} ({exc.category})"]
            failure_warnings.extend(
                record_structured_failure(
                    sink=self.failure_sink,
                    task="memory_extraction",
                    error=exc,
                    model=route.model,
                    session_id=session.id,
                    hidden_facts=collect_hidden_facts(persona, scene),
                )
            )
            memory_written = self._fallback_after_curator_failure(
                session_id=session.id,
                candidates=fallback_candidates,
                warnings=failure_warnings,
            )
            return MemoryStageResult(
                memory_written=memory_written,
                warnings=tuple(failure_warnings),
            )
        except Exception as exc:
            failure_warnings = [f"memory curation skipped: {exc}"]
            memory_written = self._fallback_after_curator_failure(
                session_id=session.id,
                candidates=fallback_candidates,
                warnings=failure_warnings,
            )
            return MemoryStageResult(
                memory_written=memory_written,
                warnings=tuple(failure_warnings),
            )

    def _fallback_after_curator_failure(
        self,
        *,
        session_id: str,
        candidates: list[MemoryCandidate],
        warnings: list[str],
    ) -> bool:
        if not candidates:
            return False
        warnings.append(
            f"deterministic memory fallback added {len(candidates)} explicit durable event(s)"
        )
        try:
            return self._persist_and_index(
                session_id=session_id,
                candidates=candidates,
                warnings=warnings,
            )
        except Exception as exc:
            warnings.append(f"deterministic memory fallback skipped: {exc}")
            return False

    def _persist_and_index(
        self,
        *,
        session_id: str,
        candidates: list[MemoryCandidate],
        warnings: list[str],
    ) -> bool:
        if not candidates or self.memory_store is None:
            return False
        candidates = self._deduplicator.drop_duplicates(
            session_id=session_id,
            candidates=candidates,
            warnings=warnings,
            store=self.memory_store,
        )
        if not candidates:
            return False
        persisted = self.memory_store.persist_memories(
            session_id=session_id,
            memories=candidates,
        )
        self._summary_cache.append(session_id, [episode.summary for episode in persisted])
        if self.memory_indexer is not None:
            try:
                self.memory_indexer.index_memories(persisted)
            except Exception as exc:
                warnings.append(f"memory indexing skipped: {exc}")
        return len(persisted) > 0
