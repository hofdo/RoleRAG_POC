from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.domain import (
    MemoryCandidate,
    MemoryCuratorResult,
    MemoryEpisode,
    PersonaCard,
    SceneState,
    SessionState,
)
from app.llm.provider import LlmProvider
from app.llm.router import ModelRoute
from app.llm.structured_output import StructuredOutputError
from app.memory import MemoryEpisodeStore
from app.memory.deterministic_extractor import (
    contains_durable_event_terms,
    extract_explicit_durable_events,
    is_covered_by_summaries,
)
from app.memory.semantic_dedup import is_semantic_duplicate
from app.orchestration.stages.critique import record_structured_failure
from app.orchestration.stages.failure_log import StructuredFailureRecording
from app.orchestration.stages.routing import TurnRoutingStage
from app.rag.embeddings import EmbeddingProvider


class MemoryCuratingAgent(Protocol):
    async def curate(
        self,
        *,
        provider: LlmProvider,
        route: ModelRoute,
        session: SessionState,
        scene: SceneState,
        persona: PersonaCard,
        user_message: str,
        assistant_message: str,
    ) -> MemoryCuratorResult: ...


class MemoryIndexing(Protocol):
    def index_memories(self, memories: list[MemoryEpisode]) -> object: ...


@dataclass(frozen=True)
class MemoryStageResult:
    memory_written: bool
    warnings: tuple[str, ...]


class TurnMemoryStage:
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
    ) -> None:
        self.provider = provider
        self.memory_store = memory_store
        self.memory_curator = memory_curator
        self.memory_indexer = memory_indexer
        self.routing_stage = routing_stage
        self.failure_sink = failure_sink
        self.gating = gating
        self.embedding_provider = embedding_provider
        self.write_dedup_cosine_threshold = write_dedup_cosine_threshold

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
        candidates = self._drop_duplicate_candidates(
            session_id=session_id,
            candidates=candidates,
            warnings=warnings,
        )
        if not candidates:
            return False
        persisted = self.memory_store.persist_memories(
            session_id=session_id,
            memories=candidates,
        )
        if self.memory_indexer is not None:
            try:
                self.memory_indexer.index_memories(persisted)
            except Exception as exc:
                warnings.append(f"memory indexing skipped: {exc}")
        return len(persisted) > 0

    def _drop_duplicate_candidates(
        self,
        *,
        session_id: str,
        candidates: list[MemoryCandidate],
        warnings: list[str],
    ) -> list[MemoryCandidate]:
        """Skip candidates already covered by persisted session memories.

        Always-on curation writes ~1.7 memories per turn; without this cap the
        store fills with near-duplicates that crowd real events out of
        retrieval in long sessions.
        """
        assert self.memory_store is not None
        try:
            existing = [
                episode.summary
                for episode in self.memory_store.list_memories_for_session(session_id)
            ]
        except Exception as exc:
            warnings.append(f"memory dedup skipped: {exc}")
            return candidates
        if not existing:
            return candidates
        kept: list[MemoryCandidate] = []
        for candidate in candidates:
            if is_covered_by_summaries(candidate.summary, existing):
                continue
            kept.append(candidate)
            existing.append(candidate.summary)
        dropped = len(candidates) - len(kept)
        if dropped:
            warnings.append(f"memory dedup dropped {dropped} duplicate candidate(s)")
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
        """Drop candidates whose embedding is near-identical to an existing or
        already-kept memory. Inert unless the cosine threshold is below 1.0."""
        if (
            self.embedding_provider is None
            or self.write_dedup_cosine_threshold >= 1.0
            or not candidates
        ):
            return candidates
        try:
            reference_vectors = list(self.embedding_provider.embed_batch(existing_summaries))
            kept: list[MemoryCandidate] = []
            for candidate in candidates:
                vector = self.embedding_provider.embed_text(candidate.summary)
                if is_semantic_duplicate(
                    vector,
                    reference_vectors,
                    threshold=self.write_dedup_cosine_threshold,
                ):
                    continue
                kept.append(candidate)
                reference_vectors.append(vector)
        except Exception as exc:
            warnings.append(f"semantic memory dedup skipped: {exc}")
            return candidates
        dropped = len(candidates) - len(kept)
        if dropped:
            warnings.append(f"semantic memory dedup dropped {dropped} near-duplicate candidate(s)")
        return kept
