from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.domain import (
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
from app.orchestration.stages.critique import record_structured_failure
from app.orchestration.stages.failure_log import StructuredFailureRecording
from app.orchestration.stages.routing import TurnRoutingStage


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
    ) -> None:
        self.provider = provider
        self.memory_store = memory_store
        self.memory_curator = memory_curator
        self.memory_indexer = memory_indexer
        self.routing_stage = routing_stage
        self.failure_sink = failure_sink

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
            if not memory_result.write_memory:
                return MemoryStageResult(memory_written=False, warnings=())
            persisted = self.memory_store.persist_memories(
                session_id=session.id,
                memories=memory_result.memories,
            )
            memory_written = len(persisted) > 0
            if self.memory_indexer is not None:
                try:
                    self.memory_indexer.index_memories(persisted)
                except Exception as exc:
                    warnings.append(f"memory indexing skipped: {exc}")
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
            return MemoryStageResult(
                memory_written=False,
                warnings=tuple(failure_warnings),
            )
        except Exception as exc:
            return MemoryStageResult(
                memory_written=False,
                warnings=(f"memory curation skipped: {exc}",),
            )
