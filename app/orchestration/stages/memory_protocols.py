from __future__ import annotations

from collections.abc import Sequence
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

    async def consolidate(
        self,
        *,
        provider: LlmProvider,
        route: ModelRoute,
        summaries: list[str],
    ) -> str: ...


class MemoryIndexing(Protocol):
    def index_memories(self, memories: list[MemoryEpisode]) -> object: ...

    def unindex(self, memory_ids: Sequence[str]) -> None: ...


@dataclass(frozen=True)
class MemoryStageResult:
    memory_written: bool
    warnings: tuple[str, ...]
