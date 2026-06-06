from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.domain import CriticResult, PersonaCard, RetrievedChunk, SceneState
from app.llm.provider import LlmMessage, LlmProvider
from app.llm.router import ModelRoute
from app.orchestration.stages.routing import TurnRoutingStage


class CriticEvaluatingAgent(Protocol):
    async def evaluate(
        self,
        *,
        provider: LlmProvider,
        route: ModelRoute,
        persona: PersonaCard,
        scene: SceneState,
        user_message: str,
        draft: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> CriticResult: ...

    def build_local_repair_messages(
        self,
        *,
        actor_messages: list[LlmMessage],
        rejected_draft: str,
        issues: list[str],
        repair_instruction: str | None,
    ) -> list[LlmMessage]: ...

    def build_cloud_repair_messages(
        self,
        *,
        actor_messages: list[LlmMessage],
        issues: list[str],
    ) -> list[LlmMessage]: ...


@dataclass(frozen=True)
class CritiqueStageResult:
    critique: CriticResult | None
    warnings: tuple[str, ...]


class TurnCritiqueStage:
    def __init__(
        self,
        *,
        provider: LlmProvider,
        critic_agent: CriticEvaluatingAgent,
        routing_stage: TurnRoutingStage,
    ) -> None:
        self.provider = provider
        self.critic_agent = critic_agent
        self.routing_stage = routing_stage

    async def run(
        self,
        *,
        persona: PersonaCard,
        scene: SceneState,
        user_message: str,
        draft: str,
        retrieved_chunks: tuple[RetrievedChunk, ...],
    ) -> CritiqueStageResult:
        try:
            critique = await self.critic_agent.evaluate(
                provider=self.provider,
                route=self.routing_stage.critic(),
                persona=persona,
                scene=scene,
                user_message=user_message,
                draft=draft,
                retrieved_chunks=list(retrieved_chunks),
            )
            return CritiqueStageResult(critique=critique, warnings=())
        except Exception as exc:
            return CritiqueStageResult(
                critique=None,
                warnings=(f"critic skipped: {exc}",),
            )
