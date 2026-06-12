from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.domain import CriticResult, PersonaCard, RetrievedChunk, SceneState
from app.llm.provider import LlmMessage, LlmProvider
from app.llm.router import (
    HIGH_SCENE_COMPLEXITY,
    LOW_RETRIEVAL_CONFIDENCE,
    ModelProviderName,
    ModelRoute,
)
from app.llm.structured_output import StructuredOutputError
from app.orchestration.stages.failure_log import StructuredFailureRecording
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
        failure_sink: StructuredFailureRecording | None = None,
        gating: str = "always",
    ) -> None:
        self.provider = provider
        self.critic_agent = critic_agent
        self.routing_stage = routing_stage
        self.failure_sink = failure_sink
        self.gating = gating

    async def run(
        self,
        *,
        persona: PersonaCard,
        scene: SceneState,
        user_message: str,
        draft: str,
        retrieved_chunks: tuple[RetrievedChunk, ...],
        validator_flagged: bool = False,
        retrieval_confidence: float | None = None,
        scene_complexity: int = 0,
        route_provider: ModelProviderName = ModelProviderName.LOCAL,
    ) -> CritiqueStageResult:
        if self.gating == "auto" and not self._is_risky_turn(
            validator_flagged=validator_flagged,
            retrieval_confidence=retrieval_confidence,
            scene_complexity=scene_complexity,
            route_provider=route_provider,
        ):
            return CritiqueStageResult(
                critique=None,
                warnings=("critic gated: low-risk turn",),
            )
        route = self.routing_stage.critic()
        try:
            critique = await self.critic_agent.evaluate(
                provider=self.provider,
                route=route,
                persona=persona,
                scene=scene,
                user_message=user_message,
                draft=draft,
                retrieved_chunks=list(retrieved_chunks),
            )
            return CritiqueStageResult(critique=critique, warnings=())
        except StructuredOutputError as exc:
            warnings = [f"critic skipped: {exc} ({exc.category})"]
            warnings.extend(
                record_structured_failure(
                    sink=self.failure_sink,
                    task="critic",
                    error=exc,
                    model=route.model,
                )
            )
            return CritiqueStageResult(critique=None, warnings=tuple(warnings))
        except Exception as exc:
            return CritiqueStageResult(
                critique=None,
                warnings=(f"critic skipped: {exc}",),
            )

    @staticmethod
    def _is_risky_turn(
        *,
        validator_flagged: bool,
        retrieval_confidence: float | None,
        scene_complexity: int,
        route_provider: ModelProviderName,
    ) -> bool:
        return (
            validator_flagged
            or retrieval_confidence is None
            or retrieval_confidence < LOW_RETRIEVAL_CONFIDENCE
            or scene_complexity >= HIGH_SCENE_COMPLEXITY
            or route_provider == ModelProviderName.CLOUD
        )


def record_structured_failure(
    *,
    sink: StructuredFailureRecording | None,
    task: str,
    error: StructuredOutputError,
    model: str,
    session_id: str | None = None,
) -> tuple[str, ...]:
    """Record a raw structured-output failure; report sink problems as warnings."""
    if sink is None:
        return ()
    try:
        sink.record(
            task=task,
            category=error.category,
            raw_text=error.raw_text,
            model=model,
            session_id=session_id,
        )
    except Exception as exc:
        return (f"structured failure log skipped: {exc}",)
    return ()
