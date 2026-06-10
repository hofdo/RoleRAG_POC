from __future__ import annotations

from dataclasses import dataclass

from app.domain import CriticResult, TurnOutcome
from app.llm.provider import LlmMessage
from app.llm.router import ModelProviderName, ModelRoute, ModelTask
from app.orchestration.stages.critique import TurnCritiqueStage
from app.orchestration.stages.generation import TurnGenerationStage
from app.orchestration.stages.retrieval import RetrievalStageResult
from app.orchestration.stages.routing import RoutingStageResult, TurnRoutingStage
from app.orchestration.stages.session import LoadedTurnContext

CONTROLLED_FAILURE_TEXT = (
    "The system could not produce a response that passed validation. "
    "No memory or world state was changed."
)


@dataclass(frozen=True)
class RepairStageResult:
    text: str
    route: ModelRoute
    finish_reason: str | None
    outcome: TurnOutcome
    warnings: tuple[str, ...]


class TurnRepairStage:
    def __init__(
        self,
        *,
        generation_stage: TurnGenerationStage,
        critique_stage: TurnCritiqueStage,
        routing_stage: TurnRoutingStage,
    ) -> None:
        self.generation_stage = generation_stage
        self.critique_stage = critique_stage
        self.routing_stage = routing_stage

    async def run(
        self,
        *,
        context: LoadedTurnContext,
        user_message: str,
        actor_messages: tuple[LlmMessage, ...],
        draft: str,
        route: ModelRoute,
        critique: CriticResult,
        retrieval: RetrievalStageResult,
        routing: RoutingStageResult,
    ) -> RepairStageResult:
        warnings: list[str] = []
        local_route = self.routing_stage.repair(
            failed_local_attempts=1,
            retrieval_confidence=retrieval.confidence,
            scene_complexity=routing.scene_complexity,
        )
        repaired_text, repaired_route, repaired_finish_reason, generation_warnings = (
            await self.generation_stage.generate_messages(
                route=local_route,
                messages=self.critique_stage.critic_agent.build_local_repair_messages(
                    actor_messages=list(actor_messages),
                    rejected_draft=draft,
                    issues=critique.issues,
                    repair_instruction=critique.repair_instruction,
                ),
                task=ModelTask.REPAIR,
                retrieval_confidence=retrieval.confidence,
                scene_complexity=routing.scene_complexity,
            )
        )
        warnings.extend(generation_warnings)
        repaired_critique = await self.critique_stage.run(
            persona=context.persona,
            scene=context.scene,
            user_message=user_message,
            draft=repaired_text,
            retrieved_chunks=retrieval.chunks,
        )
        warnings.extend(repaired_critique.warnings)
        if repaired_critique.critique is None or repaired_critique.critique.accepted:
            return RepairStageResult(
                text=repaired_text,
                route=repaired_route,
                finish_reason=repaired_finish_reason,
                outcome=TurnOutcome.SUCCESS,
                warnings=tuple(warnings),
            )

        cloud_route = self.routing_stage.repair(
            failed_local_attempts=2,
            retrieval_confidence=retrieval.confidence,
            scene_complexity=routing.scene_complexity,
        )
        if cloud_route.provider == ModelProviderName.LOCAL:
            warnings.append(self.routing_stage.warning_for_skipped_cloud(cloud_route.reason))
            return self._controlled_failure(
                cloud_route,
                warnings,
                finish_reason=repaired_finish_reason,
            )
        if cloud_route.requires_user_confirmation:
            warnings.append(
                "cloud repair skipped: "
                f"confirmation required for {cloud_route.model} ({cloud_route.reason})"
            )
            return self._controlled_failure(
                cloud_route,
                warnings,
                finish_reason=repaired_finish_reason,
            )

        cloud_text, cloud_final_route, cloud_finish_reason, generation_warnings = (
            await self.generation_stage.generate_messages(
                route=cloud_route,
                messages=self.critique_stage.critic_agent.build_cloud_repair_messages(
                    actor_messages=list(actor_messages),
                    issues=repaired_critique.critique.issues,
                ),
                task=ModelTask.REPAIR,
                retrieval_confidence=retrieval.confidence,
                scene_complexity=routing.scene_complexity,
            )
        )
        warnings.extend(generation_warnings)
        cloud_critique = await self.critique_stage.run(
            persona=context.persona,
            scene=context.scene,
            user_message=user_message,
            draft=cloud_text,
            retrieved_chunks=retrieval.chunks,
        )
        warnings.extend(cloud_critique.warnings)
        if cloud_critique.critique is None or cloud_critique.critique.accepted:
            return RepairStageResult(
                text=cloud_text,
                route=cloud_final_route,
                finish_reason=cloud_finish_reason,
                outcome=TurnOutcome.SUCCESS,
                warnings=tuple(warnings),
            )
        return self._controlled_failure(
            cloud_final_route,
            warnings,
            finish_reason=cloud_finish_reason,
        )

    @staticmethod
    def _controlled_failure(
        route: ModelRoute,
        warnings: list[str],
        *,
        finish_reason: str | None,
    ) -> RepairStageResult:
        return RepairStageResult(
            text=CONTROLLED_FAILURE_TEXT,
            route=route,
            finish_reason=finish_reason,
            outcome=TurnOutcome.CONTROLLED_FAILURE,
            warnings=tuple(warnings),
        )
