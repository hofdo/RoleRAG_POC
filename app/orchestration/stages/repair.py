from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter

from app.domain import CriticResult, CriticStatus, TurnOutcome
from app.llm.provider import LlmMessage
from app.llm.router import ModelRoute
from app.orchestration.draft_validator import DraftValidationResult
from app.orchestration.stages.critique import CritiqueStageResult, TurnCritiqueStage
from app.orchestration.stages.generation import (
    EmptyProviderResponseError,
    GenerationStageResult,
    TruncatedProviderResponseError,
    TurnGenerationStage,
)
from app.orchestration.stages.retrieval import RetrievalStageResult
from app.orchestration.stages.routing import RoutingStageResult, TurnRoutingStage
from app.orchestration.stages.session import LoadedTurnContext

CONTROLLED_FAILURE_TEXT = (
    "The system could not produce a response that passed validation. "
    "No memory or world state was changed."
)


def _emit_stage(on_stage: Callable[[str], None] | None, stage: str) -> None:
    # Private duplicate of turn_orchestrator._emit_stage: importing from turn_orchestrator here
    # would create an import cycle (turn_orchestrator imports this module via
    # app.orchestration.stages).
    if on_stage is None:
        return
    try:
        on_stage(stage)
    except Exception:  # noqa: BLE001 - progress reporting must never fail a turn
        pass


@dataclass(frozen=True)
class RepairStageResult:
    text: str
    route: ModelRoute
    finish_reason: str | None
    outcome: TurnOutcome
    warnings: tuple[str, ...]
    usage: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class RepairResolution:
    """The orchestrator-applicable outcome of the critique→repair decision.

    When ``controlled_failure`` is True the caller returns a CONTROLLED_FAILURE turn;
    otherwise it applies text/route/finish_reason/critic_status and continues. ``repair_duration``
    is set only when a repair attempt actually ran (so ``stage_timings['repair']`` matches today).
    ``usage`` follows the same generation as ``finish_reason``: the repair generation's
    usage when a repair ran, otherwise the initial actor generation's (#69).
    """

    text: str
    route: ModelRoute
    finish_reason: str | None
    critic_status: CriticStatus
    warnings: tuple[str, ...]
    controlled_failure: bool
    repair_duration: float | None = None
    usage: dict[str, int] = field(default_factory=dict)


def _validation_repair_instruction(validation: DraftValidationResult) -> str:
    parts = []
    if validation.unsupported_entities:
        names = ", ".join(validation.unsupported_entities)
        parts.append(f"Remove or replace details not present in the scene: {names}.")
    if validation.evaded_action:
        parts.append("Directly address the player's question or action.")
    return " ".join(parts)


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
        """One same-provider repair pass, then a re-critique. A rejection here goes
        straight to controlled failure -- there is no second (cloud) attempt. The
        repair route is the session's provider, same as the actor route, so on a
        cloud session this pass runs on cloud."""
        warnings: list[str] = []
        repair_route = self.routing_stage.repair(provider=context.session.provider)
        repair_messages = self.critique_stage.critic_agent.build_local_repair_messages(
            actor_messages=list(actor_messages),
            rejected_draft=draft,
            issues=critique.issues,
            repair_instruction=critique.repair_instruction,
        )
        (
            repaired_text,
            repaired_route,
            repaired_finish_reason,
            repaired_usage,
            generation_warnings,
        ) = await self.generation_stage.generate_messages(
            route=repair_route,
            messages=repair_messages,
        )
        warnings.extend(generation_warnings)
        repaired_critique = await self.critique_stage.run(
            persona=context.persona,
            scene=context.scene,
            user_message=user_message,
            draft=repaired_text,
            retrieved_chunks=retrieval.chunks,
            # Without this, route_provider defaults to LOCAL and the re-check critic
            # would be routed to the local provider even on a cloud session -- the
            # same session-provider affinity the repair route itself already respects
            # two lines up.
            route_provider=context.session.provider,
        )
        warnings.extend(repaired_critique.warnings)
        if repaired_critique.failed:
            # The re-check critic errored: same fail-closed rule as the initial critique (#17).
            # Don't serve the unvalidated repaired draft.
            return self._controlled_failure(
                repaired_route,
                warnings,
                finish_reason=repaired_finish_reason,
                usage=repaired_usage,
            )
        if repaired_critique.critique is None or repaired_critique.critique.accepted:
            return RepairStageResult(
                text=repaired_text,
                route=repaired_route,
                finish_reason=repaired_finish_reason,
                outcome=TurnOutcome.SUCCESS,
                warnings=tuple(warnings),
                usage=repaired_usage,
            )
        return self._controlled_failure(
            repaired_route,
            warnings,
            finish_reason=repaired_finish_reason,
            usage=repaired_usage,
        )

    async def resolve(
        self,
        *,
        context: LoadedTurnContext,
        user_message: str,
        generation: GenerationStageResult,
        validation: DraftValidationResult,
        critique: CritiqueStageResult,
        retrieval: RetrievalStageResult,
        routing: RoutingStageResult,
        on_stage: Callable[[str], None] | None = None,
    ) -> RepairResolution:
        """Decide critique→repair end-to-end and return an orchestrator-applicable resolution."""
        if critique.failed:
            # The critic ERRORED, so the secret-leak gate never ran. Refuse to serve the
            # unvalidated draft (and write no memory/world state) rather than risk a leak.
            # Re-running generation would not help: the validator is what is broken, not the draft.
            return RepairResolution(
                text=CONTROLLED_FAILURE_TEXT,
                route=generation.route,
                finish_reason=generation.finish_reason,
                critic_status=CriticStatus.REJECTED,
                warnings=("draft withheld: critic validation unavailable",),
                controlled_failure=True,
                usage=generation.usage,
            )
        base_status = (
            CriticStatus.SKIPPED if critique.critique is None else CriticStatus.ACCEPTED
        )
        effective_critique = critique.critique
        validator_forced = False
        if validation.flags and (effective_critique is None or effective_critique.accepted):
            validator_forced = True
            effective_critique = CriticResult(
                accepted=False,
                issues=list(validation.flags),
                repair_instruction=_validation_repair_instruction(validation),
            )
        if effective_critique is None or effective_critique.accepted:
            return RepairResolution(
                text=generation.text,
                route=generation.route,
                finish_reason=generation.finish_reason,
                critic_status=base_status,
                warnings=(),
                controlled_failure=False,
                usage=generation.usage,
            )

        # A repair pass will actually run from here on, so this is the first point where
        # emitting the "repair" stage frame is not misleading to the SPA's progress display.
        _emit_stage(on_stage, "repair")
        warnings: list[str] = []
        repair: RepairStageResult | None = None
        repair_failure: str | None = None
        started = perf_counter()
        try:
            repair = await self.run(
                context=context,
                user_message=user_message,
                actor_messages=generation.messages,
                draft=generation.text,
                route=generation.route,
                critique=effective_critique,
                retrieval=retrieval,
                routing=routing,
            )
        except (EmptyProviderResponseError, TruncatedProviderResponseError) as exc:
            repair_failure = str(exc)
        repair_duration = perf_counter() - started
        if repair is not None:
            warnings.extend(repair.warnings)

        if repair_failure is not None or (
            repair is not None and repair.outcome == TurnOutcome.CONTROLLED_FAILURE
        ):
            # The validator is heuristic; when it alone forced the repair, an imperfect
            # original draft beats a controlled failure.
            if validator_forced:
                warnings.append(
                    "draft validation repair failed; returning original draft"
                    + (f": {repair_failure}" if repair_failure else "")
                )
                return RepairResolution(
                    text=generation.text,
                    route=generation.route,
                    finish_reason=generation.finish_reason,
                    critic_status=base_status,
                    warnings=tuple(warnings),
                    controlled_failure=False,
                    repair_duration=repair_duration,
                    usage=generation.usage,
                )
            if repair_failure is not None:
                warnings.append(f"repair failed: {repair_failure}")
            return RepairResolution(
                text=repair.text if repair is not None else CONTROLLED_FAILURE_TEXT,
                route=repair.route if repair is not None else generation.route,
                finish_reason=(
                    repair.finish_reason if repair is not None else generation.finish_reason
                ),
                critic_status=CriticStatus.REJECTED,
                warnings=tuple(warnings),
                controlled_failure=True,
                repair_duration=repair_duration,
                usage=repair.usage if repair is not None else generation.usage,
            )

        assert repair is not None
        return RepairResolution(
            text=repair.text,
            route=repair.route,
            finish_reason=repair.finish_reason,
            critic_status=CriticStatus.REPAIRED,
            warnings=tuple(warnings),
            controlled_failure=False,
            usage=repair.usage,
            repair_duration=repair_duration,
        )

    @staticmethod
    def _controlled_failure(
        route: ModelRoute,
        warnings: list[str],
        *,
        finish_reason: str | None,
        usage: dict[str, int] | None = None,
    ) -> RepairStageResult:
        return RepairStageResult(
            text=CONTROLLED_FAILURE_TEXT,
            route=route,
            finish_reason=finish_reason,
            outcome=TurnOutcome.CONTROLLED_FAILURE,
            warnings=tuple(warnings),
            usage=usage or {},
        )
