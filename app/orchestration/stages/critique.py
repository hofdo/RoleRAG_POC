"""Critic stage: evaluates the actor draft and gates when it runs.

``TurnCritiqueStage`` invokes the critic on the session's bound provider and
redacts any hidden-fact echo from its output via ``app.agents.secret_guard``
before it is reused. It is fail-closed: a structured-output error or any other
exception yields a failed critique so the orchestrator can escalate to a
controlled failure rather than emit an unreviewed draft. Under ``auto`` gating,
``_is_risky_turn`` decides whether to critique at all — flagged by the draft
validator, low/absent retrieval confidence (``LOW_RETRIEVAL_CONFIDENCE``), high
scene complexity (``HIGH_SCENE_COMPLEXITY``), or a cloud route; ``always`` gating
critiques every turn.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from app.agents.secret_guard import collect_hidden_facts, redact_hidden_facts, scan_reply
from app.domain import CriticResult, PersonaCard, RetrievedChunk, SceneState
from app.llm.provider import LlmMessage, LlmProvider, resolve_provider
from app.llm.router import (
    HIGH_SCENE_COMPLEXITY,
    LOW_RETRIEVAL_CONFIDENCE,
    ModelProviderName,
    ModelRoute,
)
from app.llm.structured_output import StructuredOutputError
from app.orchestration.stages.failure_log import StructuredFailureRecording
from app.orchestration.stages.routing import TurnRoutingStage

GATING_MODES: frozenset[str] = frozenset({"always", "auto"})


def validate_gating(gating: str) -> str:
    """Reject an unknown gating mode at construction instead of silently treating it as 'always'."""
    if gating not in GATING_MODES:
        raise ValueError(f"gating must be one of {sorted(GATING_MODES)}, got {gating!r}")
    return gating


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
        include_hidden: bool = True,
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
    # True when the critic ERRORED (could not validate), as opposed to a deliberate gated skip.
    # A failed critique fails the turn closed downstream; a gated one still serves the draft.
    failed: bool = False


class TurnCritiqueStage:
    def __init__(
        self,
        *,
        provider: LlmProvider,
        critic_agent: CriticEvaluatingAgent,
        routing_stage: TurnRoutingStage,
        cloud_provider: LlmProvider | None = None,
        failure_sink: StructuredFailureRecording | None = None,
        gating: str = "always",
        low_retrieval_confidence: float = LOW_RETRIEVAL_CONFIDENCE,
        high_scene_complexity: int = HIGH_SCENE_COMPLEXITY,
    ) -> None:
        self.provider = provider
        self.cloud_provider = cloud_provider
        self.critic_agent = critic_agent
        self.routing_stage = routing_stage
        self.failure_sink = failure_sink
        self.gating = validate_gating(gating)
        self.low_retrieval_confidence = low_retrieval_confidence
        self.high_scene_complexity = high_scene_complexity

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
        route_provider: ModelProviderName,
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
        # The critic follows the session's provider (route_provider is the actor
        # route's provider, which IS the session provider -- see SessionState.provider).
        route = self.routing_stage.critic(provider=route_provider)
        try:
            critique = await self.critic_agent.evaluate(
                provider=resolve_provider(route, local=self.provider, cloud=self.cloud_provider),
                route=route,
                persona=persona,
                scene=scene,
                user_message=user_message,
                draft=draft,
                retrieved_chunks=list(retrieved_chunks),
                # Hidden authored content never leaves the machine: cloud critics
                # check prose/consistency only; the deterministic local
                # secret_guard scan remains the containment layer.
                include_hidden=route.provider == ModelProviderName.LOCAL,
            )
            issues, repair_instruction, leaked = redact_hidden_facts(
                issues=list(critique.issues),
                repair_instruction=critique.repair_instruction,
                hidden_facts=collect_hidden_facts(persona, scene),
            )
            if leaked:
                critique = critique.model_copy(
                    update={"issues": issues, "repair_instruction": repair_instruction}
                )
                return CritiqueStageResult(
                    critique=critique,
                    warnings=("critic output redacted: prevented hidden-fact leak",),
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
                    hidden_facts=collect_hidden_facts(persona, scene),
                )
            )
            return CritiqueStageResult(critique=None, warnings=tuple(warnings), failed=True)
        except Exception as exc:
            return CritiqueStageResult(
                critique=None,
                warnings=(f"critic skipped: {exc}",),
                failed=True,
            )

    def _is_risky_turn(
        self,
        *,
        validator_flagged: bool,
        retrieval_confidence: float | None,
        scene_complexity: int,
        route_provider: ModelProviderName,
    ) -> bool:
        return (
            validator_flagged
            or retrieval_confidence is None
            or retrieval_confidence < self.low_retrieval_confidence
            or scene_complexity >= self.high_scene_complexity
            or route_provider == ModelProviderName.CLOUD
        )


def record_structured_failure(
    *,
    sink: StructuredFailureRecording | None,
    task: str,
    error: StructuredOutputError,
    model: str,
    session_id: str | None = None,
    hidden_facts: Sequence[str] = (),
) -> tuple[str, ...]:
    """Record a raw structured-output failure; report sink problems as warnings.

    The failed model output may contain confabulated hidden facts, so verbatim
    hidden-fact echoes are redacted before the raw text is written to the sink.
    """
    if sink is None:
        return ()
    raw_text = error.raw_text
    if hidden_facts and raw_text:
        raw_text = scan_reply(raw_text, hidden_facts).text
    try:
        sink.record(
            task=task,
            category=error.category,
            raw_text=raw_text,
            model=model,
            session_id=session_id,
        )
    except Exception as exc:
        return (f"structured failure log skipped: {exc}",)
    return ()
