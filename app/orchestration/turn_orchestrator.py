"""Coordinates a single turn through the actor-critic pipeline.

``TurnOrchestrator`` runs the ordered stages defined in
``app.orchestration.stages`` — load session/scene/persona, retrieve context,
route to the session's bound provider, generate the actor draft, critique it
(with one bounded same-provider repair pass), curate and index memory, and
persist — returning a ``TurnResult``. It applies output-side secret containment
via ``app.agents.secret_guard`` before persisting, degrades curator failures to
warnings, and fails closed to a controlled failure when the critic rejects or
errors. ``TurnOrchestratorConfig`` carries the scalar tunables so the
constructor takes dependencies plus one config object.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter

from app.agents import ActorAgent
from app.agents.secret_guard import (
    DEFAULT_PARAPHRASE_OVERLAP,
    collect_hidden_facts,
    scan_reply,
)
from app.domain import (
    CriticStatus,
    DeferredMemoryJob,
    SessionState,
    TurnDiagnostics,
    TurnInput,
    TurnOutcome,
    TurnResult,
    TurnRetrievalDiagnostics,
)
from app.llm.provider import LlmProvider
from app.llm.router import (
    HIGH_SCENE_COMPLEXITY,
    LOW_RETRIEVAL_CONFIDENCE,
    ModelProviderName,
    ModelRoute,
)
from app.memory import MemoryEpisodeStore, RecentDialogueStore
from app.orchestration.context_budget import ContextBudget
from app.orchestration.draft_validator import validate_draft
from app.orchestration.stages import (
    CONTROLLED_FAILURE_TEXT,
    ActorContextRetrieving,
    CriticEvaluatingAgent,
    EmptyProviderResponseError,
    LoadedTurnContext,
    MemoryCuratingAgent,
    MemoryIndexing,
    RetrievalStageResult,
    StructuredFailureRecording,
    TruncatedProviderResponseError,
    TurnCritiqueStage,
    TurnDataLoader,
    TurnDataLoaderFactory,
    TurnGenerationStage,
    TurnMemoryStage,
    TurnPersistenceStage,
    TurnRepairStage,
    TurnRetrievalStage,
    TurnRoutingStage,
    TurnSessionLoader,
)
from app.persistence.repositories import CanonRepository, SessionRepository, TurnRepository
from app.rag.embeddings import EmbeddingProvider


def _visible_texts(
    *,
    context: LoadedTurnContext,
    retrieval: RetrievalStageResult,
) -> list[str]:
    texts = [
        context.scene.title,
        context.scene.location,
        context.scene.player_visible_summary,
        context.persona.name,
        context.persona.public_description,
        context.persona.speaking_style,
    ]
    texts.extend(chunk.text for chunk in retrieval.chunks)
    for turn in context.recent_turns:
        texts.append(turn.user_message)
        texts.append(turn.assistant_message)
    return texts


@contextmanager
def _stage_timer(timings: dict[str, float], stage: str) -> Iterator[None]:
    started = perf_counter()
    try:
        yield
    finally:
        timings[stage] = perf_counter() - started


def _emit_stage(on_stage: Callable[[str], None] | None, stage: str) -> None:
    if on_stage is None:
        return
    try:
        on_stage(stage)
    except Exception:  # noqa: BLE001 - progress reporting must never fail a turn
        pass


@dataclass(frozen=True, kw_only=True)
class TurnOrchestratorConfig:
    """Scalar turn tunables, grouped so the orchestrator ctor takes deps + one config.

    Defaults mirror the prior TurnOrchestrator.__init__ defaults exactly (behavior-preserving);
    composition builds this from Settings via build_orchestrator_config(). Mirrors RankingWeights.
    """

    local_model: str
    cloud_model: str
    local_max_tokens: int
    cloud_max_tokens: int
    local_temperature: float
    cloud_temperature: float
    content_root: str = "data"
    local_structured_max_tokens: int = 350
    retrieval_top_k: int = 5
    max_retrieved_chunk_chars: int = 800
    recent_dialogue_max_message_chars: int = 900
    critic_gating: str = "always"
    curator_gating: str = "always"
    canon_importance_floor: int = 4
    canon_max_items: int = 8
    canon_max_chars: int = 900
    write_dedup_cosine_threshold: float = 1.0
    low_retrieval_confidence: float = LOW_RETRIEVAL_CONFIDENCE
    high_scene_complexity: int = HIGH_SCENE_COMPLEXITY
    truncation_retry_budget_multiplier: int = 2
    containment_overlap_threshold: float = DEFAULT_PARAPHRASE_OVERLAP
    memory_consolidation_threshold: int = 0
    memory_consolidation_max_importance: int = 3


class TurnOrchestrator:
    def __init__(
        self,
        *,
        loader: TurnDataLoader,
        provider: LlmProvider,
        critic_agent: CriticEvaluatingAgent,
        session_repository: SessionRepository,
        turn_repository: TurnRepository,
        recent_dialogue_store: RecentDialogueStore,
        config: TurnOrchestratorConfig,
        loader_factory: TurnDataLoaderFactory | None = None,
        cloud_provider: LlmProvider | None = None,
        memory_store: MemoryEpisodeStore | None = None,
        memory_curator: MemoryCuratingAgent | None = None,
        memory_indexer: MemoryIndexing | None = None,
        actor_context_retriever: ActorContextRetrieving | None = None,
        structured_failure_sink: StructuredFailureRecording | None = None,
        memory_embedding_provider: EmbeddingProvider | None = None,
        canon_repository: CanonRepository | None = None,
    ) -> None:
        self.config = config
        self.loader = loader
        self.loader_factory = loader_factory
        self.provider = provider
        self.cloud_provider = cloud_provider
        self.session_repository = session_repository
        self.turn_repository = turn_repository
        self.recent_dialogue_store = recent_dialogue_store
        self.memory_store = memory_store
        self.memory_curator = memory_curator
        self.memory_indexer = memory_indexer
        self.context_budget = ContextBudget(
            retrieved_chunks=config.retrieval_top_k,
            max_retrieved_chunk_chars=config.max_retrieved_chunk_chars,
        )
        self.actor_agent = ActorAgent()
        self.containment_overlap_threshold = config.containment_overlap_threshold

        self.session_stage = TurnSessionLoader(
            loader=loader,
            loader_factory=loader_factory,
            content_root=config.content_root,
            session_repository=session_repository,
            recent_dialogue_store=recent_dialogue_store,
            memory_store=memory_store,
            canon_repository=canon_repository,
            canon_importance_floor=config.canon_importance_floor,
            canon_max_items=config.canon_max_items,
            canon_max_chars=config.canon_max_chars,
        )
        self.routing_stage = TurnRoutingStage(
            local_model=config.local_model,
            cloud_model=config.cloud_model,
            local_max_tokens=config.local_max_tokens,
            local_structured_max_tokens=config.local_structured_max_tokens,
            cloud_max_tokens=config.cloud_max_tokens,
            local_temperature=config.local_temperature,
            cloud_temperature=config.cloud_temperature,
        )
        self.retrieval_stage = TurnRetrievalStage(
            actor_context_retriever=actor_context_retriever,
            context_budget=self.context_budget,
        )
        self.generation_stage = TurnGenerationStage(
            provider=provider,
            cloud_provider=cloud_provider,
            routing_stage=self.routing_stage,
            context_budget=self.context_budget,
            recent_dialogue_max_message_chars=config.recent_dialogue_max_message_chars,
            actor_agent=self.actor_agent,
            truncation_retry_budget_multiplier=config.truncation_retry_budget_multiplier,
        )
        self.critique_stage = TurnCritiqueStage(
            provider=provider,
            cloud_provider=cloud_provider,
            critic_agent=critic_agent,
            routing_stage=self.routing_stage,
            failure_sink=structured_failure_sink,
            gating=config.critic_gating,
            low_retrieval_confidence=config.low_retrieval_confidence,
            high_scene_complexity=config.high_scene_complexity,
        )
        self.repair_stage = TurnRepairStage(
            generation_stage=self.generation_stage,
            critique_stage=self.critique_stage,
            routing_stage=self.routing_stage,
        )
        self.persistence_stage = TurnPersistenceStage(
            session_repository=session_repository,
            turn_repository=turn_repository,
        )
        self.memory_stage = TurnMemoryStage(
            provider=provider,
            cloud_provider=cloud_provider,
            memory_store=memory_store,
            memory_curator=memory_curator,
            memory_indexer=memory_indexer,
            routing_stage=self.routing_stage,
            failure_sink=structured_failure_sink,
            gating=config.curator_gating,
            embedding_provider=memory_embedding_provider,
            write_dedup_cosine_threshold=config.write_dedup_cosine_threshold,
            consolidation_threshold=config.memory_consolidation_threshold,
            consolidation_importance_ceiling=config.memory_consolidation_max_importance,
        )

    @property
    def critic_agent(self) -> CriticEvaluatingAgent:
        return self.critique_stage.critic_agent

    @critic_agent.setter
    def critic_agent(self, value: CriticEvaluatingAgent) -> None:
        self.critique_stage.critic_agent = value

    @property
    def actor_context_retriever(self) -> ActorContextRetrieving | None:
        return self.retrieval_stage.actor_context_retriever

    @actor_context_retriever.setter
    def actor_context_retriever(self, value: ActorContextRetrieving | None) -> None:
        self.retrieval_stage.actor_context_retriever = value

    def create_session(
        self,
        *,
        world_id: str,
        scene_id: str,
        active_persona_id: str,
        player_name: str,
        session_id: str | None = None,
        content_root: str | None = None,
        provider: ModelProviderName = ModelProviderName.LOCAL,
    ) -> SessionState:
        return self.session_stage.create_session(
            world_id=world_id,
            scene_id=scene_id,
            active_persona_id=active_persona_id,
            player_name=player_name,
            session_id=session_id,
            content_root=content_root,
            provider=provider,
        )

    def resume_session(self, session_id: str) -> SessionState:
        return self.session_stage.resume_session(session_id)

    async def run_turn(
        self,
        *,
        turn_input: TurnInput,
        on_stage: Callable[[str], None] | None = None,
        defer_memory: bool = False,
    ) -> TurnResult:
        timings: dict[str, float] = {}
        _emit_stage(on_stage, "session")
        with _stage_timer(timings, "session"):
            context = self.session_stage.load(turn_input)
        _emit_stage(on_stage, "retrieval")
        with _stage_timer(timings, "retrieval"):
            retrieval = self.retrieval_stage.run(turn_input=turn_input, context=context)
        _emit_stage(on_stage, "routing")
        with _stage_timer(timings, "routing"):
            routing = self.routing_stage.actor(
                provider=context.session.provider,
                scene=context.scene,
            )
        try:
            _emit_stage(on_stage, "generation")
            with _stage_timer(timings, "generation"):
                generation = await self.generation_stage.run(
                    turn_input=turn_input,
                    context=context,
                    retrieval=retrieval,
                    routing=routing,
                )
        except (EmptyProviderResponseError, TruncatedProviderResponseError) as exc:
            failure_warnings = [
                *retrieval.warnings,
                *routing.warnings,
                f"actor failed: {exc}",
            ]
            self._persist_controlled_failure(
                context=context,
                user_message=turn_input.message,
                text=CONTROLLED_FAILURE_TEXT,
                route=routing.route,
                finish_reason=None,
                critic_status=CriticStatus.SKIPPED,
                warnings=failure_warnings,
                timings=timings,
                retrieval_diagnostics=retrieval.diagnostics,
                on_stage=on_stage,
            )
            return TurnResult(
                text=CONTROLLED_FAILURE_TEXT,
                route=routing.route,
                finish_reason=None,
                memory_written=False,
                critic_status=CriticStatus.SKIPPED,
                warnings=failure_warnings,
                retrieval=retrieval.diagnostics,
                stage_timings=timings,
                outcome=TurnOutcome.CONTROLLED_FAILURE,
            )
        warnings = [
            *retrieval.warnings,
            *routing.warnings,
            *generation.warnings,
        ]
        _emit_stage(on_stage, "validation")
        with _stage_timer(timings, "validation"):
            validation = validate_draft(
                draft=generation.text,
                player_message=turn_input.message,
                visible_texts=_visible_texts(context=context, retrieval=retrieval),
            )
        warnings.extend(validation.flags)
        _emit_stage(on_stage, "critique")
        with _stage_timer(timings, "critique"):
            critique = await self.critique_stage.run(
                persona=context.persona,
                scene=context.scene,
                user_message=turn_input.message,
                draft=generation.text,
                retrieved_chunks=retrieval.chunks,
                validator_flagged=bool(validation.flags),
                retrieval_confidence=retrieval.confidence,
                scene_complexity=routing.scene_complexity,
                route_provider=generation.route.provider,
            )
        warnings.extend(critique.warnings)

        resolution = await self.repair_stage.resolve(
            context=context,
            user_message=turn_input.message,
            generation=generation,
            validation=validation,
            critique=critique,
            retrieval=retrieval,
            routing=routing,
            on_stage=on_stage,
        )
        warnings.extend(resolution.warnings)
        if resolution.repair_duration is not None:
            timings["repair"] = resolution.repair_duration
        if resolution.controlled_failure:
            self._persist_controlled_failure(
                context=context,
                user_message=turn_input.message,
                text=resolution.text,
                route=resolution.route,
                finish_reason=resolution.finish_reason,
                critic_status=resolution.critic_status,
                warnings=warnings,
                timings=timings,
                retrieval_diagnostics=retrieval.diagnostics,
                on_stage=on_stage,
            )
            return TurnResult(
                text=resolution.text,
                route=resolution.route,
                finish_reason=resolution.finish_reason,
                memory_written=False,
                critic_status=resolution.critic_status,
                warnings=warnings,
                retrieval=retrieval.diagnostics,
                stage_timings=timings,
                outcome=TurnOutcome.CONTROLLED_FAILURE,
            )
        final_text = resolution.text
        final_route = resolution.route
        final_finish_reason = resolution.finish_reason
        critic_status = resolution.critic_status

        # Deterministic output-side containment backstop behind the LLM critic: redact
        # verbatim hidden-fact echoes and flag likely paraphrases before the reply is
        # persisted, fed to memory extraction, or returned to the player.
        containment = scan_reply(
            final_text,
            collect_hidden_facts(context.persona, context.scene),
            paraphrase_overlap_threshold=self.containment_overlap_threshold,
        )
        final_text = containment.text
        if containment.verbatim_redacted:
            warnings.append("containment: redacted verbatim hidden-fact echo from reply")
        if containment.paraphrased_facts:
            warnings.append(
                "containment risk: reply may paraphrase a hidden fact "
                f"({len(containment.paraphrased_facts)} flagged)"
            )

        _emit_stage(on_stage, "persistence")
        with _stage_timer(timings, "persistence"):
            persistence = self.persistence_stage.run(
                session=context.session,
                user_message=turn_input.message,
                assistant_message=final_text,
                route=final_route,
            )
        if context.persona_switched:
            # The turn has now actually persisted, so it is safe to commit the
            # persona override durably. Deferred from TurnSessionLoader.load so
            # that a failed turn (confirmation required, generation/provider
            # error, controlled failure -- all of which return earlier than this
            # point) never leaves a persona switch committed for a turn the
            # player never saw succeed.
            self.session_repository.update_active_persona(
                context.session.id, context.session.active_persona_id
            )
        if defer_memory:
            warnings.append("memory curation deferred: runs after this response")
            self.turn_repository.update_turn_diagnostics(
                persistence.turn.id,
                TurnDiagnostics(
                    retrieval=retrieval.diagnostics,
                    stage_timings=timings,
                    critic_status=critic_status,
                    finish_reason=final_finish_reason,
                    warnings=warnings,
                    memory_written=False,
                ),
            )
            return TurnResult(
                text=final_text,
                route=final_route,
                finish_reason=final_finish_reason,
                memory_written=False,
                critic_status=critic_status,
                warnings=warnings,
                retrieval=retrieval.diagnostics,
                stage_timings=timings,
                deferred_memory=DeferredMemoryJob(
                    session_id=context.session.id,
                    turn_id=persistence.turn.id,
                    scene_id=persistence.turn.scene_id,
                    persona_id=persistence.turn.persona_id,
                    user_message=turn_input.message,
                    assistant_message=final_text,
                    retrieval_confidence=retrieval.confidence,
                    scene_complexity=routing.scene_complexity,
                ),
            )
        _emit_stage(on_stage, "memory")
        with _stage_timer(timings, "memory"):
            memory = await self.memory_stage.run(
                session=context.session,
                scene=context.scene,
                persona=context.persona,
                user_message=turn_input.message,
                assistant_message=final_text,
                retrieval_confidence=retrieval.confidence,
                scene_complexity=routing.scene_complexity,
            )
        warnings.extend(memory.warnings)
        # Persist turn diagnostics from the same values the TurnResult carries so the
        # stored record matches the live response (built after the memory stage so
        # stage_timings is complete).
        self.turn_repository.update_turn_diagnostics(
            persistence.turn.id,
            TurnDiagnostics(
                retrieval=retrieval.diagnostics,
                stage_timings=timings,
                critic_status=critic_status,
                finish_reason=final_finish_reason,
                warnings=warnings,
                memory_written=memory.memory_written,
            ),
        )
        return TurnResult(
            text=final_text,
            route=final_route,
            finish_reason=final_finish_reason,
            memory_written=memory.memory_written,
            critic_status=critic_status,
            warnings=warnings,
            retrieval=retrieval.diagnostics,
            stage_timings=timings,
        )

    async def run_deferred_memory(self, job: DeferredMemoryJob) -> None:
        """Memory stage for an already-persisted turn, after the response was sent.

        Loads persona/scene from job.scene_id/job.persona_id -- the scene and persona
        the turn was actually generated under -- rather than the session's *current*
        live fields. The session's active scene/persona can change between the
        response being sent and this job running (a scene switch or a later turn's
        persona override), and using the live fields would mis-attribute this turn's
        memories to whatever is active when the job happens to run instead of what
        was active when the turn happened.
        """
        session = self.session_stage.resume_session(job.session_id)
        loader = self.session_stage.loader_for_content_root(session.content_root)
        persona = loader.load_persona(job.persona_id)
        scene = loader.load_scene(job.scene_id)
        memory = await self.memory_stage.run(
            session=session,
            scene=scene,
            persona=persona,
            user_message=job.user_message,
            assistant_message=job.assistant_message,
            retrieval_confidence=job.retrieval_confidence,
            scene_complexity=job.scene_complexity,
        )
        self.turn_repository.append_memory_outcome(
            job.turn_id,
            memory_written=memory.memory_written,
            warnings=list(memory.warnings),
        )

    def _persist_controlled_failure(
        self,
        *,
        context: LoadedTurnContext,
        user_message: str,
        text: str,
        route: ModelRoute,
        finish_reason: str | None,
        critic_status: CriticStatus,
        warnings: list[str],
        timings: dict[str, float],
        retrieval_diagnostics: TurnRetrievalDiagnostics | None,
        on_stage: Callable[[str], None] | None,
    ) -> None:
        """Keep failed turns in history: the player's message survives, the failure
        diagnostics become queryable, and acceptance tooling can account for every
        attempted turn. The row is written with outcome=CONTROLLED_FAILURE and is
        excluded from recent-dialogue prompt context by the repository. A pending
        persona switch is deliberately NOT committed here -- that stays tied to a
        turn the player actually saw succeed. Persistence failure degrades to a
        warning: it must never mask the original failure."""
        try:
            _emit_stage(on_stage, "persistence")
            with _stage_timer(timings, "persistence"):
                persistence = self.persistence_stage.run(
                    session=context.session,
                    user_message=user_message,
                    assistant_message=text,
                    route=route,
                    outcome=TurnOutcome.CONTROLLED_FAILURE,
                )
            self.turn_repository.update_turn_diagnostics(
                persistence.turn.id,
                TurnDiagnostics(
                    retrieval=retrieval_diagnostics,
                    stage_timings=timings,
                    critic_status=critic_status,
                    finish_reason=finish_reason,
                    warnings=warnings,
                    memory_written=False,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"failed-turn persistence skipped: {exc}")

    def loader_for_session(self, session: SessionState) -> TurnDataLoader:
        return self.session_stage.loader_for_content_root(session.content_root)


__all__ = ["CONTROLLED_FAILURE_TEXT", "TurnOrchestrator"]
