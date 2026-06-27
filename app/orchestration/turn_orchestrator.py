from __future__ import annotations

from collections.abc import Iterator
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
    SessionState,
    TurnDiagnostics,
    TurnInput,
    TurnOutcome,
    TurnResult,
)
from app.llm.provider import LlmProvider
from app.llm.router import (
    HIGH_SCENE_COMPLEXITY,
    LOW_RETRIEVAL_CONFIDENCE,
    CloudMode,
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
    cloud_mode: CloudMode | str
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
            cloud_mode=config.cloud_mode,
            low_retrieval_confidence=config.low_retrieval_confidence,
            high_scene_complexity=config.high_scene_complexity,
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

    @property
    def cloud_mode(self) -> CloudMode:
        return self.routing_stage.cloud_mode

    @cloud_mode.setter
    def cloud_mode(self, value: CloudMode | str) -> None:
        self.routing_stage.cloud_mode = CloudMode(value)

    def create_session(
        self,
        *,
        world_id: str,
        scene_id: str,
        active_persona_id: str,
        player_name: str,
        session_id: str | None = None,
        content_root: str | None = None,
    ) -> SessionState:
        return self.session_stage.create_session(
            world_id=world_id,
            scene_id=scene_id,
            active_persona_id=active_persona_id,
            player_name=player_name,
            session_id=session_id,
            content_root=content_root,
        )

    def resume_session(self, session_id: str) -> SessionState:
        return self.session_stage.resume_session(session_id)

    async def run_turn(self, *, turn_input: TurnInput) -> TurnResult:
        timings: dict[str, float] = {}
        with _stage_timer(timings, "session"):
            context = self.session_stage.load(turn_input)
        with _stage_timer(timings, "retrieval"):
            retrieval = self.retrieval_stage.run(turn_input=turn_input, context=context)
        with _stage_timer(timings, "routing"):
            routing = self.routing_stage.actor(
                turn_input=turn_input,
                scene=context.scene,
                retrieval_confidence=retrieval.confidence,
            )
        if routing.route.requires_user_confirmation:
            return TurnResult(
                text="",
                route=routing.route,
                finish_reason=None,
                memory_written=False,
                critic_status=CriticStatus.SKIPPED,
                warnings=[*retrieval.warnings, *routing.warnings],
                retrieval=retrieval.diagnostics,
                stage_timings=timings,
                outcome=TurnOutcome.CONFIRMATION_REQUIRED,
            )
        try:
            with _stage_timer(timings, "generation"):
                generation = await self.generation_stage.run(
                    turn_input=turn_input,
                    context=context,
                    retrieval=retrieval,
                    routing=routing,
                )
        except (EmptyProviderResponseError, TruncatedProviderResponseError) as exc:
            return TurnResult(
                text=CONTROLLED_FAILURE_TEXT,
                route=routing.route,
                finish_reason=None,
                memory_written=False,
                critic_status=CriticStatus.SKIPPED,
                warnings=[
                    *retrieval.warnings,
                    *routing.warnings,
                    f"actor failed: {exc}",
                ],
                retrieval=retrieval.diagnostics,
                stage_timings=timings,
                outcome=TurnOutcome.CONTROLLED_FAILURE,
            )
        warnings = [
            *retrieval.warnings,
            *routing.warnings,
            *generation.warnings,
        ]
        with _stage_timer(timings, "validation"):
            validation = validate_draft(
                draft=generation.text,
                player_message=turn_input.message,
                visible_texts=_visible_texts(context=context, retrieval=retrieval),
            )
        warnings.extend(validation.flags)
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
        )
        warnings.extend(resolution.warnings)
        if resolution.repair_duration is not None:
            timings["repair"] = resolution.repair_duration
        if resolution.controlled_failure:
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

        with _stage_timer(timings, "persistence"):
            persistence = self.persistence_stage.run(
                session=context.session,
                user_message=turn_input.message,
                assistant_message=final_text,
                route=final_route,
            )
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

    def loader_for_session(self, session: SessionState) -> TurnDataLoader:
        return self.session_stage.loader_for_content_root(session.content_root)

    def _loader_for_content_root(self, content_root: str) -> TurnDataLoader:
        return self.session_stage.loader_for_content_root(content_root)

    def _build_local_route(self, *, reason: str) -> ModelRoute:
        return self.routing_stage.build_local_route(reason=reason)


__all__ = ["CONTROLLED_FAILURE_TEXT", "TurnOrchestrator"]
