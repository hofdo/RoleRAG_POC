from __future__ import annotations

from app.agents import ActorAgent
from app.domain import SessionState, TurnInput, TurnOutcome, TurnResult
from app.llm.provider import LlmProvider
from app.llm.router import CloudMode, ModelRoute
from app.memory import MemoryEpisodeStore, RecentDialogueStore
from app.orchestration.context_budget import ContextBudget
from app.orchestration.stages import (
    CONTROLLED_FAILURE_TEXT,
    ActorContextRetrieving,
    CriticEvaluatingAgent,
    MemoryCuratingAgent,
    MemoryIndexing,
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
from app.persistence.repositories import SessionRepository, TurnRepository


class TurnOrchestrator:
    def __init__(
        self,
        *,
        loader: TurnDataLoader,
        loader_factory: TurnDataLoaderFactory | None = None,
        content_root: str = "data",
        provider: LlmProvider,
        critic_agent: CriticEvaluatingAgent,
        session_repository: SessionRepository,
        turn_repository: TurnRepository,
        recent_dialogue_store: RecentDialogueStore,
        local_model: str,
        cloud_model: str,
        local_max_tokens: int,
        local_structured_max_tokens: int = 350,
        cloud_max_tokens: int,
        local_temperature: float,
        cloud_temperature: float,
        cloud_mode: CloudMode | str,
        cloud_provider: LlmProvider | None = None,
        memory_store: MemoryEpisodeStore | None = None,
        memory_curator: MemoryCuratingAgent | None = None,
        memory_indexer: MemoryIndexing | None = None,
        actor_context_retriever: ActorContextRetrieving | None = None,
        retrieval_top_k: int = 5,
        max_retrieved_chunk_chars: int = 800,
        recent_dialogue_max_message_chars: int = 900,
    ) -> None:
        self.loader = loader
        self.loader_factory = loader_factory
        self.content_root = content_root
        self.provider = provider
        self.cloud_provider = cloud_provider
        self.session_repository = session_repository
        self.turn_repository = turn_repository
        self.recent_dialogue_store = recent_dialogue_store
        self.memory_store = memory_store
        self.memory_curator = memory_curator
        self.memory_indexer = memory_indexer
        self.context_budget = ContextBudget(
            retrieved_chunks=retrieval_top_k,
            max_retrieved_chunk_chars=max_retrieved_chunk_chars,
        )
        self.actor_agent = ActorAgent()
        self.local_model = local_model
        self.cloud_model = cloud_model
        self.local_max_tokens = local_max_tokens
        self.local_structured_max_tokens = local_structured_max_tokens
        self.cloud_max_tokens = cloud_max_tokens
        self.local_temperature = local_temperature
        self.cloud_temperature = cloud_temperature
        self.recent_dialogue_max_message_chars = recent_dialogue_max_message_chars

        self.session_stage = TurnSessionLoader(
            loader=loader,
            loader_factory=loader_factory,
            content_root=content_root,
            session_repository=session_repository,
            recent_dialogue_store=recent_dialogue_store,
        )
        self.routing_stage = TurnRoutingStage(
            local_model=local_model,
            cloud_model=cloud_model,
            local_max_tokens=local_max_tokens,
            local_structured_max_tokens=local_structured_max_tokens,
            cloud_max_tokens=cloud_max_tokens,
            local_temperature=local_temperature,
            cloud_temperature=cloud_temperature,
            cloud_mode=cloud_mode,
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
            recent_dialogue_max_message_chars=recent_dialogue_max_message_chars,
            actor_agent=self.actor_agent,
        )
        self.critique_stage = TurnCritiqueStage(
            provider=provider,
            critic_agent=critic_agent,
            routing_stage=self.routing_stage,
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
        context = self.session_stage.load(turn_input)
        retrieval = self.retrieval_stage.run(turn_input=turn_input, context=context)
        routing = self.routing_stage.actor(
            turn_input=turn_input,
            scene=context.scene,
            retrieval_confidence=retrieval.confidence,
        )
        generation = await self.generation_stage.run(
            turn_input=turn_input,
            context=context,
            retrieval=retrieval,
            routing=routing,
        )
        warnings = [
            *retrieval.warnings,
            *routing.warnings,
            *generation.warnings,
        ]
        critique = await self.critique_stage.run(
            persona=context.persona,
            scene=context.scene,
            user_message=turn_input.message,
            draft=generation.text,
            retrieved_chunks=retrieval.chunks,
        )
        warnings.extend(critique.warnings)

        final_text = generation.text
        final_route = generation.route
        if critique.critique is not None and not critique.critique.accepted:
            repair = await self.repair_stage.run(
                context=context,
                user_message=turn_input.message,
                actor_messages=generation.messages,
                draft=generation.text,
                route=generation.route,
                critique=critique.critique,
                retrieval=retrieval,
                routing=routing,
            )
            warnings.extend(repair.warnings)
            if repair.outcome == TurnOutcome.CONTROLLED_FAILURE:
                return TurnResult(
                    text=repair.text,
                    route=repair.route,
                    memory_written=False,
                    warnings=warnings,
                    outcome=repair.outcome,
                )
            final_text = repair.text
            final_route = repair.route

        self.persistence_stage.run(
            session=context.session,
            user_message=turn_input.message,
            assistant_message=final_text,
            route=final_route,
        )
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
        return TurnResult(
            text=final_text,
            route=final_route,
            memory_written=memory.memory_written,
            warnings=warnings,
        )

    def loader_for_session(self, session: SessionState) -> TurnDataLoader:
        return self.session_stage.loader_for_content_root(session.content_root)

    def _loader_for_content_root(self, content_root: str) -> TurnDataLoader:
        return self.session_stage.loader_for_content_root(content_root)

    def _build_local_route(self, *, reason: str) -> ModelRoute:
        return self.routing_stage.build_local_route(reason=reason)


__all__ = ["CONTROLLED_FAILURE_TEXT", "TurnOrchestrator"]
