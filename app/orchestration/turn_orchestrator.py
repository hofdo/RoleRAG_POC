from __future__ import annotations

from typing import Protocol
from uuid import uuid4

from app.agents import ActorAgent
from app.domain import (
    CriticResult,
    MemoryCuratorResult,
    PersonaCard,
    RetrievedChunk,
    SceneState,
    SessionState,
    TurnInput,
    TurnResult,
)
from app.llm.provider import LlmMessage, LlmProvider
from app.llm.router import CloudMode, ModelRoute, ModelTask, choose_route
from app.memory import MemoryEpisodeStore, RecentDialogueStore
from app.orchestration.context_budget import ContextBudget
from app.orchestration.context_builder import build_actor_messages
from app.persistence import DemoWorldRecord, SessionNotFoundError
from app.persistence.repositories import SessionRepository, TurnRepository
from app.rag.retriever import build_retrieval_query


class TurnDataLoader(Protocol):
    def load_world(self, world_id: str) -> DemoWorldRecord: ...

    def load_persona(self, persona_id: str) -> PersonaCard: ...

    def load_scene(self, scene_id: str) -> SceneState: ...


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


class ActorContextRetrieving(Protocol):
    def retrieve_for_actor(
        self,
        *,
        query: str,
        world_id: str,
        session_id: str,
        persona_id: str,
        top_k: int,
    ) -> list[RetrievedChunk]: ...


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


CONTROLLED_FAILURE_TEXT = (
    "The system could not produce a response that passed validation. "
    "No memory or world state was changed."
)


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
        local_model: str,
        cloud_model: str,
        local_max_tokens: int,
        cloud_max_tokens: int,
        local_temperature: float,
        cloud_temperature: float,
        cloud_mode: CloudMode | str,
        cloud_provider: LlmProvider | None = None,
        memory_store: MemoryEpisodeStore | None = None,
        memory_curator: MemoryCuratingAgent | None = None,
        actor_context_retriever: ActorContextRetrieving | None = None,
        retrieval_top_k: int = 5,
        max_retrieved_chunk_chars: int = 800,
    ) -> None:
        self.loader = loader
        self.provider = provider
        self.cloud_provider = cloud_provider
        self.critic_agent = critic_agent
        self.session_repository = session_repository
        self.turn_repository = turn_repository
        self.recent_dialogue_store = recent_dialogue_store
        self.memory_store = memory_store
        self.memory_curator = memory_curator
        self.actor_context_retriever = actor_context_retriever
        self.context_budget = ContextBudget(
            retrieved_chunks=retrieval_top_k,
            max_retrieved_chunk_chars=max_retrieved_chunk_chars,
        )
        self.actor_agent = ActorAgent()
        self.local_model = local_model
        self.cloud_model = cloud_model
        self.local_max_tokens = local_max_tokens
        self.cloud_max_tokens = cloud_max_tokens
        self.local_temperature = local_temperature
        self.cloud_temperature = cloud_temperature
        self.cloud_mode = CloudMode(cloud_mode)

    def create_session(
        self,
        *,
        world_id: str,
        scene_id: str,
        active_persona_id: str,
        player_name: str,
        session_id: str | None = None,
    ) -> SessionState:
        world = self.loader.load_world(world_id)
        if active_persona_id not in world.persona_ids:
            raise ValueError(
                f"Unknown persona for world {world_id}: {active_persona_id}"
            )
        if scene_id not in world.scene_ids:
            raise ValueError(f"Unknown scene for world {world_id}: {scene_id}")
        self.loader.load_persona(active_persona_id)
        self.loader.load_scene(scene_id)
        return self.session_repository.create_session(
            SessionState(
                id=session_id or str(uuid4()),
                world_id=world_id,
                active_scene_id=scene_id,
                active_persona_id=active_persona_id,
                player_name=player_name,
            )
        )

    def resume_session(self, session_id: str) -> SessionState:
        session = self.session_repository.get_session(session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        return session

    async def run_turn(self, *, turn_input: TurnInput) -> TurnResult:
        session = self.resume_session(turn_input.session_id)
        persona_id = session.active_persona_id
        if turn_input.active_persona_id is not None and turn_input.active_persona_id != persona_id:
            raise ValueError("Turn persona override does not match the stored session persona")

        world = self.loader.load_world(session.world_id)
        if persona_id not in world.persona_ids:
            raise ValueError(f"Unknown persona for world {session.world_id}: {persona_id}")
        if session.active_scene_id not in world.scene_ids:
            raise ValueError(
                f"Unknown scene for world {session.world_id}: {session.active_scene_id}"
            )

        persona = self.loader.load_persona(persona_id)
        scene = self.loader.load_scene(session.active_scene_id)
        recent_turns = self.recent_dialogue_store.load_recent_dialogue(session.id)
        warnings: list[str] = []
        retrieved_chunks: list[RetrievedChunk] = []
        if self.actor_context_retriever is not None:
            query = build_retrieval_query(
                user_message=turn_input.message,
                scene=scene,
                persona=persona,
                recent_turns=recent_turns,
            )
            try:
                retrieved_chunks = self.actor_context_retriever.retrieve_for_actor(
                    query=query,
                    world_id=session.world_id,
                    session_id=session.id,
                    persona_id=persona.id,
                    top_k=self.context_budget.retrieved_chunks,
                )
            except Exception as exc:
                warnings.append(f"retrieval skipped: {exc}")
        route = choose_route(
            task=ModelTask.ACTOR_RESPONSE,
            cloud_mode=self.cloud_mode,
            local_model=self.local_model,
            cloud_model=self.cloud_model,
            local_max_tokens=self.local_max_tokens,
            cloud_max_tokens=self.cloud_max_tokens,
            local_temperature=self.local_temperature,
            cloud_temperature=self.cloud_temperature,
            failed_local_attempts=0,
            retrieval_confidence=None,
            scene_complexity=1,
        )
        messages = build_actor_messages(
            persona=persona,
            scene=scene,
            turn_input=turn_input,
            recent_turns=recent_turns,
            retrieved_chunks=retrieved_chunks,
            context_budget=self.context_budget,
        )
        final_text: str
        final_route: ModelRoute

        text = await self._generate_actor_response(route=route, messages=messages)
        critic_result = await self._evaluate_draft(
            persona=persona,
            scene=scene,
            user_message=turn_input.message,
            draft=text,
            retrieved_chunks=retrieved_chunks,
            warnings=warnings,
        )
        if critic_result is None:
            final_text = text
            final_route = route
        elif critic_result.accepted:
            final_text = text
            final_route = route
        else:
            local_repair_route = choose_route(
                task=ModelTask.REPAIR,
                cloud_mode=self.cloud_mode,
                local_model=self.local_model,
                cloud_model=self.cloud_model,
                local_max_tokens=self.local_max_tokens,
                cloud_max_tokens=self.cloud_max_tokens,
                local_temperature=self.local_temperature,
                cloud_temperature=self.cloud_temperature,
                failed_local_attempts=1,
                retrieval_confidence=None,
                scene_complexity=1,
            )
            repaired_text = await self._generate_actor_response(
                route=local_repair_route,
                messages=self.critic_agent.build_local_repair_messages(
                    actor_messages=messages,
                    rejected_draft=text,
                    issues=critic_result.issues,
                    repair_instruction=critic_result.repair_instruction,
                ),
            )
            repaired_critic_result = await self._evaluate_draft(
                persona=persona,
                scene=scene,
                user_message=turn_input.message,
                draft=repaired_text,
                retrieved_chunks=retrieved_chunks,
                warnings=warnings,
            )
            if repaired_critic_result is None:
                final_text = repaired_text
                final_route = local_repair_route
            elif repaired_critic_result.accepted:
                final_text = repaired_text
                final_route = local_repair_route
            else:
                cloud_repair_route = choose_route(
                    task=ModelTask.REPAIR,
                    cloud_mode=self.cloud_mode,
                    local_model=self.local_model,
                    cloud_model=self.cloud_model,
                    local_max_tokens=self.local_max_tokens,
                    cloud_max_tokens=self.cloud_max_tokens,
                    local_temperature=self.local_temperature,
                    cloud_temperature=self.cloud_temperature,
                    failed_local_attempts=2,
                    retrieval_confidence=None,
                    scene_complexity=1,
                )
                if cloud_repair_route.provider != route.provider:
                    if cloud_repair_route.requires_user_confirmation:
                        warnings.append(
                            "cloud repair skipped: "
                            f"confirmation required for {cloud_repair_route.model}"
                        )
                        return TurnResult(
                            text=CONTROLLED_FAILURE_TEXT,
                            route=cloud_repair_route,
                            memory_written=False,
                            warnings=warnings,
                        )
                    if self.cloud_provider is None:
                        warnings.append("cloud repair skipped: cloud provider unavailable")
                        return TurnResult(
                            text=CONTROLLED_FAILURE_TEXT,
                            route=cloud_repair_route,
                            memory_written=False,
                            warnings=warnings,
                        )
                    cloud_repaired_text = await self._generate_actor_response(
                        route=cloud_repair_route,
                        messages=self.critic_agent.build_cloud_repair_messages(
                            actor_messages=messages,
                            issues=repaired_critic_result.issues,
                        ),
                    )
                    cloud_critic_result = await self._evaluate_draft(
                        persona=persona,
                        scene=scene,
                        user_message=turn_input.message,
                        draft=cloud_repaired_text,
                        retrieved_chunks=retrieved_chunks,
                        warnings=warnings,
                    )
                    if cloud_critic_result is None:
                        final_text = cloud_repaired_text
                        final_route = cloud_repair_route
                    elif cloud_critic_result.accepted:
                        final_text = cloud_repaired_text
                        final_route = cloud_repair_route
                    else:
                        return TurnResult(
                            text=CONTROLLED_FAILURE_TEXT,
                            route=cloud_repair_route,
                            memory_written=False,
                            warnings=warnings,
                        )
                else:
                    return TurnResult(
                        text=CONTROLLED_FAILURE_TEXT,
                        route=local_repair_route,
                        memory_written=False,
                        warnings=warnings,
                    )

        persisted_turn = self.turn_repository.append_turn(
            session_id=session.id,
            scene_id=session.active_scene_id,
            persona_id=persona_id,
            user_message=turn_input.message,
            assistant_message=final_text,
            route=final_route,
        )
        self.session_repository.update_session_activity(
            session.id,
            updated_at=persisted_turn.created_at,
        )

        memory_written = False
        if self.memory_curator is not None and self.memory_store is not None:
            memory_route = choose_route(
                task=ModelTask.MEMORY_EXTRACTION,
                cloud_mode=self.cloud_mode,
                local_model=self.local_model,
                cloud_model=self.cloud_model,
                local_max_tokens=self.local_max_tokens,
                cloud_max_tokens=self.cloud_max_tokens,
                local_temperature=self.local_temperature,
                cloud_temperature=self.cloud_temperature,
                failed_local_attempts=0,
                retrieval_confidence=None,
                scene_complexity=1,
            )
            try:
                memory_result = await self.memory_curator.curate(
                    provider=self.provider,
                    route=memory_route,
                    session=session,
                    scene=scene,
                    persona=persona,
                    user_message=turn_input.message,
                    assistant_message=final_text,
                )
                if memory_result.write_memory:
                    persisted_memories = self.memory_store.persist_memories(
                        session_id=session.id,
                        memories=memory_result.memories,
                    )
                    memory_written = len(persisted_memories) > 0
            except Exception as exc:
                warnings.append(f"memory curation skipped: {exc}")

        return TurnResult(
            text=final_text,
            route=final_route,
            memory_written=memory_written,
            warnings=warnings,
        )

    async def _generate_actor_response(
        self,
        *,
        route: ModelRoute,
        messages: list[LlmMessage],
    ) -> str:
        provider = self.provider if route.provider.value == "local" else self.cloud_provider
        if provider is None:
            raise RuntimeError(f"Missing provider for route: {route.provider.value}")
        return await self.actor_agent.generate(provider=provider, route=route, messages=messages)

    async def _evaluate_draft(
        self,
        *,
        persona: PersonaCard,
        scene: SceneState,
        user_message: str,
        draft: str,
        retrieved_chunks: list[RetrievedChunk],
        warnings: list[str],
    ) -> CriticResult | None:
        route = choose_route(
            task=ModelTask.CRITIC,
            cloud_mode=self.cloud_mode,
            local_model=self.local_model,
            cloud_model=self.cloud_model,
            local_max_tokens=self.local_max_tokens,
            cloud_max_tokens=self.cloud_max_tokens,
            local_temperature=self.local_temperature,
            cloud_temperature=self.cloud_temperature,
            failed_local_attempts=0,
            retrieval_confidence=None,
            scene_complexity=1,
        )
        try:
            return await self.critic_agent.evaluate(
                provider=self.provider,
                route=route,
                persona=persona,
                scene=scene,
                user_message=user_message,
                draft=draft,
                retrieved_chunks=retrieved_chunks,
            )
        except Exception as exc:
            warnings.append(f"critic skipped: {exc}")
            return None
