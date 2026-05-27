from __future__ import annotations

from typing import Protocol
from uuid import uuid4

from app.agents import ActorAgent
from app.domain import (
    MemoryCuratorResult,
    PersonaCard,
    SceneState,
    SessionState,
    TurnInput,
    TurnResult,
)
from app.llm.provider import LlmProvider
from app.llm.router import CloudMode, ModelRoute, ModelTask, choose_route
from app.memory import MemoryEpisodeStore, RecentDialogueStore
from app.orchestration.context_builder import build_actor_messages
from app.persistence import DemoWorldRecord, SessionNotFoundError
from app.persistence.repositories import SessionRepository, TurnRepository


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


class TurnOrchestrator:
    def __init__(
        self,
        *,
        loader: TurnDataLoader,
        provider: LlmProvider,
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
        memory_store: MemoryEpisodeStore | None = None,
        memory_curator: MemoryCuratingAgent | None = None,
    ) -> None:
        self.loader = loader
        self.provider = provider
        self.session_repository = session_repository
        self.turn_repository = turn_repository
        self.recent_dialogue_store = recent_dialogue_store
        self.memory_store = memory_store
        self.memory_curator = memory_curator
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
        recent_turns = self.recent_dialogue_store.load_recent_dialogue(session.id)
        messages = build_actor_messages(
            persona=persona,
            scene=scene,
            turn_input=turn_input,
            recent_turns=recent_turns,
        )
        text = await self.actor_agent.generate(
            provider=self.provider,
            route=route,
            messages=messages,
        )
        persisted_turn = self.turn_repository.append_turn(
            session_id=session.id,
            scene_id=session.active_scene_id,
            persona_id=persona_id,
            user_message=turn_input.message,
            assistant_message=text,
            route=route,
        )
        self.session_repository.update_session_activity(
            session.id,
            updated_at=persisted_turn.created_at,
        )

        warnings: list[str] = []
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
                    assistant_message=text,
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
            text=text,
            route=route,
            memory_written=memory_written,
            warnings=warnings,
        )
