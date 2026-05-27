from __future__ import annotations

from typing import Protocol

from app.agents import ActorAgent
from app.domain import PersonaCard, SceneState, TurnInput, TurnResult
from app.llm.provider import LlmProvider
from app.llm.router import CloudMode, ModelTask, choose_route
from app.orchestration.context_builder import build_actor_messages
from app.persistence import DemoWorldRecord


class TurnDataLoader(Protocol):
    def load_world(self, world_id: str) -> DemoWorldRecord: ...

    def load_persona(self, persona_id: str) -> PersonaCard: ...

    def load_scene(self, scene_id: str) -> SceneState: ...


class TurnOrchestrator:
    def __init__(
        self,
        *,
        loader: TurnDataLoader,
        provider: LlmProvider,
        local_model: str,
        cloud_model: str,
        local_max_tokens: int,
        cloud_max_tokens: int,
        local_temperature: float,
        cloud_temperature: float,
        cloud_mode: CloudMode | str,
    ) -> None:
        self.loader = loader
        self.provider = provider
        self.actor_agent = ActorAgent()
        self.local_model = local_model
        self.cloud_model = cloud_model
        self.local_max_tokens = local_max_tokens
        self.cloud_max_tokens = cloud_max_tokens
        self.local_temperature = local_temperature
        self.cloud_temperature = cloud_temperature
        self.cloud_mode = CloudMode(cloud_mode)

    async def run_turn(self, *, turn_input: TurnInput, world_id: str, scene_id: str) -> TurnResult:
        world = self.loader.load_world(world_id)
        if turn_input.active_persona_id not in world.persona_ids:
            raise ValueError(
                f"Unknown persona for world {world_id}: {turn_input.active_persona_id}"
            )
        if scene_id not in world.scene_ids:
            raise ValueError(f"Unknown scene for world {world_id}: {scene_id}")

        persona = self.loader.load_persona(turn_input.active_persona_id)
        scene = self.loader.load_scene(scene_id)
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
        messages = build_actor_messages(persona=persona, scene=scene, turn_input=turn_input)
        text = await self.actor_agent.generate(
            provider=self.provider,
            route=route,
            messages=messages,
        )
        return TurnResult(text=text, route=route)
