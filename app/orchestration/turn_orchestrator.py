from __future__ import annotations

from app.agents import ActorAgent
from app.domain import PersonaCard, SceneState, TurnInput, TurnResult
from app.llm.provider import LlmProvider
from app.llm.router import CloudMode, ModelTask, choose_route
from app.orchestration.context_builder import build_actor_messages

DEMO_PERSONA = PersonaCard(
    id="archivist",
    name="Iria Vale",
    role="npc",
    public_description="A composed palace archivist with an excellent memory.",
    private_description="She is quietly aiding the coup against the regent.",
    speaking_style="Precise, dry, and difficult to rattle.",
    values=["order", "discretion"],
    goals=["protect the archive", "limit political damage"],
    secrets=["She forged one inventory ledger to hide a missing document."],
    forbidden_knowledge=["The vault key sequence beneath the rose seal."],
)

DEMO_SCENE = SceneState(
    id="rose-gallery",
    title="Rose Gallery",
    location="Winter Palace",
    current_time="late evening",
    active_personas=["archivist"],
    player_visible_summary="Courtiers drift between mirrors and roses while musicians play softly.",
    gm_private_summary="A regent loyalist is watching every exchange from the south alcove.",
    recent_events=[
        "A servant spilled wine near the west door.",
        "The regent left the chamber moments ago.",
    ],
)


class TurnOrchestrator:
    def __init__(
        self,
        *,
        provider: LlmProvider,
        local_model: str,
        cloud_model: str,
        local_max_tokens: int,
        cloud_max_tokens: int,
        local_temperature: float,
        cloud_temperature: float,
        cloud_mode: CloudMode | str,
    ) -> None:
        self.provider = provider
        self.actor_agent = ActorAgent()
        self.local_model = local_model
        self.cloud_model = cloud_model
        self.local_max_tokens = local_max_tokens
        self.cloud_max_tokens = cloud_max_tokens
        self.local_temperature = local_temperature
        self.cloud_temperature = cloud_temperature
        self.cloud_mode = CloudMode(cloud_mode)

    async def run_turn(self, turn_input: TurnInput) -> TurnResult:
        persona = self._get_demo_persona(turn_input.active_persona_id)
        scene = DEMO_SCENE
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

    def _get_demo_persona(self, active_persona_id: str) -> PersonaCard:
        if active_persona_id != DEMO_PERSONA.id:
            raise ValueError(f"Unknown demo persona: {active_persona_id}")
        return DEMO_PERSONA
