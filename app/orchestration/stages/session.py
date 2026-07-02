from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

from app.domain import PersonaCard, SceneState, SessionState, StoredTurn, TurnInput
from app.memory import RecentDialogueStore
from app.memory.store import MemoryEpisodeStore
from app.orchestration.canon_builder import build_standing_facts
from app.persistence import DemoWorldRecord, SessionNotFoundError
from app.persistence.repositories import CanonRepository, SessionRepository


class TurnDataLoader(Protocol):
    def load_world(self, world_id: str) -> DemoWorldRecord: ...

    def load_persona(self, persona_id: str) -> PersonaCard: ...

    def load_scene(self, scene_id: str) -> SceneState: ...


class TurnDataLoaderFactory(Protocol):
    def __call__(self, content_root: str) -> TurnDataLoader: ...


@dataclass(frozen=True)
class LoadedTurnContext:
    session: SessionState
    persona: PersonaCard
    scene: SceneState
    recent_turns: tuple[StoredTurn, ...]
    standing_facts: tuple[str, ...] = ()
    # True when this turn carries a valid persona override that has not yet been
    # written to the session repository. The durable write is deferred until the
    # turn actually persists (see TurnOrchestrator.run_turn), so a failed turn
    # (CONTROLLED_FAILURE, confirmation required, provider error) never commits a
    # persona switch that the player never actually saw succeed.
    persona_switched: bool = False


class TurnSessionLoader:
    def __init__(
        self,
        *,
        loader: TurnDataLoader,
        loader_factory: TurnDataLoaderFactory | None,
        content_root: str,
        session_repository: SessionRepository,
        recent_dialogue_store: RecentDialogueStore,
        memory_store: MemoryEpisodeStore | None = None,
        canon_repository: CanonRepository | None = None,
        canon_importance_floor: int = 4,
        canon_max_items: int = 8,
        canon_max_chars: int = 900,
    ) -> None:
        self.loader = loader
        self.loader_factory = loader_factory
        self.content_root = content_root
        self.session_repository = session_repository
        self.recent_dialogue_store = recent_dialogue_store
        self.memory_store = memory_store
        self.canon_repository = canon_repository
        self.canon_importance_floor = canon_importance_floor
        self.canon_max_items = canon_max_items
        self.canon_max_chars = canon_max_chars

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
        resolved_content_root = content_root or self.content_root
        loader = self.loader_for_content_root(resolved_content_root)
        world = loader.load_world(world_id)
        if active_persona_id not in world.persona_ids:
            raise ValueError(f"Unknown persona for world {world_id}: {active_persona_id}")
        if scene_id not in world.scene_ids:
            raise ValueError(f"Unknown scene for world {world_id}: {scene_id}")
        loader.load_persona(active_persona_id)
        loader.load_scene(scene_id)
        return self.session_repository.create_session(
            SessionState(
                id=session_id or str(uuid4()),
                world_id=world_id,
                active_scene_id=scene_id,
                active_persona_id=active_persona_id,
                player_name=player_name,
                content_root=resolved_content_root,
            )
        )

    def resume_session(self, session_id: str) -> SessionState:
        session = self.session_repository.get_session(session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        return session

    def load(self, turn_input: TurnInput) -> LoadedTurnContext:
        session = self.resume_session(turn_input.session_id)
        persona_id = session.active_persona_id
        loader = self.loader_for_content_root(session.content_root)
        world = loader.load_world(session.world_id)
        persona_switched = False
        if (
            turn_input.active_persona_id is not None
            and turn_input.active_persona_id != persona_id
        ):
            if turn_input.active_persona_id not in world.persona_ids:
                raise ValueError(
                    f"Unknown persona for world {session.world_id}: "
                    f"{turn_input.active_persona_id}"
                )
            persona_id = turn_input.active_persona_id
            # Do NOT write through to the repository here: a durable switch before
            # the turn outcome is known would leave a failed turn (controlled
            # failure, provider error, confirmation required) with the persona
            # switch committed anyway. The in-memory session is updated so this
            # turn generates/persists under the new persona; the caller commits the
            # switch to the repository only after the turn actually persists (see
            # TurnOrchestrator.run_turn, which checks `persona_switched`).
            session = session.model_copy(update={"active_persona_id": persona_id})
            persona_switched = True

        if persona_id not in world.persona_ids:
            raise ValueError(f"Unknown persona for world {session.world_id}: {persona_id}")
        if session.active_scene_id not in world.scene_ids:
            raise ValueError(
                f"Unknown scene for world {session.world_id}: {session.active_scene_id}"
            )

        pinned_canon = (
            [fact.text for fact in self.canon_repository.list_canon_facts(session.id)]
            if self.canon_repository is not None
            else []
        )
        memories = (
            self.memory_store.list_memories_for_session(session.id)
            if self.memory_store is not None
            else []
        )
        standing_facts: tuple[str, ...] = ()
        if pinned_canon or memories:
            standing_facts = build_standing_facts(
                memories,
                pinned=pinned_canon,
                importance_floor=self.canon_importance_floor,
                max_items=self.canon_max_items,
                max_chars=self.canon_max_chars,
            )

        return LoadedTurnContext(
            session=session,
            persona=loader.load_persona(persona_id),
            scene=loader.load_scene(session.active_scene_id),
            recent_turns=tuple(self.recent_dialogue_store.load_recent_dialogue(session.id)),
            standing_facts=standing_facts,
            persona_switched=persona_switched,
        )

    def loader_for_content_root(self, content_root: str) -> TurnDataLoader:
        if self.loader_factory is None:
            return self.loader
        return self.loader_factory(content_root)
