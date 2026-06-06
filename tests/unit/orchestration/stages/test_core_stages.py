from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from app.domain import (
    PersonaCard,
    RetrievedChunk,
    SceneState,
    SessionState,
    StoredTurn,
    TurnInput,
    Visibility,
)
from app.llm.router import CloudMode, ModelProviderName, ModelRoute
from app.orchestration.context_budget import ContextBudget
from app.orchestration.stages import (
    LoadedTurnContext,
    TurnPersistenceStage,
    TurnRetrievalStage,
    TurnRoutingStage,
)


def _context() -> LoadedTurnContext:
    return LoadedTurnContext(
        session=SessionState(
            id="session",
            world_id="world",
            active_scene_id="scene",
            active_persona_id="persona",
            player_name="Player",
        ),
        persona=PersonaCard(
            id="persona",
            name="Archivist",
            role="npc",
            public_description="A careful archivist.",
            speaking_style="Precise.",
        ),
        scene=SceneState(
            id="scene",
            title="Gallery",
            location="Palace",
            player_visible_summary="A mirrored gallery.",
        ),
        recent_turns=(),
    )


def _routing(*, cloud_mode: CloudMode = CloudMode.ASK) -> TurnRoutingStage:
    return TurnRoutingStage(
        local_model="local",
        cloud_model="cloud",
        local_max_tokens=700,
        local_structured_max_tokens=350,
        cloud_max_tokens=1000,
        local_temperature=0.75,
        cloud_temperature=0.65,
        cloud_mode=cloud_mode,
    )


def test_retrieval_stage_uses_only_player_visible_scores_for_confidence() -> None:
    chunks = [
        RetrievedChunk(
            id="visible",
            source="lore.md",
            source_type="lore",
            text="Visible",
            score=0.6,
            visibility=Visibility.PLAYER,
        ),
        RetrievedChunk(
            id="hidden",
            source="gm.md",
            source_type="lore",
            text="Hidden",
            score=0.99,
            visibility=Visibility.GM,
        ),
    ]
    retriever = SimpleNamespace(retrieve_for_actor=lambda **_: chunks)
    stage = TurnRetrievalStage(
        actor_context_retriever=retriever,
        context_budget=ContextBudget(retrieved_chunks=3),
    )

    result = stage.run(
        turn_input=TurnInput(session_id="session", message="Look around."),
        context=_context(),
    )

    assert result.chunks == tuple(chunks)
    assert result.confidence == 0.6
    assert result.warnings == ()
    with pytest.raises(AttributeError):
        result.confidence = 0.1  # type: ignore[misc]


def test_retrieval_stage_degrades_to_warning() -> None:
    def fail(**_: object) -> list[RetrievedChunk]:
        raise RuntimeError("offline")

    stage = TurnRetrievalStage(
        actor_context_retriever=SimpleNamespace(retrieve_for_actor=fail),
        context_budget=ContextBudget(),
    )

    result = stage.run(
        turn_input=TurnInput(session_id="session", message="Look around."),
        context=_context(),
    )

    assert result.chunks == ()
    assert result.confidence is None
    assert result.warnings == ("retrieval skipped: offline",)


def test_routing_stage_normalizes_confirmation_required_actor_route() -> None:
    result = _routing().actor(
        turn_input=TurnInput(
            session_id="session",
            message="Use cloud.",
            user_requested_cloud=True,
        ),
        scene=_context().scene,
        retrieval_confidence=None,
    )

    assert result.route.provider == ModelProviderName.LOCAL
    assert result.route.reason == "confirmation required before cloud route: user requested cloud"
    assert result.warnings == (
        "cloud actor skipped: confirmation required for cloud (user requested cloud)",
    )


def test_persistence_stage_appends_before_updating_session_activity() -> None:
    calls: list[tuple[str, object]] = []
    created_at = datetime(2026, 1, 2, tzinfo=UTC)
    route = ModelRoute(
        provider=ModelProviderName.LOCAL,
        model="local",
        max_tokens=700,
        temperature=0.75,
        reason="default local route",
    )
    stored_turn = StoredTurn(
        id=1,
        session_id="session",
        turn_index=1,
        scene_id="scene",
        persona_id="persona",
        user_message="Question",
        assistant_message="Answer",
        route=route,
        created_at=created_at,
    )

    class TurnRepository:
        def append_turn(self, **_: object) -> StoredTurn:
            calls.append(("append", created_at))
            return stored_turn

    class SessionRepository:
        def update_session_activity(
            self,
            session_id: str,
            *,
            updated_at: datetime,
        ) -> None:
            calls.append((session_id, updated_at))

    stage = TurnPersistenceStage(
        session_repository=SessionRepository(),  # type: ignore[arg-type]
        turn_repository=TurnRepository(),  # type: ignore[arg-type]
    )
    stage.run(
        session=_context().session,
        user_message="Question",
        assistant_message="Answer",
        route=route,
    )

    assert calls == [("append", created_at), ("session", created_at)]
