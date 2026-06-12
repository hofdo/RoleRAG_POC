from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.domain import (
    CriticResult,
    CriticStatus,
    MemoryCandidate,
    MemoryCuratorResult,
    MemoryEpisode,
    PersonaCard,
    RetrievedChunk,
    SceneState,
    SessionState,
    StoredTurn,
    TurnInput,
    TurnOutcome,
    TurnResult,
    Visibility,
)
from app.llm.router import ModelProviderName, ModelRoute


def test_persona_card_accepts_valid_role_and_preserves_private_fields() -> None:
    persona = PersonaCard(
        id="npc-1",
        name="Iria Vale",
        role="npc",
        public_description="A composed archivist with a careful smile.",
        private_description="She is hiding her role in the coup.",
        speaking_style="Precise and lightly sardonic.",
        values=["order"],
        fears=["exposure"],
        goals=["protect the archive"],
        secrets=["backed the wrong claimant"],
        forbidden_knowledge=["the vault key sequence"],
        relationships={"player": "useful outsider"},
    )

    assert persona.role == "npc"
    assert persona.public_description != persona.private_description
    assert persona.secrets == ["backed the wrong claimant"]
    assert persona.forbidden_knowledge == ["the vault key sequence"]


def test_persona_card_serializes_empty_collections_with_safe_defaults() -> None:
    persona = PersonaCard(
        id="narrator",
        name="Narrator",
        role="narrator",
        public_description="Sets the scene.",
        speaking_style="Measured and vivid.",
    )

    assert persona.model_dump() == {
        "id": "narrator",
        "name": "Narrator",
        "role": "narrator",
        "public_description": "Sets the scene.",
        "private_description": None,
        "speaking_style": "Measured and vivid.",
        "values": [],
        "fears": [],
        "goals": [],
        "secrets": [],
        "forbidden_knowledge": [],
        "relationships": {},
    }


def test_scene_state_requires_player_visible_summary() -> None:
    with pytest.raises(ValidationError):
        SceneState(
            id="scene-1",
            title="Rose Gallery",
            location="Winter Palace",
        )  # type: ignore[call-arg]


def test_scene_state_serializes_safe_empty_defaults() -> None:
    scene = SceneState(
        id="scene-1",
        title="Rose Gallery",
        location="Winter Palace",
        player_visible_summary="Courtiers circle the player with polite suspicion.",
    )

    assert scene.model_dump() == {
        "id": "scene-1",
        "title": "Rose Gallery",
        "location": "Winter Palace",
        "current_time": None,
        "active_personas": [],
        "player_visible_summary": "Courtiers circle the player with polite suspicion.",
        "gm_private_summary": None,
        "open_conflicts": [],
        "active_quests": [],
        "recent_events": [],
    }


def test_session_state_allows_omitted_timestamps() -> None:
    session = SessionState(
        id="session-1",
        world_id="world-1",
        active_scene_id="scene-1",
        active_persona_id="archivist",
        player_name="Avery",
    )

    assert session.created_at is None
    assert session.updated_at is None
    assert session.recent_turn_ids == []
    assert session.content_root == "data"


def test_session_state_serializes_present_datetimes_cleanly() -> None:
    created_at = datetime(2026, 5, 27, 10, 30, tzinfo=UTC)
    updated_at = datetime(2026, 5, 27, 11, 45, tzinfo=UTC)
    session = SessionState(
        id="session-1",
        world_id="world-1",
        active_scene_id="scene-1",
        active_persona_id="archivist",
        player_name="Avery",
        created_at=created_at,
        updated_at=updated_at,
    )

    assert session.model_dump(mode="json") == {
        "id": "session-1",
        "world_id": "world-1",
        "active_scene_id": "scene-1",
        "active_persona_id": "archivist",
        "player_name": "Avery",
        "content_root": "data",
        "recent_turn_ids": [],
        "created_at": "2026-05-27T10:30:00Z",
        "updated_at": "2026-05-27T11:45:00Z",
    }


def test_memory_episode_requires_visibility() -> None:
    with pytest.raises(ValidationError):
        MemoryEpisode(
            id="mem-1",
            session_id="session-1",
            scene_id="scene-1",
            summary="The player noticed the duke's ring.",
            importance=3,
        )  # type: ignore[call-arg]


@pytest.mark.parametrize("visibility", ["secret", "public", ""])
def test_memory_episode_rejects_invalid_visibility(visibility: str) -> None:
    with pytest.raises(ValidationError):
        MemoryEpisode(
            id="mem-1",
            session_id="session-1",
            scene_id="scene-1",
            summary="The player noticed the duke's ring.",
            importance=3,
            visibility=visibility,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("importance", [0, 6])
def test_memory_episode_rejects_importance_outside_range(importance: int) -> None:
    with pytest.raises(ValidationError):
        MemoryEpisode(
            id="mem-1",
            session_id="session-1",
            scene_id="scene-1",
            summary="The player noticed the duke's ring.",
            importance=importance,
            visibility=Visibility.PLAYER,
        )


def test_memory_candidate_requires_visibility_and_constrained_importance() -> None:
    candidate = MemoryCandidate(
        summary="The player promised to return before dawn.",
        visibility=Visibility.PLAYER,
        importance=4,
        tags=["promise"],
    )

    assert candidate.visibility == Visibility.PLAYER
    assert candidate.importance == 4


def test_memory_curator_result_write_without_memories_becomes_decline() -> None:
    result = MemoryCuratorResult(
        write_memory=True,
        memories=[],
        reason="This matters later.",
    )

    assert result.write_memory is False


@pytest.mark.parametrize("visibility", ["wrong", "internal"])
def test_retrieved_chunk_rejects_invalid_visibility(visibility: str) -> None:
    with pytest.raises(ValidationError):
        RetrievedChunk(
            id="chunk-1",
            source="memory",
            source_type="lore",
            text="The old treaty was burned.",
            score=0.82,
            visibility=visibility,  # type: ignore[arg-type]
        )


def test_retrieved_chunk_accepts_optional_metadata_when_absent() -> None:
    chunk = RetrievedChunk(
        id="chunk-1",
        source="memory",
        source_type="lore",
        text="The old treaty was burned.",
        score=0.82,
        visibility=Visibility.GM,
    )

    assert chunk.model_dump() == {
        "id": "chunk-1",
        "source": "memory",
        "source_type": "lore",
        "text": "The old treaty was burned.",
        "score": 0.82,
        "visibility": Visibility.GM,
        "tags": [],
        "world_id": None,
        "scene_id": None,
        "persona_id": None,
        "session_id": None,
        "actor_id": None,
        "importance": None,
    }


def test_retrieved_chunk_serializes_tags_and_metadata_predictably() -> None:
    chunk = RetrievedChunk(
        id="chunk-2",
        source="vector-store",
        source_type="memory_episode",
        text="Captain Soren never trusted the regent.",
        score=0.97,
        visibility=Visibility.CHARACTER_PRIVATE,
        tags=["trust", "regent"],
        world_id="world-1",
        scene_id="scene-4",
        persona_id="captain-soren",
        session_id="session-1",
        actor_id="captain-soren",
        importance=4,
    )

    assert chunk.model_dump() == {
        "id": "chunk-2",
        "source": "vector-store",
        "source_type": "memory_episode",
        "text": "Captain Soren never trusted the regent.",
        "score": 0.97,
        "visibility": Visibility.CHARACTER_PRIVATE,
        "tags": ["trust", "regent"],
        "world_id": "world-1",
        "scene_id": "scene-4",
        "persona_id": "captain-soren",
        "session_id": "session-1",
        "actor_id": "captain-soren",
        "importance": 4,
    }


def test_turn_input_allows_session_backed_persona_lookup() -> None:
    turn_input = TurnInput(
        session_id="session-1",
        message="I ask the archivist what she really knows.",
    )

    assert turn_input.active_persona_id is None


def test_turn_result_accepts_nested_model_route() -> None:
    route = ModelRoute(
        provider=ModelProviderName.LOCAL,
        model="local-model",
        max_tokens=900,
        temperature=0.6,
        reason="default local route",
    )
    result = TurnResult(text="The archivist hesitates before answering.", route=route)

    assert result.route is route
    assert result.outcome == TurnOutcome.SUCCESS
    assert result.model_dump() == {
        "text": "The archivist hesitates before answering.",
        "route": {
            "provider": ModelProviderName.LOCAL,
            "model": "local-model",
            "max_tokens": 900,
            "temperature": 0.6,
            "reason": "default local route",
            "requires_user_confirmation": False,
        },
        "finish_reason": None,
        "memory_written": False,
        "critic_status": CriticStatus.SKIPPED,
        "warnings": [],
        "retrieval": None,
        "stage_timings": {},
    }


def test_turn_result_excludes_controlled_failure_outcome_from_serialization() -> None:
    route = ModelRoute(
        provider=ModelProviderName.LOCAL,
        model="local-model",
        max_tokens=900,
        temperature=0.6,
        reason="cloud mode is off",
    )
    result = TurnResult(
        text="No validated response was produced.",
        route=route,
        outcome=TurnOutcome.CONTROLLED_FAILURE,
    )

    assert result.outcome == TurnOutcome.CONTROLLED_FAILURE
    assert "outcome" not in result.model_dump()


def test_critic_result_defaults_to_empty_issues_and_nullable_repair_instruction() -> None:
    result = CriticResult(accepted=True)

    assert result.issues == []
    assert result.repair_instruction is None


def test_stored_turn_accepts_route_and_timestamp() -> None:
    route = ModelRoute(
        provider=ModelProviderName.LOCAL,
        model="local-model",
        max_tokens=700,
        temperature=0.75,
        reason="default local route",
    )
    stored_turn = StoredTurn(
        id=3,
        session_id="session-1",
        turn_index=2,
        scene_id="scene-1",
        persona_id="archivist",
        user_message="Tell me the truth.",
        assistant_message="Not here.",
        route=route,
        created_at=datetime(2026, 5, 27, 12, 0, tzinfo=UTC),
    )

    assert stored_turn.route.model == "local-model"
    assert stored_turn.turn_index == 2
