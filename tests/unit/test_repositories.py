from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from app.domain import SessionState
from app.llm.router import ModelProviderName, ModelRoute
from app.persistence.repositories import SQLiteSessionRepository, SQLiteTurnRepository
from app.persistence.sqlite import connect_sqlite, initialize_database


def _build_route() -> ModelRoute:
    return ModelRoute(
        provider=ModelProviderName.LOCAL,
        model="local-model",
        max_tokens=700,
        temperature=0.75,
        reason="default local route",
    )


def test_session_repository_creates_and_loads_session(tmp_path: Path) -> None:
    connection = connect_sqlite(tmp_path / "sessions.db")
    initialize_database(connection)
    repository = SQLiteSessionRepository(connection)
    session = SessionState(
        id="session-1",
        world_id="demo_world",
        active_scene_id="rose-gallery",
        active_persona_id="archivist",
        player_name="Avery",
    )

    created = repository.create_session(session)
    loaded = repository.get_session("session-1")

    assert created.created_at is not None
    assert created.updated_at is not None
    assert loaded == created


def test_turn_repository_appends_turn_and_loads_recent_turns_in_order(tmp_path: Path) -> None:
    connection = connect_sqlite(tmp_path / "sessions.db")
    initialize_database(connection)
    session_repository = SQLiteSessionRepository(connection)
    turn_repository = SQLiteTurnRepository(connection)
    session_repository.create_session(
        SessionState(
            id="session-1",
            world_id="demo_world",
            active_scene_id="rose-gallery",
            active_persona_id="archivist",
            player_name="Avery",
        )
    )

    first = turn_repository.append_turn(
        session_id="session-1",
        scene_id="rose-gallery",
        persona_id="archivist",
        user_message="First question",
        assistant_message="First answer",
        route=_build_route(),
    )
    second = turn_repository.append_turn(
        session_id="session-1",
        scene_id="rose-gallery",
        persona_id="archivist",
        user_message="Second question",
        assistant_message="Second answer",
        route=_build_route(),
    )

    turns = turn_repository.list_recent_turns("session-1", limit=2)

    assert turns == [first, second]
    assert turns[0].turn_index == 1
    assert turns[1].turn_index == 2
    assert turns[1].route.reason == "default local route"


def test_session_repository_updates_activity_timestamp(tmp_path: Path) -> None:
    connection = connect_sqlite(tmp_path / "sessions.db")
    initialize_database(connection)
    repository = SQLiteSessionRepository(connection)
    created = repository.create_session(
        SessionState(
            id="session-1",
            world_id="demo_world",
            active_scene_id="rose-gallery",
            active_persona_id="archivist",
            player_name="Avery",
        )
    )
    later = datetime(2026, 5, 27, 12, 0, tzinfo=UTC)

    repository.update_session_activity("session-1", updated_at=later)

    loaded = repository.get_session("session-1")
    assert loaded is not None
    assert loaded.created_at == created.created_at
    assert loaded.updated_at == later
