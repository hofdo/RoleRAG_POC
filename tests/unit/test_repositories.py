from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.domain import (
    CriticStatus,
    MemoryCandidate,
    SessionState,
    TurnDiagnostics,
    Visibility,
)
from app.llm.router import ModelProviderName, ModelRoute
from app.persistence.repositories import (
    SQLiteMemoryRepository,
    SQLiteSessionRepository,
    SQLiteTurnRepository,
)
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
    assert loaded is not None
    assert loaded.content_root == "data"


def test_session_repository_persists_custom_content_root(tmp_path: Path) -> None:
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
            content_root=str(tmp_path / "scenario-pack"),
        )
    )

    loaded = repository.get_session("session-1")
    assert loaded == created
    assert loaded is not None
    assert loaded.content_root == str(tmp_path / "scenario-pack")


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


def test_turn_repository_round_trips_route_confirmation_metadata(tmp_path: Path) -> None:
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

    route = ModelRoute(
        provider=ModelProviderName.CLOUD,
        model="cloud-model",
        max_tokens=1000,
        temperature=0.65,
        reason="user requested cloud",
        requires_user_confirmation=True,
    )
    turn_repository.append_turn(
        session_id="session-1",
        scene_id="rose-gallery",
        persona_id="archivist",
        user_message="Please use the better model.",
        assistant_message="I will answer carefully.",
        route=route,
    )

    turns = turn_repository.list_recent_turns("session-1", limit=1)

    assert turns[0].route.provider == ModelProviderName.CLOUD
    assert turns[0].route.reason == "user requested cloud"
    assert turns[0].route.requires_user_confirmation is True


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


def test_session_repository_lists_recent_sessions_with_limit_and_tie_breaker(
    tmp_path: Path,
) -> None:
    connection = connect_sqlite(tmp_path / "sessions.db")
    initialize_database(connection)
    repository = SQLiteSessionRepository(connection)
    base = datetime(2026, 5, 27, 12, 0, tzinfo=UTC)
    for index in range(3):
        repository.create_session(
            SessionState(
                id=f"session-{index}",
                world_id="demo_world",
                active_scene_id="rose-gallery",
                active_persona_id="archivist",
                player_name=f"Player {index}",
                created_at=base + timedelta(minutes=index),
                updated_at=base + timedelta(hours=index),
            )
        )
    repository.create_session(
        SessionState(
            id="latest-created-tie",
            world_id="demo_world",
            active_scene_id="rose-gallery",
            active_persona_id="archivist",
            player_name="Tie",
            created_at=base + timedelta(days=1),
            updated_at=base + timedelta(hours=2),
        )
    )

    sessions = repository.list_recent_sessions(limit=3)

    assert [session.id for session in sessions] == [
        "latest-created-tie",
        "session-2",
        "session-1",
    ]
    assert repository.list_recent_sessions(limit=0) == []


def test_memory_repository_persists_and_loads_memory_episodes(tmp_path: Path) -> None:
    connection = connect_sqlite(tmp_path / "sessions.db")
    initialize_database(connection)
    session_repository = SQLiteSessionRepository(connection)
    memory_repository = SQLiteMemoryRepository(connection)
    session_repository.create_session(
        SessionState(
            id="session-1",
            world_id="demo_world",
            active_scene_id="rose-gallery",
            active_persona_id="archivist",
            player_name="Avery",
        )
    )

    episodes = memory_repository.append_memories(
        session_id="session-1",
        memories=[
            MemoryCandidate(
                summary="The player promised to return before dawn.",
                visibility=Visibility.PLAYER,
                importance=4,
                tags=["promise", "deadline"],
                scene_id="rose-gallery",
                actor_id="archivist",
            )
        ],
    )
    loaded = memory_repository.list_memories_for_session("session-1")

    assert len(episodes) == 1
    assert loaded == episodes
    assert loaded[0].tags == ["promise", "deadline"]
    assert episodes[0].created_at is not None
    assert loaded[0].created_at == episodes[0].created_at


def test_add_tag_to_memories_marks_episodes_idempotently(tmp_path: Path) -> None:
    from app.domain import MemoryCandidate, Visibility
    from app.persistence import SQLiteMemoryRepository

    connection = connect_sqlite(tmp_path / "sessions.db")
    initialize_database(connection)
    session_repository = SQLiteSessionRepository(connection)
    memory_repository = SQLiteMemoryRepository(connection)
    session_repository.create_session(
        SessionState(
            id="session-1",
            world_id="demo_world",
            active_scene_id="rose-gallery",
            active_persona_id="archivist",
            player_name="Avery",
        )
    )
    episodes = memory_repository.append_memories(
        session_id="session-1",
        memories=[
            MemoryCandidate(
                summary="A minor observation.",
                visibility=Visibility.PLAYER,
                importance=1,
                tags=["mood"],
                scene_id="rose-gallery",
                actor_id="archivist",
            )
        ],
    )

    memory_repository.add_tag_to_memories([episodes[0].id], "consolidated")
    memory_repository.add_tag_to_memories([episodes[0].id], "consolidated")  # idempotent

    loaded = memory_repository.list_memories_for_session("session-1")
    assert loaded[0].tags == ["mood", "consolidated"]


def test_canon_repository_adds_lists_and_deletes_facts(tmp_path: Path) -> None:
    from app.persistence import SQLiteCanonRepository

    connection = connect_sqlite(tmp_path / "sessions.db")
    initialize_database(connection)
    session_repository = SQLiteSessionRepository(connection)
    canon_repository = SQLiteCanonRepository(connection)
    session_repository.create_session(
        SessionState(
            id="session-1",
            world_id="demo_world",
            active_scene_id="rose-gallery",
            active_persona_id="archivist",
            player_name="Avery",
        )
    )

    fact = canon_repository.add_canon_fact(session_id="session-1", text="The east gate stays shut.")
    canon_repository.add_canon_fact(session_id="session-1", text="Trust only the blue seal.")

    assert [item.text for item in canon_repository.list_canon_facts("session-1")] == [
        "The east gate stays shut.",
        "Trust only the blue seal.",
    ]
    assert canon_repository.delete_canon_fact(session_id="session-1", fact_id=fact.id) is True
    assert [item.text for item in canon_repository.list_canon_facts("session-1")] == [
        "Trust only the blue seal.",
    ]
    assert canon_repository.delete_canon_fact(session_id="session-1", fact_id="missing") is False


def _seed_session_with_data(tmp_path: Path) -> tuple[
    SQLiteSessionRepository, SQLiteTurnRepository, SQLiteMemoryRepository
]:
    connection = connect_sqlite(tmp_path / "sessions.db")
    initialize_database(connection)
    sessions = SQLiteSessionRepository(connection)
    turns = SQLiteTurnRepository(connection)
    memories = SQLiteMemoryRepository(connection)
    sessions.create_session(
        SessionState(
            id="session-1",
            world_id="demo_world",
            active_scene_id="rose-gallery",
            active_persona_id="archivist",
            player_name="Avery",
        )
    )
    for index in range(3):
        turns.append_turn(
            session_id="session-1",
            scene_id="rose-gallery",
            persona_id="archivist",
            user_message=f"player message {index + 1}",
            assistant_message=f"actor message {index + 1}",
            route=_build_route(),
        )
    memories.append_memories(
        session_id="session-1",
        memories=[
            MemoryCandidate(
                summary="The player promised to return before dawn.",
                visibility=Visibility.PLAYER,
                importance=4,
                tags=["promise"],
                scene_id="rose-gallery",
                actor_id="archivist",
            )
        ],
    )
    return sessions, turns, memories


def test_session_repository_delete_removes_session_turns_and_memories(
    tmp_path: Path,
) -> None:
    sessions, turns, memories = _seed_session_with_data(tmp_path)

    assert sessions.delete_session("session-1") is True

    assert sessions.get_session("session-1") is None
    assert turns.count_turns("session-1") == 0
    assert memories.list_memories_for_session("session-1") == []


def test_session_repository_delete_returns_false_for_unknown_session(
    tmp_path: Path,
) -> None:
    sessions, _, _ = _seed_session_with_data(tmp_path)

    assert sessions.delete_session("missing") is False
    assert sessions.get_session("session-1") is not None


def test_turn_repository_lists_all_turns_in_order(tmp_path: Path) -> None:
    _, turns, _ = _seed_session_with_data(tmp_path)

    listed = turns.list_all_turns("session-1")

    assert [turn.turn_index for turn in listed] == [1, 2, 3]
    assert listed[0].user_message == "player message 1"


def test_turn_repository_persists_and_loads_diagnostics(tmp_path: Path) -> None:
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
    turn = turn_repository.append_turn(
        session_id="session-1",
        scene_id="rose-gallery",
        persona_id="archivist",
        user_message="First question",
        assistant_message="First answer",
        route=_build_route(),
    )

    # A freshly appended turn has no diagnostics yet.
    assert turn_repository.list_recent_turns("session-1", limit=1)[0].diagnostics is None

    diagnostics = TurnDiagnostics(
        retrieval=None,
        stage_timings={"gen": 0.5, "critique": 0.2},
        critic_status=CriticStatus.ACCEPTED,
        finish_reason="stop",
        warnings=["test"],
        memory_written=True,
    )
    turn_repository.update_turn_diagnostics(turn.id, diagnostics)

    reloaded = turn_repository.list_recent_turns("session-1", limit=1)[0]
    assert reloaded.diagnostics == diagnostics
    assert turn_repository.list_all_turns("session-1")[0].diagnostics == diagnostics
