from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.cli import app
from app.domain import MemoryCandidate, SessionState, Visibility
from app.llm.router import ModelProviderName, ModelRoute
from app.persistence import (
    SQLiteMemoryRepository,
    SQLiteSessionRepository,
    SQLiteTurnRepository,
)
from app.persistence.sqlite import connect_sqlite, initialize_database

runner = CliRunner()


def _seed_database(tmp_path: Path) -> Path:
    database_path = tmp_path / "sessions.db"
    connection = connect_sqlite(database_path)
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
    route = ModelRoute(
        provider=ModelProviderName.LOCAL,
        model="local-model",
        max_tokens=700,
        temperature=0.75,
        reason="default local route",
    )
    for index in range(2):
        turns.append_turn(
            session_id="session-1",
            scene_id="rose-gallery",
            persona_id="archivist",
            user_message=f"player message {index + 1}",
            assistant_message=f"actor message {index + 1}",
            route=route,
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
    connection.close()
    return database_path


def test_list_sessions_shows_seeded_session(tmp_path: Path) -> None:
    database_path = _seed_database(tmp_path)

    result = runner.invoke(
        app,
        ["list-sessions"],
        env={"DATABASE_PATH": str(database_path)},
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["sessions"][0]["session_id"] == "session-1"
    assert payload["sessions"][0]["turn_count"] == 2


def test_delete_session_removes_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    database_path = _seed_database(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        ["delete-session", "--session-id", "session-1", "--yes"],
        env={"DATABASE_PATH": str(database_path)},
    )

    assert result.exit_code == 0
    connection = connect_sqlite(database_path)
    assert SQLiteSessionRepository(connection).get_session("session-1") is None
    assert SQLiteTurnRepository(connection).count_turns("session-1") == 0
    connection.close()


def test_delete_session_unknown_id_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    database_path = _seed_database(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        ["delete-session", "--session-id", "missing", "--yes"],
        env={"DATABASE_PATH": str(database_path)},
    )

    assert result.exit_code == 1


def test_export_import_round_trip_preserves_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database_path = _seed_database(tmp_path)
    export_path = tmp_path / "session-1.json"
    monkeypatch.chdir(tmp_path)

    export_result = runner.invoke(
        app,
        [
            "export-session",
            "--session-id",
            "session-1",
            "--output",
            str(export_path),
        ],
        env={"DATABASE_PATH": str(database_path)},
    )
    assert export_result.exit_code == 0
    envelope = json.loads(export_path.read_text(encoding="utf-8"))
    assert envelope["format_version"] == 1
    assert envelope["session"]["id"] == "session-1"
    assert len(envelope["turns"]) == 2
    assert len(envelope["memory_episodes"]) == 1

    delete_result = runner.invoke(
        app,
        ["delete-session", "--session-id", "session-1", "--yes"],
        env={"DATABASE_PATH": str(database_path)},
    )
    assert delete_result.exit_code == 0

    import_result = runner.invoke(
        app,
        ["import-session", "--input", str(export_path)],
        env={"DATABASE_PATH": str(database_path)},
    )
    assert import_result.exit_code == 0

    connection = connect_sqlite(database_path)
    restored = SQLiteSessionRepository(connection).get_session("session-1")
    assert restored is not None
    assert restored.player_name == "Avery"
    turns = SQLiteTurnRepository(connection).list_all_turns("session-1")
    assert [turn.user_message for turn in turns] == [
        "player message 1",
        "player message 2",
    ]
    memories = SQLiteMemoryRepository(connection).list_memories_for_session("session-1")
    assert [memory.summary for memory in memories] == [
        "The player promised to return before dawn."
    ]
    connection.close()


def test_import_session_refuses_existing_id(tmp_path: Path) -> None:
    database_path = _seed_database(tmp_path)
    export_path = tmp_path / "session-1.json"
    runner.invoke(
        app,
        ["export-session", "--session-id", "session-1", "--output", str(export_path)],
        env={"DATABASE_PATH": str(database_path)},
    )

    result = runner.invoke(
        app,
        ["import-session", "--input", str(export_path)],
        env={"DATABASE_PATH": str(database_path)},
    )

    assert result.exit_code == 1
    assert "already exists" in result.output


def test_import_session_with_new_id_remaps_children(tmp_path: Path) -> None:
    database_path = _seed_database(tmp_path)
    export_path = tmp_path / "session-1.json"
    runner.invoke(
        app,
        ["export-session", "--session-id", "session-1", "--output", str(export_path)],
        env={"DATABASE_PATH": str(database_path)},
    )

    result = runner.invoke(
        app,
        ["import-session", "--input", str(export_path), "--new-id"],
        env={"DATABASE_PATH": str(database_path)},
    )

    assert result.exit_code == 0
    new_session_id = json.loads(result.stdout)["session_id"]
    assert new_session_id != "session-1"
    connection = connect_sqlite(database_path)
    assert SQLiteTurnRepository(connection).count_turns(new_session_id) == 2
    memories = SQLiteMemoryRepository(connection).list_memories_for_session(new_session_id)
    assert len(memories) == 1
    connection.close()


def test_reset_db_wipes_all_sessions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    database_path = _seed_database(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        ["reset-db", "--yes"],
        env={"DATABASE_PATH": str(database_path)},
    )

    assert result.exit_code == 0
    connection = connect_sqlite(database_path)
    assert SQLiteSessionRepository(connection).list_recent_sessions(10) == []
    assert SQLiteMemoryRepository(connection).list_memories_for_session("session-1") == []
    connection.close()


def test_inspect_memories_lists_episode_summaries(tmp_path: Path) -> None:
    database_path = _seed_database(tmp_path)

    result = runner.invoke(
        app,
        ["inspect-memories", "--session-id", "session-1"],
        env={"DATABASE_PATH": str(database_path)},
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["session_id"] == "session-1"
    assert len(payload["memories"]) == 1
    assert payload["memories"][0]["summary"] == (
        "The player promised to return before dawn."
    )
    assert payload["memories"][0]["visibility"] == "player"


def test_backup_writes_timestamped_copy(tmp_path: Path) -> None:
    database_path = _seed_database(tmp_path)

    result = runner.invoke(
        app,
        ["backup", "--output-dir", str(tmp_path / "backups")],
        env={"DATABASE_PATH": str(database_path)},
    )

    assert result.exit_code == 0, result.output
    copies = list((tmp_path / "backups").glob("rolerag-*.db"))
    assert len(copies) == 1

    check = sqlite3.connect(copies[0])
    assert check.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
    check.close()
