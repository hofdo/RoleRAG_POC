from __future__ import annotations

import sqlite3
from pathlib import Path

from app.persistence.sqlite import connect_sqlite, initialize_database


def test_initialize_database_creates_sessions_turns_and_memory_tables(tmp_path: Path) -> None:
    connection = connect_sqlite(tmp_path / "sessions.db")

    initialize_database(connection)

    tables = {
        row["name"]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }

    assert "sessions" in tables
    assert "turns" in tables
    assert "memory_episodes" in tables
    session_columns = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(sessions)").fetchall()
    }
    assert "content_root" in session_columns


def test_initialize_database_adds_content_root_to_existing_sessions_table(
    tmp_path: Path,
) -> None:
    connection = connect_sqlite(tmp_path / "sessions.db")
    connection.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            world_id TEXT NOT NULL,
            active_scene_id TEXT NOT NULL,
            active_persona_id TEXT NOT NULL,
            player_name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        INSERT INTO sessions (
            id,
            world_id,
            active_scene_id,
            active_persona_id,
            player_name,
            created_at,
            updated_at
        ) VALUES (
            'legacy-session',
            'demo_world',
            'rose-gallery',
            'archivist',
            'Avery',
            '2026-05-27T10:30:00Z',
            '2026-05-27T10:30:00Z'
        );
        """
    )

    initialize_database(connection)

    row = connection.execute(
        "SELECT content_root FROM sessions WHERE id = 'legacy-session'"
    ).fetchone()
    assert row["content_root"] == "data"


def test_initialize_database_migration_adds_diagnostics_json_column(
    tmp_path: Path,
) -> None:
    from app.persistence.repositories import SQLiteTurnRepository

    connection = connect_sqlite(tmp_path / "sessions.db")
    connection.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            world_id TEXT NOT NULL,
            active_scene_id TEXT NOT NULL,
            active_persona_id TEXT NOT NULL,
            player_name TEXT NOT NULL,
            content_root TEXT NOT NULL DEFAULT 'data',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            turn_index INTEGER NOT NULL,
            scene_id TEXT NOT NULL,
            persona_id TEXT NOT NULL,
            user_message TEXT NOT NULL,
            assistant_message TEXT NOT NULL,
            route_provider TEXT NOT NULL,
            route_model TEXT NOT NULL,
            route_reason TEXT NOT NULL,
            route_max_tokens INTEGER NOT NULL,
            route_temperature REAL NOT NULL,
            route_requires_user_confirmation INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            UNIQUE(session_id, turn_index)
        );

        INSERT INTO turns (
            session_id,
            turn_index,
            scene_id,
            persona_id,
            user_message,
            assistant_message,
            route_provider,
            route_model,
            route_reason,
            route_max_tokens,
            route_temperature,
            route_requires_user_confirmation,
            created_at
        ) VALUES (
            'legacy-session',
            1,
            'rose-gallery',
            'archivist',
            'legacy question',
            'legacy answer',
            'local',
            'local-model',
            'default local route',
            700,
            0.75,
            0,
            '2026-05-27T10:30:00Z'
        );
        """
    )

    turn_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(turns)").fetchall()
    }
    assert "diagnostics_json" not in turn_columns

    initialize_database(connection)

    migrated_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(turns)").fetchall()
    }
    assert "diagnostics_json" in migrated_columns
    # The migrated column is nullable; the legacy row is left with a NULL value.
    legacy = connection.execute(
        "SELECT diagnostics_json FROM turns WHERE session_id = 'legacy-session'"
    ).fetchone()
    assert legacy["diagnostics_json"] is None

    # An old row loads through the repository with diagnostics=None.
    loaded = SQLiteTurnRepository(connection).list_all_turns("legacy-session")
    assert len(loaded) == 1
    assert loaded[0].diagnostics is None


def test_connect_sqlite_enables_row_access_by_name(tmp_path: Path) -> None:
    connection = connect_sqlite(tmp_path / "sessions.db")
    initialize_database(connection)

    connection.execute("SELECT 1 AS value")
    row = connection.execute("SELECT 1 AS value").fetchone()

    assert isinstance(row, sqlite3.Row)
    assert row["value"] == 1


def test_connect_sqlite_enables_wal_and_busy_timeout(tmp_path: Path) -> None:
    connection = connect_sqlite(tmp_path / "wal.db")
    try:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    finally:
        connection.close()
