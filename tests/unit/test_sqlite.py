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


def test_connect_sqlite_enables_row_access_by_name(tmp_path: Path) -> None:
    connection = connect_sqlite(tmp_path / "sessions.db")
    initialize_database(connection)

    connection.execute("SELECT 1 AS value")
    row = connection.execute("SELECT 1 AS value").fetchone()

    assert isinstance(row, sqlite3.Row)
    assert row["value"] == 1
