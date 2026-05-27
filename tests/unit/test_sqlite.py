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


def test_connect_sqlite_enables_row_access_by_name(tmp_path: Path) -> None:
    connection = connect_sqlite(tmp_path / "sessions.db")
    initialize_database(connection)

    connection.execute("SELECT 1 AS value")
    row = connection.execute("SELECT 1 AS value").fetchone()

    assert isinstance(row, sqlite3.Row)
    assert row["value"] == 1
