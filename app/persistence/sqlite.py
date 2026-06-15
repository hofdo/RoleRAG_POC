from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path


def connect_sqlite(database_path: str | Path) -> sqlite3.Connection:
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            world_id TEXT NOT NULL,
            active_scene_id TEXT NOT NULL,
            active_persona_id TEXT NOT NULL,
            player_name TEXT NOT NULL,
            content_root TEXT NOT NULL DEFAULT 'data',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
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

        CREATE INDEX IF NOT EXISTS turns_session_idx
        ON turns(session_id, turn_index DESC);

        CREATE TABLE IF NOT EXISTS memory_episodes (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            scene_id TEXT NOT NULL,
            actor_id TEXT,
            summary TEXT NOT NULL,
            importance INTEGER NOT NULL,
            visibility TEXT NOT NULL,
            tags_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS memory_episodes_session_idx
        ON memory_episodes(session_id, created_at DESC);

        CREATE TABLE IF NOT EXISTS canon_facts (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            text TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS canon_facts_session_idx
        ON canon_facts(session_id, created_at);
        """
    )
    _ensure_column(
        connection,
        table="sessions",
        column="content_root",
        ddl="ALTER TABLE sessions ADD COLUMN content_root TEXT NOT NULL DEFAULT 'data'",
    )
    connection.commit()


def _ensure_column(
    connection: sqlite3.Connection,
    *,
    table: str,
    column: str,
    ddl: str,
) -> None:
    columns = {
        row["name"]
        for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if column not in columns:
        connection.execute(ddl)


def utc_now() -> datetime:
    return datetime.now(UTC)


def serialize_datetime(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
