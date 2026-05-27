from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Protocol

from app.domain import SessionState, StoredTurn
from app.llm.router import ModelProviderName, ModelRoute
from app.persistence.sqlite import parse_datetime, serialize_datetime, utc_now


class SessionNotFoundError(LookupError):
    def __init__(self, session_id: str) -> None:
        super().__init__(f"Unknown session id: {session_id}")
        self.session_id = session_id


class SessionRepository(Protocol):
    def create_session(self, session: SessionState) -> SessionState: ...

    def get_session(self, session_id: str) -> SessionState | None: ...

    def update_session_activity(self, session_id: str, *, updated_at: datetime) -> None: ...


class TurnRepository(Protocol):
    def append_turn(
        self,
        *,
        session_id: str,
        scene_id: str,
        persona_id: str,
        user_message: str,
        assistant_message: str,
        route: ModelRoute,
    ) -> StoredTurn: ...

    def list_recent_turns(self, session_id: str, limit: int) -> list[StoredTurn]: ...

    def count_turns(self, session_id: str) -> int: ...


class SQLiteSessionRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def create_session(self, session: SessionState) -> SessionState:
        now = utc_now()
        created_at = session.created_at or now
        updated_at = session.updated_at or now
        created = session.model_copy(
            update={
                "created_at": created_at,
                "updated_at": updated_at,
            }
        )
        self.connection.execute(
            """
            INSERT INTO sessions (
                id,
                world_id,
                active_scene_id,
                active_persona_id,
                player_name,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created.id,
                created.world_id,
                created.active_scene_id,
                created.active_persona_id,
                created.player_name,
                serialize_datetime(created_at),
                serialize_datetime(updated_at),
            ),
        )
        self.connection.commit()
        return created

    def get_session(self, session_id: str) -> SessionState | None:
        row = self.connection.execute(
            """
            SELECT
                id,
                world_id,
                active_scene_id,
                active_persona_id,
                player_name,
                created_at,
                updated_at
            FROM sessions
            WHERE id = ?
            """,
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return SessionState(
            id=row["id"],
            world_id=row["world_id"],
            active_scene_id=row["active_scene_id"],
            active_persona_id=row["active_persona_id"],
            player_name=row["player_name"],
            created_at=parse_datetime(row["created_at"]),
            updated_at=parse_datetime(row["updated_at"]),
        )

    def update_session_activity(self, session_id: str, *, updated_at: datetime) -> None:
        cursor = self.connection.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (serialize_datetime(updated_at), session_id),
        )
        self.connection.commit()
        if cursor.rowcount == 0:
            raise SessionNotFoundError(session_id)


class SQLiteTurnRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def append_turn(
        self,
        *,
        session_id: str,
        scene_id: str,
        persona_id: str,
        user_message: str,
        assistant_message: str,
        route: ModelRoute,
    ) -> StoredTurn:
        turn_index = self.count_turns(session_id) + 1
        created_at = utc_now()
        cursor = self.connection.execute(
            """
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
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                turn_index,
                scene_id,
                persona_id,
                user_message,
                assistant_message,
                route.provider.value,
                route.model,
                route.reason,
                route.max_tokens,
                route.temperature,
                int(route.requires_user_confirmation),
                serialize_datetime(created_at),
            ),
        )
        self.connection.commit()
        row_id = cursor.lastrowid
        if row_id is None:
            raise RuntimeError("SQLite did not return a row id for the persisted turn")
        return StoredTurn(
            id=int(row_id),
            session_id=session_id,
            turn_index=turn_index,
            scene_id=scene_id,
            persona_id=persona_id,
            user_message=user_message,
            assistant_message=assistant_message,
            route=route,
            created_at=created_at,
        )

    def list_recent_turns(self, session_id: str, limit: int) -> list[StoredTurn]:
        if limit <= 0:
            return []
        rows = self.connection.execute(
            """
            SELECT
                id,
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
            FROM turns
            WHERE session_id = ?
            ORDER BY turn_index DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
        return [self._row_to_turn(row) for row in reversed(rows)]

    def count_turns(self, session_id: str) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) AS count FROM turns WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row["count"])

    def _row_to_turn(self, row: sqlite3.Row) -> StoredTurn:
        return StoredTurn(
            id=row["id"],
            session_id=row["session_id"],
            turn_index=row["turn_index"],
            scene_id=row["scene_id"],
            persona_id=row["persona_id"],
            user_message=row["user_message"],
            assistant_message=row["assistant_message"],
            route=ModelRoute(
                provider=ModelProviderName(row["route_provider"]),
                model=row["route_model"],
                max_tokens=row["route_max_tokens"],
                temperature=row["route_temperature"],
                reason=row["route_reason"],
                requires_user_confirmation=bool(row["route_requires_user_confirmation"]),
            ),
            created_at=parse_datetime(row["created_at"]),
        )
