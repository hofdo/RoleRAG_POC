from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Protocol
from uuid import uuid4

from app.domain import (
    CanonFact,
    MemoryCandidate,
    MemoryEpisode,
    SessionState,
    StoredTurn,
    TurnDiagnostics,
    TurnOutcome,
    Visibility,
)
from app.llm.router import ModelProviderName, ModelRoute
from app.persistence.sqlite import parse_datetime, serialize_datetime, utc_now


class SessionNotFoundError(LookupError):
    def __init__(self, session_id: str) -> None:
        super().__init__(f"Unknown session id: {session_id}")
        self.session_id = session_id


class SessionRepository(Protocol):
    def create_session(self, session: SessionState) -> SessionState: ...

    def get_session(self, session_id: str) -> SessionState | None: ...

    def list_recent_sessions(self, limit: int) -> list[SessionState]: ...

    def update_session_activity(self, session_id: str, *, updated_at: datetime) -> None: ...

    def update_active_scene(self, session_id: str, scene_id: str) -> None: ...

    def update_active_persona(self, session_id: str, persona_id: str) -> None: ...


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
        outcome: TurnOutcome = TurnOutcome.SUCCESS,
    ) -> StoredTurn: ...

    def update_turn_diagnostics(self, turn_id: int, diagnostics: TurnDiagnostics) -> None: ...

    def append_memory_outcome(
        self, turn_id: int, *, memory_written: bool, warnings: list[str]
    ) -> None: ...

    def list_recent_turns(self, session_id: str, limit: int) -> list[StoredTurn]: ...

    def list_all_turns(self, session_id: str) -> list[StoredTurn]: ...

    def count_turns(self, session_id: str) -> int: ...

    def delete_last_turn(self, session_id: str) -> StoredTurn | None: ...


def restore_persona_after_turn_delete(
    *,
    session_repository: SessionRepository,
    turn_repository: TurnRepository,
    session_id: str,
    deleted_turn: StoredTurn,
) -> None:
    """Undo the durable ``active_persona_id`` commit a deleted turn made, if any.

    ``TurnOrchestrator.run_turn`` commits a per-turn persona override to the session
    row only AFTER the turn persists with outcome SUCCESS (see
    ``app/orchestration/turn_orchestrator.py``); a CONTROLLED_FAILURE turn never
    commits one, by design (a failed turn the player never saw succeed must not
    move session state). Consequently, right after any SUCCESS turn completes,
    ``sessions.active_persona_id`` always equals that turn's own ``persona_id`` --
    whether or not that turn requested a switch, because a non-switching turn's
    persona_id already equalled the prior active_persona_id. Right after a
    CONTROLLED_FAILURE turn, active_persona_id is untouched (left exactly as it
    was before that turn ran).

    So deleting the most recent turn (reroll) must restore active_persona_id to
    whatever it held immediately before the deleted turn ran:

    - If the deleted turn was not SUCCESS, it never committed a persona change;
      no-op.
    - If it was SUCCESS, the value to restore is the persona_id of the nearest
      *remaining* SUCCESS turn (CONTROLLED_FAILURE turns never move the pointer,
      so they are skipped when walking backward).
    - If no SUCCESS turn remains, the deleted turn was the first to ever move the
      pointer since session creation. The sessions row only tracks the CURRENT
      active_persona_id, not the creation-time value, so once that first switch
      committed the original persona is no longer recoverable from stored state.
      Documented limitation (see docs/BACKLOG.md #66): active_persona_id is left
      as-is in this edge case rather than guessed at.
    """
    if deleted_turn.outcome != TurnOutcome.SUCCESS:
        return
    session = session_repository.get_session(session_id)
    if session is None or session.active_persona_id != deleted_turn.persona_id:
        # Session gone, or active_persona_id no longer matches what the deleted
        # turn committed (e.g. it was already restored, or something else moved
        # it since) -- don't clobber an unrelated value.
        return
    for turn in reversed(turn_repository.list_all_turns(session_id)):
        if turn.outcome == TurnOutcome.SUCCESS:
            if turn.persona_id != session.active_persona_id:
                session_repository.update_active_persona(session_id, turn.persona_id)
            return
    # No SUCCESS turn remains: nothing recoverable to restore to (see docstring).


class MemoryRepository(Protocol):
    def append_memories(
        self,
        *,
        session_id: str,
        memories: list[MemoryCandidate],
    ) -> list[MemoryEpisode]: ...

    def list_memories_for_session(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[MemoryEpisode]: ...

    def add_tag_to_memories(self, memory_ids: list[str], tag: str) -> None: ...

    def delete_memories_since(self, session_id: str, created_at: datetime) -> list[str]: ...


class CanonRepository(Protocol):
    def add_canon_fact(self, *, session_id: str, text: str) -> CanonFact: ...

    def list_canon_facts(self, session_id: str) -> list[CanonFact]: ...

    def delete_canon_fact(self, *, session_id: str, fact_id: str) -> bool: ...


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
                content_root,
                provider,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created.id,
                created.world_id,
                created.active_scene_id,
                created.active_persona_id,
                created.player_name,
                created.content_root,
                created.provider.value,
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
                content_root,
                provider,
                created_at,
                updated_at
            FROM sessions
            WHERE id = ?
            """,
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_session(row)

    def list_recent_sessions(self, limit: int) -> list[SessionState]:
        if limit <= 0:
            return []
        rows = self.connection.execute(
            """
            SELECT
                id,
                world_id,
                active_scene_id,
                active_persona_id,
                player_name,
                content_root,
                provider,
                created_at,
                updated_at
            FROM sessions
            ORDER BY updated_at DESC, created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [self._row_to_session(row) for row in rows]

    @staticmethod
    def _row_to_session(row: sqlite3.Row) -> SessionState:
        return SessionState(
            id=row["id"],
            world_id=row["world_id"],
            active_scene_id=row["active_scene_id"],
            active_persona_id=row["active_persona_id"],
            player_name=row["player_name"],
            content_root=row["content_root"],
            # provider column is absent on legacy DBs written before session-bound routing.
            provider=(
                ModelProviderName(row["provider"])
                if "provider" in row.keys()
                else ModelProviderName.LOCAL
            ),
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

    def delete_session(self, session_id: str) -> bool:
        cursor = self.connection.execute(
            "DELETE FROM sessions WHERE id = ?",
            (session_id,),
        )
        self.connection.commit()
        return cursor.rowcount > 0

    def update_active_scene(self, session_id: str, scene_id: str) -> None:
        self.connection.execute(
            "UPDATE sessions SET active_scene_id = ?, updated_at = ? WHERE id = ?",
            (scene_id, serialize_datetime(utc_now()), session_id),
        )
        self.connection.commit()

    def update_active_persona(self, session_id: str, persona_id: str) -> None:
        self.connection.execute(
            "UPDATE sessions SET active_persona_id = ?, updated_at = ? WHERE id = ?",
            (persona_id, serialize_datetime(utc_now()), session_id),
        )
        self.connection.commit()


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
        outcome: TurnOutcome = TurnOutcome.SUCCESS,
    ) -> StoredTurn:
        created_at = utc_now()
        # turn_index is assigned atomically inside the INSERT itself (a correlated
        # subselect over the same table), not via a separate count-then-write step.
        # Two concurrent writers on the same session_id both execute this single
        # statement; SQLite serializes them on the write lock (WAL + busy_timeout,
        # see app/persistence/sqlite.py), so the second writer's subselect always
        # observes the first writer's committed row and gets the next index -- no
        # UNIQUE(session_id, turn_index) collision is possible. RETURNING reports
        # back the index actually assigned, which may differ from a value computed
        # ahead of time under concurrency.
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
                created_at,
                outcome
            ) VALUES (
                ?,
                (SELECT COALESCE(MAX(turn_index), 0) + 1 FROM turns WHERE session_id = ?),
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            RETURNING turn_index, id
            """,
            (
                session_id,
                session_id,
                scene_id,
                persona_id,
                user_message,
                assistant_message,
                route.provider.value,
                route.model,
                route.reason,
                route.max_tokens,
                route.temperature,
                # ModelRoute no longer has a confirmation flag; the column stays
                # (no destructive migration) but is always written as 0 now.
                0,
                serialize_datetime(created_at),
                outcome.value,
            ),
        )
        returned = cursor.fetchone()
        self.connection.commit()
        if returned is None:
            raise RuntimeError("SQLite did not return the persisted turn's row id/index")
        turn_index = int(returned["turn_index"])
        row_id = returned["id"]
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
            outcome=outcome,
        )

    def update_turn_diagnostics(self, turn_id: int, diagnostics: TurnDiagnostics) -> None:
        self.connection.execute(
            "UPDATE turns SET diagnostics_json = ? WHERE id = ?",
            (diagnostics.model_dump_json(), turn_id),
        )
        self.connection.commit()

    def append_memory_outcome(
        self, turn_id: int, *, memory_written: bool, warnings: list[str]
    ) -> None:
        row = self.connection.execute(
            "SELECT diagnostics_json FROM turns WHERE id = ?", (turn_id,)
        ).fetchone()
        if row is None or row["diagnostics_json"] is None:
            return
        payload = json.loads(row["diagnostics_json"])
        payload["memory_written"] = memory_written
        payload["warnings"] = [*payload.get("warnings", []), *warnings]
        self.connection.execute(
            "UPDATE turns SET diagnostics_json = ? WHERE id = ?",
            (json.dumps(payload), turn_id),
        )
        self.connection.commit()

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
                diagnostics_json,
                created_at,
                outcome
            FROM turns
            WHERE session_id = ? AND outcome = 'success'
            ORDER BY turn_index DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
        # Controlled failures are excluded: recent dialogue feeds the actor prompt
        # and the recent-session view, and neither should echo the canned failure
        # line back. list_all_turns keeps the unfiltered history.
        return [self._row_to_turn(row) for row in reversed(rows)]

    def list_all_turns(self, session_id: str) -> list[StoredTurn]:
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
                diagnostics_json,
                created_at,
                outcome
            FROM turns
            WHERE session_id = ?
            ORDER BY turn_index ASC
            """,
            (session_id,),
        ).fetchall()
        return [self._row_to_turn(row) for row in rows]

    def count_turns(self, session_id: str) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) AS count FROM turns WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row["count"])

    def delete_last_turn(self, session_id: str) -> StoredTurn | None:
        row = self.connection.execute(
            "SELECT * FROM turns WHERE session_id = ? ORDER BY turn_index DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        self.connection.execute("DELETE FROM turns WHERE id = ?", (row["id"],))
        self.connection.commit()
        return self._row_to_turn(row)

    def _row_to_turn(self, row: sqlite3.Row) -> StoredTurn:
        return StoredTurn(
            id=row["id"],
            session_id=row["session_id"],
            turn_index=row["turn_index"],
            scene_id=row["scene_id"],
            persona_id=row["persona_id"],
            user_message=row["user_message"],
            assistant_message=row["assistant_message"],
            # route_requires_user_confirmation is read from the row for legacy DBs but
            # ModelRoute no longer has that field, so it is intentionally ignored here.
            route=ModelRoute(
                provider=ModelProviderName(row["route_provider"]),
                model=row["route_model"],
                max_tokens=row["route_max_tokens"],
                temperature=row["route_temperature"],
                reason=row["route_reason"],
            ),
            created_at=parse_datetime(row["created_at"]),
            diagnostics=self._parse_diagnostics(row),
            outcome=TurnOutcome(row["outcome"]) if "outcome" in row.keys() else TurnOutcome.SUCCESS,
        )

    @staticmethod
    def _parse_diagnostics(row: sqlite3.Row) -> TurnDiagnostics | None:
        # Very old rows predate the diagnostics_json column; sqlite3.Row has no
        # .get(), so guard on the key being present before reading it.
        if "diagnostics_json" not in row.keys():
            return None
        raw = row["diagnostics_json"]
        if raw is None:
            return None
        return TurnDiagnostics.model_validate_json(raw)


class SQLiteMemoryRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def append_memories(
        self,
        *,
        session_id: str,
        memories: list[MemoryCandidate],
    ) -> list[MemoryEpisode]:
        created_at = utc_now()
        episodes: list[MemoryEpisode] = []
        for memory in memories:
            episode = MemoryEpisode(
                id=str(uuid4()),
                session_id=session_id,
                scene_id=memory.scene_id or "",
                actor_id=memory.actor_id,
                summary=memory.summary,
                importance=memory.importance,
                visibility=memory.visibility,
                tags=memory.tags,
                created_at=created_at,
                source_turn_id=memory.source_turn_id,
            )
            self.connection.execute(
                """
                INSERT INTO memory_episodes (
                    id,
                    session_id,
                    scene_id,
                    actor_id,
                    summary,
                    importance,
                    visibility,
                    tags_json,
                    created_at,
                    source_turn_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode.id,
                    episode.session_id,
                    episode.scene_id,
                    episode.actor_id,
                    episode.summary,
                    episode.importance,
                    episode.visibility.value,
                    json.dumps(episode.tags),
                    serialize_datetime(created_at),
                    episode.source_turn_id,
                ),
            )
            episodes.append(episode)
        self.connection.commit()
        return episodes

    def list_memories_for_session(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[MemoryEpisode]:
        sql = """
            SELECT
                id,
                session_id,
                scene_id,
                actor_id,
                summary,
                importance,
                visibility,
                tags_json,
                created_at,
                source_turn_id
            FROM memory_episodes
            WHERE session_id = ?
            ORDER BY created_at DESC
        """
        params: tuple[object, ...]
        if limit is None:
            params = (session_id,)
        else:
            sql += " LIMIT ?"
            params = (session_id, limit)
        rows = self.connection.execute(sql, params).fetchall()
        return [
            MemoryEpisode(
                id=row["id"],
                session_id=row["session_id"],
                scene_id=row["scene_id"],
                actor_id=row["actor_id"],
                summary=row["summary"],
                importance=row["importance"],
                visibility=Visibility(row["visibility"]),
                tags=json.loads(row["tags_json"]),
                created_at=parse_datetime(row["created_at"]),
                # source_turn_id predates this column on rows written before #76
                # landed; sqlite3.Row has no .get(), so guard on the key being
                # present (mirrors _row_to_turn's outcome/_row_to_session's provider
                # handling for the same _ensure_column migration pattern).
                source_turn_id=(
                    row["source_turn_id"] if "source_turn_id" in row.keys() else None
                ),
            )
            for row in reversed(rows)
        ]

    def add_tag_to_memories(self, memory_ids: list[str], tag: str) -> None:
        if not memory_ids:
            return
        placeholders = ",".join("?" for _ in memory_ids)
        rows = self.connection.execute(
            f"SELECT id, tags_json FROM memory_episodes WHERE id IN ({placeholders})",
            tuple(memory_ids),
        ).fetchall()
        for row in rows:
            tags = json.loads(row["tags_json"])
            if tag in tags:
                continue
            tags.append(tag)
            self.connection.execute(
                "UPDATE memory_episodes SET tags_json = ? WHERE id = ?",
                (json.dumps(tags), row["id"]),
            )
        self.connection.commit()

    def delete_memories_since(self, session_id: str, created_at: datetime) -> list[str]:
        # ponytail: don't compare created_at as TEXT in SQL — serialize_datetime
        # omits .ffffff when microsecond == 0, so a chronologically *later*
        # timestamp in the same second that carries fractional seconds sorts
        # lexicographically *before* a zero-microsecond cutoff
        # ("...T12:00:00.500000Z" < "...T12:00:00Z" as strings, because '.' (0x2E)
        # < 'Z' (0x5A)) — a TEXT created_at >= cutoff compare would wrongly skip
        # that row. Parse and compare as datetimes instead.
        rows = self.connection.execute(
            "SELECT id, created_at FROM memory_episodes WHERE session_id = ?",
            (session_id,),
        ).fetchall()
        ids = [row["id"] for row in rows if parse_datetime(row["created_at"]) >= created_at]
        if ids:
            placeholders = ",".join("?" for _ in ids)
            self.connection.execute(
                f"DELETE FROM memory_episodes WHERE id IN ({placeholders})", ids
            )
            self.connection.commit()
        return ids


class SQLiteCanonRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def add_canon_fact(self, *, session_id: str, text: str) -> CanonFact:
        created_at = utc_now()
        fact = CanonFact(
            id=str(uuid4()),
            session_id=session_id,
            text=text,
            created_at=created_at,
        )
        self.connection.execute(
            "INSERT INTO canon_facts (id, session_id, text, created_at) VALUES (?, ?, ?, ?)",
            (fact.id, fact.session_id, fact.text, serialize_datetime(created_at)),
        )
        self.connection.commit()
        return fact

    def list_canon_facts(self, session_id: str) -> list[CanonFact]:
        rows = self.connection.execute(
            """
            SELECT id, session_id, text, created_at
            FROM canon_facts
            WHERE session_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (session_id,),
        ).fetchall()
        return [
            CanonFact(
                id=row["id"],
                session_id=row["session_id"],
                text=row["text"],
                created_at=parse_datetime(row["created_at"]),
            )
            for row in rows
        ]

    def delete_canon_fact(self, *, session_id: str, fact_id: str) -> bool:
        cursor = self.connection.execute(
            "DELETE FROM canon_facts WHERE id = ? AND session_id = ?",
            (fact_id, session_id),
        )
        self.connection.commit()
        return cursor.rowcount > 0
