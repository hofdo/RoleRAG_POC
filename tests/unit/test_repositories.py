from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.domain import (
    CriticStatus,
    MemoryCandidate,
    SessionState,
    TurnDiagnostics,
    TurnOutcome,
    Visibility,
)
from app.llm.router import ModelProviderName, ModelRoute
from app.persistence.repositories import (
    SQLiteMemoryRepository,
    SQLiteSessionRepository,
    SQLiteTurnRepository,
    restore_persona_after_turn_delete,
)
from app.persistence.sqlite import connect_sqlite, initialize_database, serialize_datetime


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


def test_turn_repository_writes_zero_for_the_legacy_confirmation_column(tmp_path: Path) -> None:
    # ModelRoute no longer has a confirmation flag; the column is kept (no destructive
    # migration) but must always be written as 0, and never read back onto the model.
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
        reason="session provider: cloud",
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
    assert turns[0].route.reason == "session provider: cloud"
    row = connection.execute(
        "SELECT route_requires_user_confirmation FROM turns WHERE session_id = ?",
        ("session-1",),
    ).fetchone()
    assert row["route_requires_user_confirmation"] == 0


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


def test_update_active_scene_and_persona(tmp_path: Path) -> None:
    connection = connect_sqlite(tmp_path / "sessions.db")
    initialize_database(connection)
    session_repository = SQLiteSessionRepository(connection)
    session = session_repository.create_session(
        SessionState(
            id="session-1",
            world_id="demo_world",
            active_scene_id="rose-gallery",
            active_persona_id="archivist",
            player_name="Avery",
        )
    )

    session_repository.update_active_scene(session.id, "east_wing")
    session_repository.update_active_persona(session.id, "warden")
    reloaded = session_repository.get_session(session.id)

    assert reloaded is not None
    assert reloaded.active_scene_id == "east_wing"
    assert reloaded.active_persona_id == "warden"
    assert reloaded.updated_at is not None
    assert session.updated_at is not None
    assert reloaded.updated_at >= session.updated_at


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


def test_delete_last_turn_removes_and_returns_it(tmp_path: Path) -> None:
    _, turns, _ = _seed_session_with_data(tmp_path)

    deleted = turns.delete_last_turn("session-1")

    assert deleted is not None and deleted.turn_index == 3
    remaining = turns.list_all_turns("session-1")
    assert [t.turn_index for t in remaining] == [1, 2]
    assert turns.delete_last_turn("missing-session") is None


def test_delete_memories_since_removes_only_at_or_after_cutoff(tmp_path: Path) -> None:
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

    first = memory_repository.append_memories(
        session_id="session-1",
        memories=[
            MemoryCandidate(
                summary="First memory.",
                visibility=Visibility.PLAYER,
                importance=2,
                tags=["first"],
                scene_id="rose-gallery",
                actor_id="archivist",
            )
        ],
    )[0]
    second = memory_repository.append_memories(
        session_id="session-1",
        memories=[
            MemoryCandidate(
                summary="Second memory.",
                visibility=Visibility.PLAYER,
                importance=2,
                tags=["second"],
                scene_id="rose-gallery",
                actor_id="archivist",
            )
        ],
    )[0]
    # Cutoff sits strictly between the two timestamps (both were captured after
    # append_memories persisted, so `second.created_at` is a safe upper bound
    # that is also >= itself — the comparison below is exclusive of `first`).
    assert first.created_at is not None
    assert second.created_at is not None
    assert second.created_at > first.created_at
    cutoff = first.created_at + (second.created_at - first.created_at) / 2

    deleted_ids = memory_repository.delete_memories_since("session-1", cutoff)

    assert deleted_ids == [second.id]
    assert [m.id for m in memory_repository.list_memories_for_session("session-1")] == [first.id]


def _insert_memory_row(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
    session_id: str,
    created_at: datetime,
) -> None:
    # append_memories always stamps created_at via utc_now(), so it can't produce
    # the microsecond == 0 vs fractional-second adversarial pairing below. Insert
    # directly to control created_at exactly, matching the memory_episodes schema.
    connection.execute(
        """
        INSERT INTO memory_episodes (
            id, session_id, scene_id, actor_id, summary, importance,
            visibility, tags_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            memory_id,
            session_id,
            "rose-gallery",
            "archivist",
            f"Memory {memory_id}.",
            2,
            Visibility.PLAYER.value,
            json.dumps([]),
            serialize_datetime(created_at),
        ),
    )
    connection.commit()


def test_delete_memories_since_handles_zero_microsecond_text_sort_defect(
    tmp_path: Path,
) -> None:
    # ponytail: regression for the TEXT-lexicographic-compare defect — serialize_datetime
    # omits ".ffffff" when microsecond == 0. Cutoff is a zero-microsecond timestamp
    # ("...12:00:00Z"); the adversarial row is 0.5s *later* chronologically but carries a
    # fractional second ("...12:00:00.500000Z"). As plain strings,
    # "...12:00:00.500000Z" >= "...12:00:00Z" is FALSE (the byte "." sorts below "Z" is
    # irrelevant — the string is simply longer with a lower-valued extra "0" segment
    # before matching further, and Python string compare stops at the first differing
    # byte: "." (0x2E) < "Z" (0x5A)), so the old SQL `created_at >= cutoff` would wrongly
    # SKIP this row even though it is chronologically at/after the cutoff and must be
    # deleted.
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

    before_cutoff = datetime(2026, 1, 1, 11, 59, 59, 0, tzinfo=UTC)
    cutoff_zero_microsecond = datetime(2026, 1, 1, 12, 0, 0, 0, tzinfo=UTC)
    after_cutoff_fractional = datetime(2026, 1, 1, 12, 0, 0, 500_000, tzinfo=UTC)

    assert before_cutoff < cutoff_zero_microsecond < after_cutoff_fractional
    # The TEXT-sort defect this guards against: the chronologically later row sorts
    # as lexicographically SMALLER than the cutoff string, so a naive
    # `created_at >= cutoff` SQL compare would exclude it.
    assert (
        serialize_datetime(after_cutoff_fractional)
        < serialize_datetime(cutoff_zero_microsecond)
    )

    _insert_memory_row(
        connection,
        memory_id="mem-before-cutoff",
        session_id="session-1",
        created_at=before_cutoff,
    )
    _insert_memory_row(
        connection,
        memory_id="mem-at-cutoff",
        session_id="session-1",
        created_at=cutoff_zero_microsecond,
    )
    _insert_memory_row(
        connection,
        memory_id="mem-after-cutoff-fractional",
        session_id="session-1",
        created_at=after_cutoff_fractional,
    )

    deleted_ids = memory_repository.delete_memories_since(
        "session-1", cutoff_zero_microsecond
    )

    assert set(deleted_ids) == {"mem-at-cutoff", "mem-after-cutoff-fractional"}
    remaining_ids = {
        m.id for m in memory_repository.list_memories_for_session("session-1")
    }
    assert remaining_ids == {"mem-before-cutoff"}


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
    # token_usage (#69) is optional-and-additive: an explicit round trip carries it.
    assert reloaded.diagnostics is not None
    assert reloaded.diagnostics.token_usage is None


def test_turn_repository_loads_diagnostics_json_predating_token_usage_field(
    tmp_path: Path,
) -> None:
    # docs #69: token_usage was added to TurnDiagnostics after diagnostics_json rows
    # already existed in the wild. A row persisted before that change has no
    # "token_usage" key in its JSON at all -- simulate that directly (rather than via
    # model_dump_json, which would always include the new field) and confirm it still
    # deserializes, defaulting the field to None instead of raising.
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
    pre_69_diagnostics_json = json.dumps(
        {
            "retrieval": None,
            "stage_timings": {"gen": 0.5},
            "critic_status": "accepted",
            "finish_reason": "stop",
            "warnings": [],
            "memory_written": True,
            # no "token_usage" key at all -- the pre-#69 shape.
        }
    )
    connection.execute(
        "UPDATE turns SET diagnostics_json = ? WHERE id = ?",
        (pre_69_diagnostics_json, turn.id),
    )
    connection.commit()

    reloaded = turn_repository.list_all_turns("session-1")[0]

    assert reloaded.diagnostics is not None
    assert reloaded.diagnostics.token_usage is None
    assert reloaded.diagnostics.memory_written is True


def test_append_memory_outcome_merges_into_diagnostics(tmp_path: Path) -> None:
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
    turn_repository.update_turn_diagnostics(
        turn.id,
        TurnDiagnostics(
            retrieval=None,
            stage_timings={"gen": 0.5},
            critic_status=CriticStatus.ACCEPTED,
            finish_reason="stop",
            warnings=["memory curation deferred: runs after this response"],
            memory_written=False,
        ),
    )

    turn_repository.append_memory_outcome(
        turn.id,
        memory_written=True,
        warnings=["memory dedup dropped 1 duplicate candidate(s)"],
    )

    stored = turn_repository.list_all_turns("session-1")[-1]
    assert stored.diagnostics is not None
    assert stored.diagnostics.memory_written is True
    assert len(stored.diagnostics.warnings) == 2
    assert stored.diagnostics.warnings == [
        "memory curation deferred: runs after this response",
        "memory dedup dropped 1 duplicate candidate(s)",
    ]


def test_turn_outcome_round_trips_and_defaults_to_success(tmp_path: Path) -> None:
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

    turn_repository.append_turn(
        session_id="session-1",
        scene_id="rose-gallery",
        persona_id="archivist",
        user_message="First question",
        assistant_message="First answer",
        route=_build_route(),
    )
    turn_repository.append_turn(
        session_id="session-1",
        scene_id="rose-gallery",
        persona_id="archivist",
        user_message="Doomed question",
        assistant_message="The system could not produce a response.",
        route=_build_route(),
        outcome=TurnOutcome.CONTROLLED_FAILURE,
    )

    stored = turn_repository.list_all_turns("session-1")
    assert [turn.outcome for turn in stored] == [
        TurnOutcome.SUCCESS,
        TurnOutcome.CONTROLLED_FAILURE,
    ]
    deleted = turn_repository.delete_last_turn("session-1")
    assert deleted is not None
    assert deleted.outcome == TurnOutcome.CONTROLLED_FAILURE


def test_list_recent_turns_excludes_controlled_failures(tmp_path: Path) -> None:
    # Prompt context and the recent-session view must never feed the canned
    # failure line back to the model.
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

    for index, outcome in enumerate(
        [TurnOutcome.SUCCESS, TurnOutcome.CONTROLLED_FAILURE, TurnOutcome.SUCCESS]
    ):
        turn_repository.append_turn(
            session_id="session-1",
            scene_id="rose-gallery",
            persona_id="archivist",
            user_message=f"Question {index + 1}",
            assistant_message=f"Answer {index + 1}",
            route=_build_route(),
            outcome=outcome,
        )

    recent = turn_repository.list_recent_turns("session-1", limit=8)
    assert [turn.user_message for turn in recent] == ["Question 1", "Question 3"]
    # The unfiltered view still returns everything (Inspector / resume).
    assert len(turn_repository.list_all_turns("session-1")) == 3


def test_session_provider_round_trips_and_defaults_to_local(tmp_path: Path) -> None:
    connection = connect_sqlite(tmp_path / "sessions.db")
    initialize_database(connection)
    repository = SQLiteSessionRepository(connection)
    repository.create_session(
        SessionState(
            id="cloud-session",
            world_id="demo_world",
            active_scene_id="rose-gallery",
            active_persona_id="archivist",
            player_name="Avery",
            provider=ModelProviderName.CLOUD,
        )
    )
    repository.create_session(
        SessionState(
            id="local-session",
            world_id="demo_world",
            active_scene_id="rose-gallery",
            active_persona_id="archivist",
            player_name="Avery",
        )
    )
    cloud = repository.get_session("cloud-session")
    local = repository.get_session("local-session")
    assert cloud is not None and cloud.provider == ModelProviderName.CLOUD
    assert local is not None and local.provider == ModelProviderName.LOCAL
    assert {s.id: s.provider for s in repository.list_recent_sessions(10)} == {
        "cloud-session": ModelProviderName.CLOUD,
        "local-session": ModelProviderName.LOCAL,
    }


def _new_session_repos(tmp_path: Path) -> tuple[SQLiteSessionRepository, SQLiteTurnRepository]:
    connection = connect_sqlite(tmp_path / "sessions.db")
    initialize_database(connection)
    sessions = SQLiteSessionRepository(connection)
    turns = SQLiteTurnRepository(connection)
    sessions.create_session(
        SessionState(
            id="session-1",
            world_id="demo_world",
            active_scene_id="rose-gallery",
            active_persona_id="archivist",
            player_name="Avery",
        )
    )
    return sessions, turns


def test_restore_persona_after_turn_delete_reverts_a_committed_switch(tmp_path: Path) -> None:
    # Mirrors what TurnOrchestrator.run_turn does on a successful persona-switching
    # turn: persist the turn under the new persona, then commit active_persona_id.
    # A preceding non-switching turn establishes "archivist" as the value a reroll
    # of the switching turn must restore.
    sessions, turns = _new_session_repos(tmp_path)
    turns.append_turn(
        session_id="session-1",
        scene_id="rose-gallery",
        persona_id="archivist",
        user_message="Hello there.",
        assistant_message="Greetings.",
        route=_build_route(),
    )
    switched = turns.append_turn(
        session_id="session-1",
        scene_id="rose-gallery",
        persona_id="warden",
        user_message="Who are you?",
        assistant_message="I am the warden.",
        route=_build_route(),
    )
    sessions.update_active_persona("session-1", "warden")
    turns.delete_last_turn("session-1")

    restore_persona_after_turn_delete(
        session_repository=sessions,
        turn_repository=turns,
        session_id="session-1",
        deleted_turn=switched,
    )

    reloaded = sessions.get_session("session-1")
    assert reloaded is not None
    assert reloaded.active_persona_id == "archivist"


def test_restore_persona_after_turn_delete_restores_the_nearest_surviving_success_turn(
    tmp_path: Path,
) -> None:
    # Turn 1 switches archivist -> warden (SUCCESS). Turn 2 switches warden -> regent
    # but ends CONTROLLED_FAILURE, so it never commits. Turn 3 switches warden ->
    # regent and succeeds. Deleting turn 3 must restore "warden" (turn 1's persona),
    # not "archivist" (session creation) and not skip past the failed turn 2.
    sessions, turns = _new_session_repos(tmp_path)
    turns.append_turn(
        session_id="session-1",
        scene_id="rose-gallery",
        persona_id="warden",
        user_message="Turn 1",
        assistant_message="Reply 1",
        route=_build_route(),
    )
    sessions.update_active_persona("session-1", "warden")
    turns.append_turn(
        session_id="session-1",
        scene_id="rose-gallery",
        persona_id="regent",
        user_message="Turn 2",
        assistant_message="The system could not produce a response.",
        route=_build_route(),
        outcome=TurnOutcome.CONTROLLED_FAILURE,
    )
    last = turns.append_turn(
        session_id="session-1",
        scene_id="rose-gallery",
        persona_id="regent",
        user_message="Turn 3",
        assistant_message="Reply 3",
        route=_build_route(),
    )
    sessions.update_active_persona("session-1", "regent")
    turns.delete_last_turn("session-1")

    restore_persona_after_turn_delete(
        session_repository=sessions,
        turn_repository=turns,
        session_id="session-1",
        deleted_turn=last,
    )

    reloaded = sessions.get_session("session-1")
    assert reloaded is not None
    assert reloaded.active_persona_id == "warden"


def test_restore_persona_after_turn_delete_is_a_noop_for_controlled_failure(
    tmp_path: Path,
) -> None:
    # A CONTROLLED_FAILURE turn never commits a persona switch (see
    # TurnOrchestrator._persist_controlled_failure), so deleting one must never
    # touch active_persona_id even if the failed turn's own persona_id differs.
    sessions, turns = _new_session_repos(tmp_path)
    failed = turns.append_turn(
        session_id="session-1",
        scene_id="rose-gallery",
        persona_id="warden",
        user_message="Turn 1",
        assistant_message="The system could not produce a response.",
        route=_build_route(),
        outcome=TurnOutcome.CONTROLLED_FAILURE,
    )
    turns.delete_last_turn("session-1")

    restore_persona_after_turn_delete(
        session_repository=sessions,
        turn_repository=turns,
        session_id="session-1",
        deleted_turn=failed,
    )

    reloaded = sessions.get_session("session-1")
    assert reloaded is not None
    assert reloaded.active_persona_id == "archivist"


def test_restore_persona_after_turn_delete_leaves_unrecoverable_creation_persona_alone(
    tmp_path: Path,
) -> None:
    # Deleting the very first turn of a session that switched persona on turn 1
    # leaves no remaining SUCCESS turn to recover a prior value from, and the
    # sessions row only tracks the CURRENT active_persona_id -- the session's
    # creation-time persona is not stored anywhere once overwritten. Documented
    # limitation (docs/BACKLOG.md #66): active_persona_id is left as-is rather
    # than guessed at.
    sessions, turns = _new_session_repos(tmp_path)
    only_turn = turns.append_turn(
        session_id="session-1",
        scene_id="rose-gallery",
        persona_id="warden",
        user_message="Turn 1",
        assistant_message="Reply 1",
        route=_build_route(),
    )
    sessions.update_active_persona("session-1", "warden")
    turns.delete_last_turn("session-1")

    restore_persona_after_turn_delete(
        session_repository=sessions,
        turn_repository=turns,
        session_id="session-1",
        deleted_turn=only_turn,
    )

    reloaded = sessions.get_session("session-1")
    assert reloaded is not None
    assert reloaded.active_persona_id == "warden"


def test_restore_persona_after_turn_delete_does_not_clobber_an_already_changed_value(
    tmp_path: Path,
) -> None:
    # If active_persona_id no longer matches what the deleted turn committed (e.g.
    # something else already changed it), the helper must not overwrite it.
    sessions, turns = _new_session_repos(tmp_path)
    switched = turns.append_turn(
        session_id="session-1",
        scene_id="rose-gallery",
        persona_id="warden",
        user_message="Turn 1",
        assistant_message="Reply 1",
        route=_build_route(),
    )
    sessions.update_active_persona("session-1", "warden")
    turns.delete_last_turn("session-1")
    sessions.update_active_persona("session-1", "regent")

    restore_persona_after_turn_delete(
        session_repository=sessions,
        turn_repository=turns,
        session_id="session-1",
        deleted_turn=switched,
    )

    reloaded = sessions.get_session("session-1")
    assert reloaded is not None
    assert reloaded.active_persona_id == "regent"
