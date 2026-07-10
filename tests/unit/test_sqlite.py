from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from app.llm.router import ModelProviderName, ModelRoute
from app.persistence.repositories import SQLiteTurnRepository
from app.persistence.sqlite import connect_sqlite, initialize_database


def _insert_session(connection: sqlite3.Connection, session_id: str) -> None:
    connection.execute(
        """
        INSERT INTO sessions
            (id, world_id, active_scene_id, active_persona_id, player_name, created_at, updated_at)
        VALUES (?, 'w', 'sc', 'p', 'player', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')
        """,
        (session_id,),
    )


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


def test_wal_reader_is_not_blocked_by_an_open_writer(tmp_path: Path) -> None:
    # WAL lets a second connection keep reading committed data while another holds an
    # uncommitted write — the documented "CLI and API share this file" property (#64).
    path = tmp_path / "shared.db"
    writer = connect_sqlite(path)
    initialize_database(writer)
    _insert_session(writer, "s1")
    writer.commit()

    reader = connect_sqlite(path)
    try:
        _insert_session(writer, "s2")  # holds the write lock, uncommitted
        seen_during_write = {r["id"] for r in reader.execute("SELECT id FROM sessions")}
        assert seen_during_write == {"s1"}  # reader sees the last committed snapshot, not s2

        writer.commit()
        seen_after_commit = {r["id"] for r in reader.execute("SELECT id FROM sessions")}
        assert seen_after_commit == {"s1", "s2"}
    finally:
        reader.close()
        writer.close()


def test_second_writer_waits_on_busy_timeout_instead_of_locking(tmp_path: Path) -> None:
    # A second writer must WAIT for the first to commit (busy_timeout), not fail with
    # "database is locked" — proving the pragma value actually changes behavior.
    path = tmp_path / "shared.db"
    conn_a = connect_sqlite(path)
    initialize_database(conn_a)
    conn_b = connect_sqlite(path)

    holds_write_lock = threading.Event()

    def hold_then_commit() -> None:
        _insert_session(conn_b, "b")  # acquires the WAL write lock
        holds_write_lock.set()
        time.sleep(0.2)  # keep the lock held so conn_a genuinely contends
        conn_b.commit()

    holder = threading.Thread(target=hold_then_commit)
    holder.start()
    try:
        assert holds_write_lock.wait(timeout=5.0)
        # Without busy_timeout this raises sqlite3.OperationalError immediately; with it, the
        # write blocks until the holder commits (~0.2s) and then succeeds.
        _insert_session(conn_a, "a")
        conn_a.commit()
    finally:
        holder.join()

    rows = {r["id"] for r in conn_a.execute("SELECT id FROM sessions")}
    assert rows == {"a", "b"}
    conn_a.close()
    conn_b.close()


# (#70) The tests above prove WAL behavior with two connections in ONE process -- they
# never exercised a second OS process despite living next to a commit titled "cross-process
# concurrency". This worker script backs a genuine second-process test below: it is invoked
# via `python -c <script>` as an independent interpreter, connecting to the same on-disk
# database file as the pytest process and racing it to append turns to one session.
_CROSS_PROCESS_WORKER_SOURCE = """
import json
import sys
import time
from pathlib import Path

from app.llm.router import ModelProviderName, ModelRoute
from app.persistence.repositories import SQLiteTurnRepository
from app.persistence.sqlite import connect_sqlite

db_path, session_id, count, ready_path, go_path, output_path = sys.argv[1:7]
count = int(count)

connection = connect_sqlite(db_path)
repository = SQLiteTurnRepository(connection)
route = ModelRoute(
    provider=ModelProviderName.LOCAL,
    model="local-model",
    max_tokens=10,
    temperature=0.1,
    reason="child process worker",
)

# Signal readiness, then wait for the parent's go-ahead so both processes start
# writing at roughly the same wall-clock moment (filesystem handshake, no fixed sleeps).
Path(ready_path).write_text("ready")
deadline = time.time() + 10.0
while not Path(go_path).exists():
    if time.time() > deadline:
        raise SystemExit("child worker timed out waiting for the go signal")
    time.sleep(0.005)

indices = []
errors = []
for i in range(count):
    try:
        turn = repository.append_turn(
            session_id=session_id,
            scene_id="sc",
            persona_id="p",
            user_message=f"child-message-{i}",
            assistant_message=f"child-reply-{i}",
            route=route,
        )
        indices.append(turn.turn_index)
    except Exception as exc:  # noqa: BLE001 -- report every failure mode back to the parent
        errors.append(repr(exc))

Path(output_path).write_text(json.dumps({"indices": indices, "errors": errors}))
"""


def test_append_turn_assigns_contiguous_unique_indices_across_real_processes(
    tmp_path: Path,
) -> None:
    # A genuine second OS process (not a second connection in this interpreter, unlike the
    # WAL tests above) races the pytest process to append turns to the same session in the
    # same on-disk database file (#70). Before the atomic-INSERT fix, both sides computed
    # turn_index via a read-then-write count() and could collide on UNIQUE(session_id,
    # turn_index), raising sqlite3.IntegrityError. After the fix, the index is assigned by
    # a correlated subselect inside the INSERT itself, so concurrent writers serialize on
    # SQLite's write lock and each gets a distinct, contiguous index.
    db_path = tmp_path / "shared.db"
    connection = connect_sqlite(db_path)
    initialize_database(connection)
    _insert_session(connection, "s1")
    connection.commit()

    ready_path = tmp_path / "child_ready"
    go_path = tmp_path / "go"
    output_path = tmp_path / "child_output.json"
    count_per_side = 15

    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _CROSS_PROCESS_WORKER_SOURCE,
            str(db_path),
            "s1",
            str(count_per_side),
            str(ready_path),
            str(go_path),
            str(output_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.time() + 10.0
        while not ready_path.exists():
            if proc.poll() is not None:
                _, stderr = proc.communicate()
                pytest.fail(f"child worker process exited early:\n{stderr}")
            if time.time() > deadline:
                pytest.fail("child worker process never signaled ready")
            time.sleep(0.005)

        repository = SQLiteTurnRepository(connection)
        route = ModelRoute(
            provider=ModelProviderName.LOCAL,
            model="local-model",
            max_tokens=10,
            temperature=0.1,
            reason="parent process worker",
        )

        # Release the child now that both sides are ready to write, then immediately
        # start writing ourselves so the two processes' writes genuinely interleave.
        go_path.write_text("go")

        main_indices: list[int] = []
        main_errors: list[str] = []
        for i in range(count_per_side):
            try:
                turn = repository.append_turn(
                    session_id="s1",
                    scene_id="sc",
                    persona_id="p",
                    user_message=f"main-message-{i}",
                    assistant_message=f"main-reply-{i}",
                    route=route,
                )
                main_indices.append(turn.turn_index)
            except sqlite3.IntegrityError as exc:
                main_errors.append(repr(exc))

        _, stderr = proc.communicate(timeout=15)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()

    assert proc.returncode == 0, f"child worker process failed:\n{stderr}"
    assert main_errors == [], f"parent process saw IntegrityError(s): {main_errors}"

    child_payload = json.loads(output_path.read_text())
    assert child_payload["errors"] == [], (
        f"child process saw error(s): {child_payload['errors']}"
    )

    total = count_per_side * 2
    all_indices = main_indices + child_payload["indices"]
    assert len(all_indices) == total
    assert len(set(all_indices)) == total  # every (session_id, turn_index) pair is unique
    assert sorted(all_indices) == list(range(1, total + 1))  # contiguous 1..N, no gaps

    db_indices = [
        row["turn_index"]
        for row in connection.execute(
            "SELECT turn_index FROM turns WHERE session_id = ? ORDER BY turn_index",
            ("s1",),
        ).fetchall()
    ]
    assert db_indices == list(range(1, total + 1))
    connection.close()
