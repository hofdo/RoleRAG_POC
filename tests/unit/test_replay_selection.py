from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from app.diagnostics.replay_selection import (
    DEFAULT_IMPORTANCE_FLOOR,
    MemoryEligibility,
    load_memories,
    main,
    replay_lexical,
    replay_selection,
    summarize,
    target_rank,
)
from app.memory.consolidation import CONSOLIDATED_TAG
from app.persistence.sqlite import connect_sqlite, initialize_database

_SESSION_ID = "replay-session"


def _seed_database(db_path: Path) -> None:
    """A synthetic artifact-shaped SQLite DB -- never touches the real D3 artifact."""
    connection = connect_sqlite(db_path)
    initialize_database(connection)
    now = "2026-07-12T00:00:00Z"
    connection.execute(
        "INSERT INTO sessions "
        "(id, world_id, active_scene_id, active_persona_id, player_name, created_at, "
        "updated_at) VALUES (?,?,?,?,?,?,?)",
        (_SESSION_ID, "world", "scene", "persona", "Player", now, now),
    )
    # 8-char-or-shorter ids so the CLI's memory_id[:8] truncation is a no-op in
    # assertions below.
    rows = [
        # id, importance, visibility, tags
        ("elig-all", 4, "player", ["promise"]),
        # The blue-seal shape (docs/26 3.3): tag-eligible but sub-floor -- the dead
        # predicate this script exists to surface.
        ("elig-tag", 2, "player", ["rule"]),
        ("elig-imp", 4, "player", ["lore"]),
        ("elig-non", 1, "player", []),
        # GM-visible, otherwise identical to "elig-all" -- must be excluded
        # entirely by the visibility filter regardless of importance/tags.
        ("excluded", 5, "gm", ["promise"]),
    ]
    for memory_id, importance, visibility, tags in rows:
        connection.execute(
            "INSERT INTO memory_episodes "
            "(id, session_id, scene_id, actor_id, summary, importance, visibility, "
            "tags_json, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                memory_id,
                _SESSION_ID,
                "scene",
                "persona",
                f"summary for {memory_id}",
                importance,
                visibility,
                json.dumps(tags),
                now,
            ),
        )
    connection.commit()
    connection.close()


def _find(memories: list[MemoryEligibility], memory_id: str) -> MemoryEligibility:
    return next(memory for memory in memories if memory.memory_id == memory_id)


def test_replay_selection_reports_visibility_tag_and_floor_eligibility(tmp_path: Path) -> None:
    db_path = tmp_path / "synthetic.db"
    _seed_database(db_path)

    memories = replay_selection(database_path=db_path, session_id=_SESSION_ID)

    # GM-visible row excluded entirely -- build_standing_facts never considers it.
    assert {memory.memory_id for memory in memories} == {
        "elig-all",
        "elig-tag",
        "elig-imp",
        "elig-non",
    }

    floor_and_tag = _find(memories, "elig-all")
    assert floor_and_tag.matched_canon_tags == ("promise",)
    assert floor_and_tag.eligible is True

    tag_only = _find(memories, "elig-tag")
    assert tag_only.matched_canon_tags == ("rule",)
    assert tag_only.eligible is False  # the dead-predicate case (docs/26 3.3)

    floor_only = _find(memories, "elig-imp")
    assert floor_only.matched_canon_tags == ()
    assert floor_only.eligible is False

    neither = _find(memories, "elig-non")
    assert neither.matched_canon_tags == ()
    assert neither.eligible is False


def test_replay_selection_default_importance_floor_is_four(tmp_path: Path) -> None:
    assert DEFAULT_IMPORTANCE_FLOOR == 4
    db_path = tmp_path / "synthetic.db"
    _seed_database(db_path)

    default_floor = replay_selection(database_path=db_path, session_id=_SESSION_ID)
    explicit_floor = replay_selection(
        database_path=db_path, session_id=_SESSION_ID, importance_floor=4
    )

    assert default_floor == explicit_floor


def test_replay_selection_lower_floor_widens_eligibility(tmp_path: Path) -> None:
    db_path = tmp_path / "synthetic.db"
    _seed_database(db_path)

    memories = replay_selection(database_path=db_path, session_id=_SESSION_ID, importance_floor=2)

    assert _find(memories, "elig-tag").eligible is True
    assert _find(memories, "elig-all").eligible is True


def test_replay_selection_unknown_session_returns_empty(tmp_path: Path) -> None:
    db_path = tmp_path / "synthetic.db"
    _seed_database(db_path)

    assert replay_selection(database_path=db_path, session_id="no-such-session") == []


def test_replay_selection_opens_read_only_and_leaves_no_sidecar_files(tmp_path: Path) -> None:
    # The exact hazard docs/26 Stage 0 calls out: mode=ro alone still lets SQLite
    # create -wal/-shm sidecars for a WAL-mode database on open.
    db_path = tmp_path / "synthetic.db"
    _seed_database(db_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["synthetic.db"]

    replay_selection(database_path=db_path, session_id=_SESSION_ID)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["synthetic.db"]


def test_summarize_reports_distribution_and_eligibility_counts(tmp_path: Path) -> None:
    db_path = tmp_path / "synthetic.db"
    _seed_database(db_path)
    memories = replay_selection(database_path=db_path, session_id=_SESSION_ID)

    line = summarize(memories, importance_floor=DEFAULT_IMPORTANCE_FLOOR)

    assert "player_memories=4" in line
    assert "importance_floor=4" in line
    # elig-non=1, elig-tag=2, elig-all=4, elig-imp=4 -> one 1, one 2, two 4s.
    assert "importance_distribution=[1x1, 1x2, 2x4]" in line
    assert "tag_eligible=2" in line  # elig-all (promise), elig-tag (rule)
    assert "eligible=1" in line  # elig-all only: elig-tag fails the floor


def test_main_cli_prints_per_memory_lines_and_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    db_path = tmp_path / "synthetic.db"
    _seed_database(db_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["replay_selection", "--database-path", str(db_path), "--session-id", _SESSION_ID],
    )

    main()

    out = capsys.readouterr().out
    assert "elig-all importance=4 tags=['promise'] canon_tags=['promise'] eligible=yes" in out
    assert "elig-tag importance=2 tags=['rule'] canon_tags=['rule'] eligible=no" in out
    assert "player_memories=4" in out


# --- docs/26 §3.3 --pinning flag-on eligibility mode (#78) ------------------------


def test_replay_selection_pinning_widens_eligibility_to_tag_only_memory(
    tmp_path: Path,
) -> None:
    # elig-tag is the blue-seal shape: tag-eligible (rule) but sub-floor (importance=2)
    # -- excluded without pinning, eligible with it, exactly as build_standing_facts's
    # own `... or canon_tag_pinning` clause behaves.
    db_path = tmp_path / "synthetic.db"
    _seed_database(db_path)

    without_pinning = replay_selection(database_path=db_path, session_id=_SESSION_ID)
    with_pinning = replay_selection(
        database_path=db_path, session_id=_SESSION_ID, canon_tag_pinning=True
    )

    assert _find(without_pinning, "elig-tag").eligible is False
    assert _find(with_pinning, "elig-tag").eligible is True
    # Already-floor-eligible facts are unaffected by the flag.
    assert _find(with_pinning, "elig-all").eligible is True


def test_replay_selection_pinning_does_not_widen_untagged_memory(tmp_path: Path) -> None:
    # elig-imp clears the importance floor but carries no CANON_TAG -- pinning widens
    # the floor requirement, not the tag requirement, so it stays ineligible.
    db_path = tmp_path / "synthetic.db"
    _seed_database(db_path)

    with_pinning = replay_selection(
        database_path=db_path, session_id=_SESSION_ID, canon_tag_pinning=True
    )

    assert _find(with_pinning, "elig-imp").eligible is False
    assert _find(with_pinning, "elig-non").eligible is False


def test_summarize_reports_canon_tag_pinning_state(tmp_path: Path) -> None:
    db_path = tmp_path / "synthetic.db"
    _seed_database(db_path)
    memories = replay_selection(
        database_path=db_path, session_id=_SESSION_ID, canon_tag_pinning=True
    )

    line = summarize(memories, importance_floor=DEFAULT_IMPORTANCE_FLOOR, canon_tag_pinning=True)

    assert "canon_tag_pinning=True" in line
    # elig-all and elig-tag both eligible with pinning on; elig-imp/elig-non are not.
    assert "eligible=2" in line


def test_main_cli_pinning_flag_reports_widened_eligibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    db_path = tmp_path / "synthetic.db"
    _seed_database(db_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "replay_selection",
            "--database-path",
            str(db_path),
            "--session-id",
            _SESSION_ID,
            "--pinning",
        ],
    )

    main()

    out = capsys.readouterr().out
    assert "elig-tag importance=2 tags=['rule'] canon_tags=['rule'] eligible=yes" in out
    assert "canon_tag_pinning=True" in out
    assert "eligible=2" in out


# --- Lane B lexical replay (docs/26 Stage 4, #79) --------------------------------

_LEXICAL_QUERY = "what rule did we agree to trust"


def _seed_lexical_database(db_path: Path) -> None:
    """A synthetic artifact-shaped DB for the lexical scorer -- never the real D3."""
    connection = connect_sqlite(db_path)
    initialize_database(connection)
    now = "2026-07-12T00:00:00Z"
    connection.execute(
        "INSERT INTO sessions "
        "(id, world_id, active_scene_id, active_persona_id, player_name, created_at, "
        "updated_at) VALUES (?,?,?,?,?,?,?)",
        (_SESSION_ID, "world", "scene", "persona", "Player", now, now),
    )
    rows = [
        # id, summary, visibility, tags -- "target" is the only memory carrying the
        # rare terms (rule/trust) the query shares; fillers share nothing.
        ("target", "the trust rule uses a blue wax seal", "player", ["rule"]),
        ("filler1", "iria stood near the busy market", "player", []),
        ("filler2", "the palace courtyard was quiet", "player", []),
        # Excluded from the pool: gm visibility, and a consolidation-retired original.
        ("gmrow", "the trust rule uses a blue wax seal", "gm", ["rule"]),
        ("retired", "the trust rule uses a blue wax seal", "player", ["rule", CONSOLIDATED_TAG]),
    ]
    for memory_id, summary, visibility, tags in rows:
        connection.execute(
            "INSERT INTO memory_episodes "
            "(id, session_id, scene_id, actor_id, summary, importance, visibility, "
            "tags_json, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                memory_id,
                _SESSION_ID,
                "scene",
                "persona",
                summary,
                2,
                visibility,
                json.dumps(tags),
                now,
            ),
        )
    connection.commit()
    connection.close()


def test_replay_lexical_ranks_the_rare_term_target_first(tmp_path: Path) -> None:
    db_path = tmp_path / "synthetic.db"
    _seed_lexical_database(db_path)

    hits = replay_lexical(
        database_path=db_path, session_id=_SESSION_ID, query_text=_LEXICAL_QUERY
    )

    # Only "target" carries the rare rule/trust terms; gm + consolidated rows are
    # excluded from the pool entirely.
    assert [hit.memory_id for hit in hits] == ["target"]
    assert target_rank(hits, target_prefix="target") == 1


def test_replay_lexical_target_absent_when_no_terms_match(tmp_path: Path) -> None:
    db_path = tmp_path / "synthetic.db"
    _seed_lexical_database(db_path)

    hits = replay_lexical(
        database_path=db_path, session_id=_SESSION_ID, query_text="unrelated words entirely"
    )

    assert target_rank(hits, target_prefix="target") is None


def test_load_memories_is_read_only_and_leaves_no_sidecar_files(tmp_path: Path) -> None:
    db_path = tmp_path / "synthetic.db"
    _seed_lexical_database(db_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["synthetic.db"]

    memories = load_memories(database_path=db_path, session_id=_SESSION_ID)

    assert len(memories) == 5  # all rows loaded; the scorer applies the pool filter
    assert sorted(p.name for p in tmp_path.iterdir()) == ["synthetic.db"]


def test_main_cli_lexical_mode_reports_rank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    db_path = tmp_path / "synthetic.db"
    _seed_lexical_database(db_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "replay_selection",
            "--database-path",
            str(db_path),
            "--session-id",
            _SESSION_ID,
            "--lexical-query",
            _LEXICAL_QUERY,
            "--target-id",
            "target",
        ],
    )

    main()

    out = capsys.readouterr().out
    assert "#1 target" in out
    assert "target=target -> rank 1/1" in out
