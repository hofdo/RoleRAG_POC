from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from app.diagnostics.live_checkpoint import (
    STORY_EVENTS,
    CheckpointError,
    EventAttribution,
    PersistenceInspection,
    build_event_attribution,
    conversation_messages,
    resolve_turn_count,
    run_checkpoint,
    validate_warnings,
    write_reports,
)
from app.domain import MemoryEpisode, Visibility


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeClient:
    def __init__(self, *, turn_overrides: Mapping[int, Mapping[str, Any]] | None = None) -> None:
        self.turn_overrides = turn_overrides or {}
        self.turn_index = 0
        self.messages: list[str] = []

    def __enter__(self) -> FakeClient:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def get(self, url: str) -> FakeResponse:
        if url == "/runtime/status":
            return FakeResponse(
                {
                    "content_catalog_available": True,
                    "local_provider_configured": True,
                    "retrieval_configured": True,
                }
            )
        if url == "/content/catalog":
            return FakeResponse({"worlds": [{"id": "demo_world"}]})
        if url == "/sessions/session-1":
            recent = self.messages[-8:]
            return FakeResponse(
                {
                    "session_id": "session-1",
                    "recent_turns": [
                        {"turn_index": index, "user_message": message}
                        for index, message in enumerate(recent, start=1)
                    ],
                }
            )
        raise AssertionError(f"unexpected GET {url}")

    def post(self, url: str, *, json: Mapping[str, Any]) -> FakeResponse:
        if url == "/sessions":
            return FakeResponse({"session_id": "session-1"})
        if url == "/sessions/session-1/turns":
            self.turn_index += 1
            self.messages.append(str(json["message"]))
            payload: dict[str, Any] = {
                "text": (
                    "I remember your promise to return before dawn."
                    if self.turn_index == 13
                    else f"response {self.turn_index}"
                ),
                "route": {"provider": "local", "model": "model-1", "reason": "local"},
                "finish_reason": "stop",
                "memory_written": self.turn_index == 3,
                "warnings": [],
            }
            payload.update(self.turn_overrides.get(self.turn_index, {}))
            return FakeResponse(payload)
        raise AssertionError(f"unexpected POST {url}")


def _inspection(turn_count: int, memory_count: int = 1) -> PersistenceInspection:
    return PersistenceInspection(
        persisted_turn_count=turn_count,
        persisted_memory_count=memory_count,
        canon_lore_count=1,
        session_memory_count=memory_count,
    )


def _attribution(**overrides: Any) -> EventAttribution:
    values: dict[str, Any] = {
        "event_key": "before_dawn_promise",
        "query": "User message: remember my promise",
        "matching_memory_ids": ("memory-1",),
        "indexed_memory_ids": ("memory-1",),
        "selected_memory_ids": ("memory-1",),
        "selected_visibilities": ("player",),
    }
    values.update(overrides)
    return EventAttribution(**values)


def _run(
    *,
    turn_count: int = 13,
    turn_overrides: Mapping[int, Mapping[str, Any]] | None = None,
    inspection: PersistenceInspection | None = None,
    attribution: EventAttribution | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    client = FakeClient(turn_overrides=turn_overrides)
    clock = iter(float(index) for index in range(turn_count * 2))
    return run_checkpoint(
        client_factory=lambda: client,
        inspector=lambda _session_id: inspection or _inspection(turn_count),
        event_inspector=lambda _session_id, _event: attribution or _attribution(),
        expected_model="model-1",
        turn_count=turn_count,
        fail_on_structured_warnings=strict,
        monotonic=lambda: next(clock),
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [("5", 5), ("8", 8), ("12", 12), ("20", 20), ("50", 50)],
)
def test_resolve_turn_count_accepts_supported_values(value: str, expected: int) -> None:
    assert resolve_turn_count(value) == expected


@pytest.mark.parametrize("value", ["4", "51", "eight", "5.5"])
def test_resolve_turn_count_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="5 through 50"):
        resolve_turn_count(value)


def test_all_lengths_share_the_same_story_prefix() -> None:
    fifty = conversation_messages(50)
    for count in (5, 8, 12, 20, 50):
        assert conversation_messages(count) == fifty[:count]
    assert "promise to return before dawn" in fifty[2]
    assert len(fifty) == 50


def test_structured_warnings_are_strict_or_report_only() -> None:
    warnings = [
        "critic skipped: invalid JSON",
        "memory curation skipped: invalid JSON",
        "memory indexing skipped: qdrant offline",
        "retrieval skipped: qdrant offline",
    ]
    with pytest.raises(CheckpointError, match="checkpoint warnings"):
        validate_warnings(warnings, strict=True, turn_index=2)
    assert validate_warnings(warnings, strict=False, turn_index=2)["retrieval"] == 1


def test_build_event_attribution_tracks_exact_memory_ids() -> None:
    event = STORY_EVENTS[0]
    memories = [
        MemoryEpisode(
            id="match",
            session_id="session",
            scene_id="rose-gallery",
            summary="The player promised to return before dawn.",
            importance=4,
            visibility=Visibility.PLAYER,
        ),
        MemoryEpisode(
            id="other",
            session_id="session",
            scene_id="rose-gallery",
            summary="The gallery clock is silent.",
            importance=2,
            visibility=Visibility.PLAYER,
        ),
    ]
    result = build_event_attribution(
        event=event,
        query="real query",
        memories=memories,
        indexed_memory_ids=["match"],
        selected=[
            {
                "id": "match",
                "collection": "session_memory",
                "visibility": "player",
            }
        ],
    )
    assert result.matching_memory_ids == ("match",)
    assert result.indexed_memory_ids == ("match",)
    assert result.selected_memory_ids == ("match",)
    assert result.missed_memory_ids == ()


def test_build_event_attribution_records_retrieval_miss_rank() -> None:
    event = STORY_EVENTS[0]
    memories = [
        MemoryEpisode(
            id="match",
            session_id="session",
            scene_id="rose-gallery",
            summary="The player promised to return before dawn.",
            importance=4,
            visibility=Visibility.PLAYER,
        ),
    ]
    result = build_event_attribution(
        event=event,
        query="real query",
        memories=memories,
        indexed_memory_ids=["match"],
        selected=[
            {
                "id": "lore-1",
                "collection": "canon_lore",
                "visibility": "player",
                "selected_rank": 1,
            }
        ],
        rejected=[
            {"id": "noise", "selected_rank": None, "adjusted_score": 0.4},
            {"id": "match", "selected_rank": None, "adjusted_score": 0.3},
        ],
    )

    assert result.selected_memory_ids == ()
    assert result.missed_memory_ids == ("match",)
    # 1 selected chunk + second position in the rejected tail.
    assert result.missed_memory_ranks == (3,)


def test_extraction_miss_is_report_only_when_not_strict() -> None:
    summary = _run(
        strict=False,
        attribution=_attribution(
            matching_memory_ids=(),
            indexed_memory_ids=(),
            selected_memory_ids=(),
        ),
    )
    assert summary["events"][0]["extracted"] is False
    assert summary["quality_metrics"]["memory_extraction_misses"] == 1


def test_extraction_miss_fails_strict_checkpoint() -> None:
    with pytest.raises(CheckpointError, match="no persisted matching memory"):
        _run(
            attribution=_attribution(
                matching_memory_ids=(),
                indexed_memory_ids=(),
                selected_memory_ids=(),
            )
        )


def test_indexing_miss_is_application_failure() -> None:
    with pytest.raises(CheckpointError, match="missing from Qdrant"):
        _run(attribution=_attribution(indexed_memory_ids=()))


def test_retrieval_miss_is_report_only_when_not_strict() -> None:
    summary = _run(strict=False, attribution=_attribution(selected_memory_ids=()))
    assert summary["events"][0]["selected"] is False


def test_retrieval_miss_fails_strict_checkpoint() -> None:
    with pytest.raises(CheckpointError, match="not selected by callback retrieval"):
        _run(attribution=_attribution(selected_memory_ids=()))


def test_stage_latency_means_are_reported_from_turn_stage_timings() -> None:
    overrides = {
        1: {"stage_timings": {"generation": 10.0, "critique": 4.0, "memory": 2.0}},
        2: {"stage_timings": {"generation": 20.0, "critique": 6.0, "memory": 4.0}},
    }
    summary = _run(turn_overrides=overrides)

    means = summary["quality_metrics"]["stage_latency_means"]
    assert means["generation"] == 15.0
    assert means["critique"] == 5.0
    assert means["memory"] == 3.0


def test_stage_latency_means_are_empty_without_stage_timings() -> None:
    summary = _run(turn_overrides={})

    assert summary["quality_metrics"]["stage_latency_means"] == {}


def test_successful_callback_recall_is_reported() -> None:
    summary = _run()
    assert summary["events"][0]["recalled"] is True
    assert summary["quality_metrics"]["callback_recall_misses"] == 0


def test_callback_miss_is_report_only() -> None:
    summary = _run(turn_overrides={13: {"text": "I remember nothing relevant."}})
    assert summary["events"][0]["recalled"] is False
    assert summary["quality_metrics"]["callback_recall_misses"] == 1


def test_hidden_memory_leakage_is_application_failure() -> None:
    with pytest.raises(CheckpointError, match="selected hidden memory"):
        _run(attribution=_attribution(selected_visibilities=("gm",)))


def test_memory_count_mismatch_fails() -> None:
    with pytest.raises(CheckpointError, match="SQLite/Qdrant"):
        _run(
            inspection=PersistenceInspection(
                persisted_turn_count=13,
                persisted_memory_count=2,
                canon_lore_count=1,
                session_memory_count=1,
            )
        )


def test_report_serialization_redacts_secrets(tmp_path: Path) -> None:
    summary = _run(turn_count=5, inspection=_inspection(5, memory_count=0))
    summary["credentials"] = {
        "api_key": "top-secret",
        "note": "Authorization Bearer top-secret",
    }
    json_path = tmp_path / "checkpoint.json"
    markdown_path = tmp_path / "checkpoint.md"
    write_reports(
        summary,
        json_path=json_path,
        markdown_path=markdown_path,
        secret_values=("top-secret",),
    )
    serialized = json_path.read_text(encoding="utf-8")
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "top-secret" not in serialized
    assert json.loads(serialized)["credentials"]["api_key"] == "***"
    assert "#### Turn 5" in markdown


def test_run_checkpoint_reports_progress_after_each_turn() -> None:
    snapshots: list[dict[str, Any]] = []

    client = FakeClient()
    clock = iter(float(index) for index in range(13 * 2))
    summary = run_checkpoint(
        client_factory=lambda: client,
        inspector=lambda _session_id: _inspection(13),
        event_inspector=lambda _session_id, _event: _attribution(),
        expected_model="model-1",
        turn_count=13,
        fail_on_structured_warnings=True,
        monotonic=lambda: next(clock),
        progress_writer=lambda snapshot: snapshots.append(dict(snapshot)),
    )

    assert summary["status"] == "pass"
    # One snapshot per completed turn plus one taken right after the
    # turn-13 event inspection, before its callback request is sent.
    assert len(snapshots) == 14
    assert snapshots[0]["status"] == "in_progress"
    assert len(snapshots[0]["turns"]) == 1
    pre_callback = snapshots[12]
    assert len(pre_callback["turns"]) == 12
    assert "before_dawn_promise" in pre_callback["event_inspections"]
    assert len(snapshots[-1]["turns"]) == 13
    assert snapshots[-1]["configuration"]["turn_count"] == 13
    assert "before_dawn_promise" in snapshots[-1]["event_inspections"]


def test_run_checkpoint_preserves_progress_when_a_turn_fails() -> None:
    snapshots: list[dict[str, Any]] = []
    client = FakeClient(turn_overrides={5: {"text": ""}})
    clock = iter(float(index) for index in range(13 * 2))

    with pytest.raises(CheckpointError):
        run_checkpoint(
            client_factory=lambda: client,
            inspector=lambda _session_id: _inspection(13),
            event_inspector=lambda _session_id, _event: _attribution(),
            expected_model="model-1",
            turn_count=13,
            fail_on_structured_warnings=True,
            monotonic=lambda: next(clock),
            progress_writer=lambda snapshot: snapshots.append(dict(snapshot)),
        )

    assert len(snapshots) == 4
    assert snapshots[-1]["status"] == "in_progress"
    assert len(snapshots[-1]["turns"]) == 4


def test_run_checkpoint_records_turn_retrieval_diagnostics() -> None:
    retrieval_payload = {
        "query": "What did I promise?",
        "selected": [
            {
                "id": "memory-1",
                "collection": "session_memory",
                "selected_rank": 1,
                "original_score": 0.6,
                "adjusted_score": 0.7,
            }
        ],
        "rejected": [],
    }
    summary = _run(turn_overrides={2: {"retrieval": retrieval_payload}})

    assert summary["turns"][0]["retrieval"] is None
    assert summary["turns"][1]["retrieval"] == retrieval_payload


def test_write_json_atomic_replaces_target_without_leftover_temp(tmp_path: Path) -> None:
    from app.diagnostics.live_checkpoint import write_json_atomic

    path = tmp_path / "nested" / "checkpoint.json"

    write_json_atomic(path, {"status": "in_progress"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "in_progress"}

    write_json_atomic(path, {"status": "pass"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "pass"}
    assert sorted(path.parent.iterdir()) == [path]
