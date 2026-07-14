from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from app.diagnostics.live_checkpoint import (
    ROSE_GALLERY_MESSAGES,
    STORY_EVENTS,
    CheckpointError,
    EventAttribution,
    PersistenceInspection,
    build_event_attribution,
    context_warning_counts,
    conversation_messages,
    events_for_turn_count,
    resolve_turn_count,
    run_checkpoint,
    semantic_match,
    validate_warnings,
    write_reports,
)
from app.domain import MemoryEpisode, Visibility
from app.rag.vector_store import QdrantVectorStore


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


def _inspection(
    turn_count: int,
    memory_count: int = 1,
    *,
    consolidated_memory_count: int = 0,
    consolidation_summary_count: int = 0,
) -> PersistenceInspection:
    return PersistenceInspection(
        persisted_turn_count=turn_count,
        persisted_memory_count=memory_count,
        canon_lore_count=1,
        session_memory_count=memory_count,
        consolidated_memory_count=consolidated_memory_count,
        consolidation_summary_count=consolidation_summary_count,
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
    [("5", 5), ("8", 8), ("12", 12), ("20", 20), ("50", 50), ("100", 100)],
)
def test_resolve_turn_count_accepts_supported_values(value: str, expected: int) -> None:
    assert resolve_turn_count(value) == expected


@pytest.mark.parametrize("value", ["4", "101", "eight", "5.5"])
def test_resolve_turn_count_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="5 through 100"):
        resolve_turn_count(value)


def test_all_lengths_share_the_same_story_prefix() -> None:
    full = conversation_messages(100)
    for count in (5, 8, 12, 20, 50, 100):
        assert conversation_messages(count) == full[:count]
    assert "promise to return before dawn" in full[2]
    assert len(full) == 100


def test_story_events_have_unique_callback_turns_within_bounds() -> None:
    callback_turns = [event.callback_turn for event in STORY_EVENTS]
    assert len(callback_turns) == len(set(callback_turns))
    for event in STORY_EVENTS:
        assert 1 <= event.definition_turn <= 100
        assert 1 <= event.callback_turn <= 100
        assert event.definition_turn < event.callback_turn


def test_story_events_still_leave_exactly_100_scripted_messages() -> None:
    assert len(ROSE_GALLERY_MESSAGES) == 100


def test_story_event_definition_messages_satisfy_their_own_semantic_match() -> None:
    # The scripted line that states the fact must contain every keyword group so a
    # faithful summary of it can plausibly satisfy semantic_match at callback time.
    for event in STORY_EVENTS:
        definition_message = ROSE_GALLERY_MESSAGES[event.definition_turn - 1]
        assert semantic_match(definition_message, event.term_groups), event.key


def test_story_event_callback_messages_reference_at_least_one_term_group() -> None:
    # Callback questions deliberately withhold most of the answer (they prompt recall
    # rather than restate it), but should still gesture at the event via some term.
    for event in STORY_EVENTS:
        callback_message = ROSE_GALLERY_MESSAGES[event.callback_turn - 1].casefold()
        assert any(
            any(term.casefold() in callback_message for term in group)
            for group in event.term_groups
        ), event.key


def test_at_least_one_story_event_is_a_long_gap_late_recall_probe() -> None:
    # Acceptance criterion from docs/22: an old durable fact must still be retrievable
    # near the end of a 100-turn run, not just shortly after it was stated.
    gaps = [event.callback_turn - event.definition_turn for event in STORY_EVENTS]
    assert max(gaps) >= 60


# Keys of the five events that existed before this change, all with callback_turn <= 50.
_ORIGINAL_FIVE_KEYS: set[str] = {
    "before_dawn_promise",
    "silver_compass",
    "blue_seal_trust_rule",
    "key_hiding_place",
    "three_tap_signal",
}


@pytest.mark.parametrize(
    ("turn_count", "expected_keys"),
    [
        (50, _ORIGINAL_FIVE_KEYS),
        (64, _ORIGINAL_FIVE_KEYS),
        (65, _ORIGINAL_FIVE_KEYS | {"amber_ring_token"}),
        (79, _ORIGINAL_FIVE_KEYS | {"amber_ring_token"}),
        (80, _ORIGINAL_FIVE_KEYS | {"amber_ring_token", "north_stair_rendezvous"}),
        (94, _ORIGINAL_FIVE_KEYS | {"amber_ring_token", "north_stair_rendezvous"}),
        (95, {event.key for event in STORY_EVENTS}),
        (100, {event.key for event in STORY_EVENTS}),
    ],
)
def test_events_for_turn_count_boundaries_at_late_recall_turns(
    turn_count: int, expected_keys: set[str]
) -> None:
    selected_keys = {event.key for event in events_for_turn_count(turn_count)}
    assert selected_keys == expected_keys


def test_events_for_turn_count_excludes_long_gap_event_before_its_callback() -> None:
    # A partial run whose turn_count falls between the long-gap event's early setup
    # (turn 17) and its late callback (turn 95) must not assert on it at all: the
    # setup happening is harmless, but the callback is out of range so it is filtered.
    selected_keys = {event.key for event in events_for_turn_count(60)}
    assert "hollow_bookend_note" not in selected_keys


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


def test_context_warning_counts_counts_both_prefixes_and_zero_case() -> None:
    # #69 evidence: context_preflight_warning and context_usage_warning both start
    # with "context preflight: " (see CONTEXT_WARNING_PREFIXES) but are told apart
    # by what follows -- "estimated" for the preflight estimate, "actual
    # prompt_tokens" for the post-hoc real-usage warning.
    warnings = [
        "context preflight: estimated 9000 tokens vs window 8000 (warn ratio 0.85)",
        "context preflight: actual prompt_tokens 8500 exceeded 0.85 * window 8000",
        "context preflight: actual prompt_tokens 8600 exceeded 0.85 * window 8000",
        "retrieval skipped: qdrant offline",
        "some unrelated warning",
    ]
    assert context_warning_counts(warnings) == {"preflight": 1, "actual": 2}
    # Zero case: MODEL_CONTEXT_WINDOW_TOKENS unset means neither warning ever fires.
    assert context_warning_counts([]) == {"preflight": 0, "actual": 0}


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


# --- #76 provenance-OR-phrasing attribution (docs/26 §3.1, §4's correction) ---
#
# A memory counts toward matching_memory_ids if it phrasing-matches (semantic_match,
# pre-#76 behavior, unchanged) OR its source_turn_id equals the probe's resolved
# definition-turn id -- never provenance-only. definition_turn_id=None (the default,
# and what every pre-#76 caller passes) disables the provenance leg entirely, which
# is exactly what the two tests above already exercise.


def test_build_event_attribution_matches_via_provenance_when_phrasing_fails() -> None:
    # The curator's paraphrase drifted far enough that semantic_match fails, but the
    # episode still carries the probe's definition-turn provenance -- this is the
    # concrete paraphrase-fragility class #76 exists to close (docs/26 §4, the
    # "symbol of a pact/promise" live finding).
    event = STORY_EVENTS[0]  # before_dawn_promise, definition_turn=3
    memories = [
        MemoryEpisode(
            id="paraphrased",
            session_id="session",
            scene_id="rose-gallery",
            summary="A pact was sealed regarding a future return.",  # no phrasing match
            importance=4,
            visibility=Visibility.PLAYER,
            source_turn_id=3,
        ),
    ]

    result = build_event_attribution(
        event=event,
        query="real query",
        memories=memories,
        indexed_memory_ids=["paraphrased"],
        selected=[],
        definition_turn_id=3,
    )

    assert result.matching_memory_ids == ("paraphrased",)


def test_build_event_attribution_phrasing_still_matches_when_provenance_points_elsewhere() -> None:
    # The OR is not an AND: a memory whose phrasing matches counts even when its
    # source_turn_id does not equal the resolved definition-turn id (e.g. the
    # episode came from a different turn, or provenance could not be resolved for
    # this particular memory).
    event = STORY_EVENTS[0]  # before_dawn_promise, definition_turn=3
    memories = [
        MemoryEpisode(
            id="match",
            session_id="session",
            scene_id="rose-gallery",
            summary="The player promised to return before dawn.",
            importance=4,
            visibility=Visibility.PLAYER,
            source_turn_id=99,  # not the resolved definition turn (3)
        ),
    ]

    result = build_event_attribution(
        event=event,
        query="real query",
        memories=memories,
        indexed_memory_ids=["match"],
        selected=[],
        definition_turn_id=3,
    )

    assert result.matching_memory_ids == ("match",)


def test_build_event_attribution_provenance_over_attributes_on_multi_memory_definition_turn() -> (
    None
):
    # Documented residual (docs/26 §4): a definition turn that produces SEVERAL
    # memory episodes links ALL of them to the same source_turn_id. If NEITHER
    # phrasing-matches the probe, provenance alone cannot tell which one is the
    # probed fact -- BOTH count as "matching" via the OR. This is the bounded
    # trade docs/26 accepts (never provenance-ONLY, but still not immune to this
    # specific multi-memory ambiguity) -- pinned here, not silently tightened away.
    event = STORY_EVENTS[0]  # before_dawn_promise, definition_turn=3
    memories = [
        MemoryEpisode(
            id="unrelated-fact-same-turn",
            session_id="session",
            scene_id="rose-gallery",
            summary="The player also asked about the west corridor guard schedule.",
            importance=2,
            visibility=Visibility.PLAYER,
            source_turn_id=3,
        ),
        MemoryEpisode(
            id="another-unrelated-fact-same-turn",
            session_id="session",
            scene_id="rose-gallery",
            summary="The player noted the gallery's clock had stopped.",
            importance=2,
            visibility=Visibility.PLAYER,
            source_turn_id=3,
        ),
    ]

    result = build_event_attribution(
        event=event,
        query="real query",
        memories=memories,
        indexed_memory_ids=["unrelated-fact-same-turn", "another-unrelated-fact-same-turn"],
        selected=[],
        definition_turn_id=3,
    )

    assert set(result.matching_memory_ids) == {
        "unrelated-fact-same-turn",
        "another-unrelated-fact-same-turn",
    }


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


# --- #74/#75 definition-turn fail-fast (docs/26 Stage 0) ---
#
# docs/25 Phase D run 4: a probe's DEFINITION turn (before_dawn_promise here: turn 3,
# callback turn 13) ending in `outcome: controlled_failure` used to march on for ten
# more turns and then fail at the callback with a misleading "no persisted matching
# memory" -- the real cause (the fact never entered the world) was only recoverable
# by a forensic DB diff. This is a precision fix, not a loosening: the run still
# fails in exactly the cases it failed before (strict mode only, since the check it
# preempts -- _validate_attribution's matching_memory_ids requirement -- was itself
# strict-only), just named at the turn that actually caused it.


def test_definition_turn_controlled_failure_fails_fast_with_named_cause() -> None:
    with pytest.raises(
        CheckpointError,
        match=r"definition turn 3 for event before_dawn_promise ended in controlled_failure",
    ) as excinfo:
        _run(turn_overrides={3: {"outcome": "controlled_failure"}})
    # The old misleading message must not be what actually surfaces here.
    assert "no persisted matching memory" not in str(excinfo.value)


def test_definition_turn_controlled_failure_is_report_only_when_not_strict() -> None:
    # Precision, not loosening, cuts both ways: a non-strict (report-only) run never
    # failed on this class of miss before, and must not start failing now either.
    summary = _run(strict=False, turn_overrides={3: {"outcome": "controlled_failure"}})
    assert summary["status"] == "pass"
    assert summary["turns"][2]["outcome"] == "controlled_failure"


def test_non_definition_turn_controlled_failure_does_not_fail_fast() -> None:
    # Turn 5 is neither a definition nor a callback turn within a 13-turn run -- a
    # controlled failure there is unrelated to any probe and must not trip the new
    # check (nor anything else, with the default passing attribution/inspection).
    summary = _run(turn_overrides={5: {"outcome": "controlled_failure"}})
    assert summary["status"] == "pass"
    assert summary["turns"][4]["outcome"] == "controlled_failure"


def test_definition_turn_controlled_failure_stops_before_later_turns() -> None:
    snapshots: list[dict[str, Any]] = []
    client = FakeClient(turn_overrides={3: {"outcome": "controlled_failure"}})
    clock = iter(float(index) for index in range(13 * 2))

    with pytest.raises(CheckpointError, match="definition turn 3"):
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

    # Turns 4-13 (including the turn-13 callback) never ran; turn 3 itself is
    # captured in the last snapshot -- auditable, not silently discarded.
    assert len(snapshots) == 3
    assert snapshots[-1]["turns"][-1]["turn_index"] == 3
    assert snapshots[-1]["turns"][-1]["outcome"] == "controlled_failure"


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


def test_report_includes_consolidation_and_context_warning_evidence_fields() -> None:
    # Step-2 prep (P2.2 consolidation + #69 context-accounting): all four fields are
    # report-only -- they must appear in the summary without perturbing anything
    # strict mode already checks (none of the injected strings match
    # STRICT_WARNING_PREFIXES, so this runs under the default strict=True).
    overrides = {
        2: {
            "warnings": [
                "context preflight: estimated 9000 tokens vs window 8000 (warn ratio 0.85)",
            ],
        },
        4: {
            "warnings": [
                "context preflight: actual prompt_tokens 8500 exceeded 0.85 * window 8000",
                "context preflight: actual prompt_tokens 8600 exceeded 0.85 * window 8000",
            ],
        },
    }
    summary = _run(
        turn_overrides=overrides,
        inspection=_inspection(
            13,
            memory_count=2,
            consolidated_memory_count=5,
            consolidation_summary_count=2,
        ),
    )

    assert summary["persisted"]["consolidated_memory_count"] == 5
    assert summary["persisted"]["consolidation_summary_count"] == 2
    assert summary["quality_metrics"]["context_preflight_warning_count"] == 1
    assert summary["quality_metrics"]["context_actual_warning_count"] == 2


def test_report_context_warning_counts_are_zero_without_any_context_warnings() -> None:
    # Zero-case, full pipeline: no turn emits a context-accounting warning (the
    # MODEL_CONTEXT_WINDOW_TOKENS-unset default), so both aggregates stay 0.
    summary = _run()

    assert summary["quality_metrics"]["context_preflight_warning_count"] == 0
    assert summary["quality_metrics"]["context_actual_warning_count"] == 0
    assert summary["persisted"]["consolidated_memory_count"] == 0
    assert summary["persisted"]["consolidation_summary_count"] == 0


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


# --- P1.4 fingerprint sentinel excluded from count paths (cross-review P4, 2026-07-11) --
#
# The embedding-model fingerprint sentinel (docs/22 P1.4) is a real point
# ``ensure_collection`` writes into every collection it touches. ``_qdrant_count`` must
# exclude it (the same ``must_not`` mechanism ``QdrantVectorStore._build_qdrant_filter``
# already applies on every search path), or the checkpoint's unfiltered canon-lore count
# would be inflated by 1 -- letting the ">=1 lore" gate pass on a stamped-but-empty
# collection and diverging from ``InMemoryVectorStore``, whose fingerprint is a separate
# dict entry, never a counted point. Runs against an embedded ``QdrantClient(":memory:")``
# (same pattern as tests/unit/test_vector_store_fingerprint.py) so the real sentinel/filter
# code path is exercised.


def _qdrant_store() -> QdrantVectorStore:
    from qdrant_client import QdrantClient

    store = QdrantVectorStore(url="local")
    store._client = QdrantClient(":memory:")
    return store


def test_qdrant_count_excludes_sentinel_on_stamped_empty_collection() -> None:
    from app.diagnostics.live_checkpoint import _qdrant_count
    from app.rag.models import RagCollection

    store = _qdrant_store()
    # Stamps the fingerprint sentinel into a brand-new, otherwise-empty collection --
    # exactly what happens the first time any indexer/ingest call passes model_key.
    store.ensure_collection(RagCollection.CANON_LORE, 3, model_key="all-MiniLM-L6-v2")

    assert _qdrant_count(store, RagCollection.CANON_LORE) == 0


def test_qdrant_count_excludes_sentinel_alongside_real_points() -> None:
    from app.diagnostics.live_checkpoint import _qdrant_count
    from app.rag.models import RagChunk, RagCollection

    store = _qdrant_store()
    store.ensure_collection(RagCollection.CANON_LORE, 3, model_key="all-MiniLM-L6-v2")
    chunks = [
        RagChunk(
            id=f"lore-{i}",
            source="lore.md",
            source_type="lore",
            text=f"canon fact {i}",
            visibility=Visibility.PLAYER,
            world_id="w1",
        )
        for i in range(3)
    ]
    store.upsert_chunks(RagCollection.CANON_LORE, chunks, [[1.0, 0.0, 0.0]] * 3)

    assert _qdrant_count(store, RagCollection.CANON_LORE) == 3


def test_qdrant_count_missing_collection_is_zero() -> None:
    from app.diagnostics.live_checkpoint import _qdrant_count
    from app.rag.models import RagCollection

    store = _qdrant_store()
    assert _qdrant_count(store, RagCollection.CANON_LORE) == 0


def test_inspect_live_state_excludes_consolidated_rows_from_persisted_memory_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # docs/22 P2 (fixed 2026-07-11): consolidation (C2) marks originals CONSOLIDATED_TAG
    # in SQLite but never deletes the row, while unindexing them from Qdrant. An
    # unfiltered SQLite count would diverge from the Qdrant-indexed count on the very
    # first roll-up and fail the checkpoint's SQLite/Qdrant parity assertion forever
    # after. persisted_memory_count must exclude consolidated-marked rows to stay
    # comparable.
    from app.config import Settings
    from app.diagnostics.live_checkpoint import inspect_live_state
    from app.persistence.sqlite import connect_sqlite, initialize_database

    db_path = tmp_path / "checkpoint.db"
    connection = connect_sqlite(db_path)
    initialize_database(connection)
    now = "2026-07-11T00:00:00Z"
    connection.execute(
        "INSERT INTO sessions "
        "(id, world_id, active_scene_id, active_persona_id, player_name, created_at, "
        "updated_at) VALUES (?,?,?,?,?,?,?)",
        ("session-1", "world", "scene", "persona", "Player", now, now),
    )
    rows = [
        ("mem-1", "[]"),  # ordinary, still retrievable
        ("mem-2", "[]"),  # ordinary, still retrievable
        ("mem-3", json.dumps(["consolidated"])),  # folded original, SQLite-only now
        ("mem-4", json.dumps(["consolidation_summary"])),  # the roll-up itself, indexed
    ]
    for memory_id, tags_json in rows:
        connection.execute(
            "INSERT INTO memory_episodes "
            "(id, session_id, scene_id, actor_id, summary, importance, visibility, "
            "tags_json, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (memory_id, "session-1", "scene", "persona", "summary", 1, "player", tags_json, now),
        )
    connection.commit()
    connection.close()

    monkeypatch.setattr(
        "app.diagnostics.live_checkpoint._qdrant_count",
        lambda vector_store, collection, session_id=None: 3,
    )

    inspection = inspect_live_state(
        database_path=db_path,
        settings=Settings(database_path=str(db_path)),
        session_id="session-1",
    )

    # 4 rows persisted, but "mem-3" (consolidated) is excluded -- matching the Qdrant
    # side, which no longer indexes it (patched above to 3: mem-1, mem-2, mem-4).
    assert inspection.persisted_memory_count == 3
    assert inspection.session_memory_count == 3


def test_inspect_live_state_reports_consolidated_and_summary_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Step-2 prep (P2.2 evidence): consolidated_memory_count/consolidation_summary_count
    # are report-only counts of CONSOLIDATED_TAG/SUMMARY_TAG rows, counted separately
    # from the parity-preserving persisted_memory_count so a P2.2 run (consolidation
    # preset) can confirm consolidation actually fired and how much it rolled up.
    from app.config import Settings
    from app.diagnostics.live_checkpoint import inspect_live_state
    from app.persistence.sqlite import connect_sqlite, initialize_database

    db_path = tmp_path / "checkpoint.db"
    connection = connect_sqlite(db_path)
    initialize_database(connection)
    now = "2026-07-11T00:00:00Z"
    connection.execute(
        "INSERT INTO sessions "
        "(id, world_id, active_scene_id, active_persona_id, player_name, created_at, "
        "updated_at) VALUES (?,?,?,?,?,?,?)",
        ("session-1", "world", "scene", "persona", "Player", now, now),
    )
    rows = [
        ("mem-1", "[]"),  # ordinary, still retrievable
        ("mem-2", json.dumps(["consolidated"])),  # folded original #1
        ("mem-3", json.dumps(["consolidated"])),  # folded original #2
        ("mem-4", json.dumps(["consolidation_summary"])),  # the one roll-up summary
    ]
    for memory_id, tags_json in rows:
        connection.execute(
            "INSERT INTO memory_episodes "
            "(id, session_id, scene_id, actor_id, summary, importance, visibility, "
            "tags_json, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (memory_id, "session-1", "scene", "persona", "summary", 1, "player", tags_json, now),
        )
    connection.commit()
    connection.close()

    monkeypatch.setattr(
        "app.diagnostics.live_checkpoint._qdrant_count",
        lambda vector_store, collection, session_id=None: 2,
    )

    inspection = inspect_live_state(
        database_path=db_path,
        settings=Settings(database_path=str(db_path)),
        session_id="session-1",
    )

    assert inspection.consolidated_memory_count == 2
    assert inspection.consolidation_summary_count == 1


def test_inspect_live_state_consolidation_counts_are_zero_without_tagged_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Zero case: a run with consolidation off (memory_consolidation_threshold <= 0,
    # the default) never tags anything, so both counts stay 0.
    from app.config import Settings
    from app.diagnostics.live_checkpoint import inspect_live_state
    from app.persistence.sqlite import connect_sqlite, initialize_database

    db_path = tmp_path / "checkpoint.db"
    connection = connect_sqlite(db_path)
    initialize_database(connection)
    now = "2026-07-11T00:00:00Z"
    connection.execute(
        "INSERT INTO sessions "
        "(id, world_id, active_scene_id, active_persona_id, player_name, created_at, "
        "updated_at) VALUES (?,?,?,?,?,?,?)",
        ("session-1", "world", "scene", "persona", "Player", now, now),
    )
    connection.execute(
        "INSERT INTO memory_episodes "
        "(id, session_id, scene_id, actor_id, summary, importance, visibility, "
        "tags_json, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        ("mem-1", "session-1", "scene", "persona", "summary", 1, "player", "[]", now),
    )
    connection.commit()
    connection.close()

    monkeypatch.setattr(
        "app.diagnostics.live_checkpoint._qdrant_count",
        lambda vector_store, collection, session_id=None: 1,
    )

    inspection = inspect_live_state(
        database_path=db_path,
        settings=Settings(database_path=str(db_path)),
        session_id="session-1",
    )

    assert inspection.consolidated_memory_count == 0
    assert inspection.consolidation_summary_count == 0


def test_qdrant_count_session_filter_still_excludes_sentinel() -> None:
    from app.diagnostics.live_checkpoint import _qdrant_count
    from app.rag.models import RagChunk, RagCollection

    store = _qdrant_store()
    store.ensure_collection(RagCollection.SESSION_MEMORY, 3, model_key="all-MiniLM-L6-v2")
    chunk = RagChunk(
        id="mem-1",
        source="memory_episode:mem-1",
        source_type="session_memory",
        text="a memory",
        visibility=Visibility.PLAYER,
        session_id="session-a",
    )
    store.upsert_chunks(RagCollection.SESSION_MEMORY, [chunk], [[1.0, 0.0, 0.0]])

    assert _qdrant_count(store, RagCollection.SESSION_MEMORY, session_id="session-a") == 1
    assert _qdrant_count(store, RagCollection.SESSION_MEMORY, session_id="session-b") == 0
