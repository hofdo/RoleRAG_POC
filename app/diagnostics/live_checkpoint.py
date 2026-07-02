from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import statistics
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import httpx

from app.composition import build_actor_context_retriever, build_file_loader
from app.config import Settings
from app.diagnostics.retrieval_miss import summarize_retrieval_miss
from app.domain import MemoryEpisode, Visibility
from app.persistence import SQLiteMemoryRepository, SQLiteTurnRepository, connect_sqlite
from app.rag.models import RagCollection
from app.rag.retriever import build_retrieval_query
from app.rag.vector_store import QdrantVectorStore

MIN_TURN_COUNT = 5
MAX_TURN_COUNT = 100
DEFAULT_TURN_COUNT = 8

ROSE_GALLERY_MESSAGES = (
    "I step into the Rose Gallery and ask Iria what first changed tonight.",
    "I lower my voice and ask which courtier she trusts least in this room.",
    "I promise to return before dawn if she can keep the archive door unbarred.",
    "I ask what the mirrors have shown her about the regent's messengers.",
    "I inspect the nearest rose arrangement and ask whether it hides a signal.",
    "I ask Iria to describe the safest path from here to the old archive.",
    "I tell her I noticed the servant by the west door and ask if that matters.",
    "I ask what price the regent would pay to control the old archive.",
    "I request a private walk with Iria through the gallery.",
    "I ask what she has truly been afraid to say aloud tonight.",
    "I ask her to name one harmless action that will not alarm the court.",
    "I study the clock and ask why its chime has been silenced.",
    "I ask whether she remembers the promise I made about returning.",
    "I ask which corridor the regent's envoy used after leaving the gallery.",
    "I compare the west-door lock with the archive key in Iria's sketch.",
    "I ask whether the roses were rearranged before or after the envoy departed.",
    "I listen for footsteps and ask Iria who is still awake below us.",
    "I ask what record in the archive would most threaten the regent.",
    "I ask Iria what she needs me to understand before I leave the gallery.",
    "I pause beside the final mirror and ask what our next move should be.",
    "I give Iria a silver compass and ask her to keep it until I return.",
    "I ask whether the eastern stair is watched at this hour.",
    "I ask what name the servants use for the sealed archive ledger.",
    "I inspect the gallery clock for scratches around its brass face.",
    "I ask Iria which messenger arrived without a palace seal.",
    "I ask whether the conservatory passage still opens behind the roses.",
    "I tell Iria we will trust only messages carrying a blue wax seal.",
    "I ask which courtier has recently changed their usual route.",
    "I ask what the old archivists recorded about the winter succession.",
    "I listen while Iria describes the sound of the hidden archive latch.",
    "I ask Iria to return the item I entrusted to her earlier.",
    "I ask whether anyone has searched the clock since the envoy left.",
    "I ask what evidence could persuade a neutral member of the court.",
    "I tell Iria I will hide the spare archive key beneath the third rose pedestal.",
    "I ask which guard changes duty at the midnight bell.",
    "I ask whether the gallery servants can be sent away without suspicion.",
    "I compare the envoy's handwriting with a note in the public ledger.",
    "I ask Iria what rule we agreed to use before trusting any message.",
    "I ask what route would let us reach the archive without crossing the ballroom.",
    "I agree that three soft taps on the west door will be our safe signal.",
    "I ask whether the regent's secretary has entered the gallery tonight.",
    "I ask Iria to recount the order in which the suspicious guests arrived.",
    "I ask which mirror gives the clearest view of the eastern stair.",
    "I ask whether the archive ledger can be moved without breaking its seal.",
    "I ask Iria where I said the spare archive key would be hidden.",
    "I ask what distraction would empty the west corridor for one minute.",
    "I ask whether the silver compass points normally inside the gallery.",
    "I ask Iria what she will do if the envoy returns before dawn.",
    "I ask which ally should receive the first page copied from the ledger.",
    "I ask Iria to repeat the safe signal we agreed to use at the west door.",
    "I ask Iria whether the midnight bell has already rung in the lower halls.",
    "I trace the conservatory passage behind the roses and ask where it ends.",
    "I ask which neutral courtier might be swayed by the ledger's contents.",
    "I tell Iria to watch the eastern stair while I approach the archive.",
    "I ask whether the regent's secretary carried anything sealed tonight.",
    "I test the west door with three soft taps and wait for her answer.",
    "I ask Iria to confirm the blue wax seal on the note just delivered.",
    "I ask how long the guards' midnight rotation leaves the corridor empty.",
    "I lift the third rose pedestal and check that the spare key remains hidden.",
    "I ask Iria which page of the ledger names the winter succession plot.",
    "I ask whether the silver compass still rests where I entrusted it.",
    "I ask her to describe the envoy's face when he first entered the gallery.",
    "I ask what sound the hidden archive latch makes when it gives way.",
    "I tell Iria I will copy a single page and leave the seal unbroken.",
    "I ask which ally should receive that copied page before dawn.",
    "I ask whether the ballroom crowd has thinned enough to cross unseen.",
    "I study the envoy's discarded note and compare its hand to the ledger.",
    "I ask Iria to recall the exact rule we set for trusting any message.",
    "I ask whether the regent has summoned anyone to the gallery tonight.",
    "I ask which mirror would warn us first if someone climbs the eastern stair.",
    "I ask Iria to repeat where I hid the spare archive key.",
    "I ask what distraction could clear the west corridor for a full minute.",
    "I tell Iria the three-tap signal still stands if we must part.",
    "I ask whether the archivists left any cipher beside the sealed ledger.",
    "I ask which servant can carry the copied page without drawing notice.",
    "I ask Iria how the regent first learned of the old archive's existence.",
    "I ask whether the clock's silenced chime was meant as a signal to someone.",
    "I ask her to name the courtier who changed their route after the envoy left.",
    "I tell Iria I trust only the blue-sealed messages from this point on.",
    "I ask whether the conservatory door can be barred from the inside.",
    "I ask what the regent fears most about the winter succession record.",
    "I ask Iria to keep the silver compass hidden until I send the safe signal.",
    "I ask which corridor the secretary used when he slipped away just now.",
    "I inspect the brass clock face again for the scratch the envoy left.",
    "I ask whether dawn light will reach the archive before the guards return.",
    "I ask Iria to confirm the order in which tonight's suspicious guests arrived.",
    "I ask what proof would turn a neutral courtier against the regent.",
    "I tell Iria to send the servants away before we open the latch.",
    "I ask whether anyone disturbed the third rose pedestal since I hid the key.",
    "I ask which page the regent would destroy first if he reached the ledger.",
    "I ask Iria to listen for three taps if I am delayed at the archive.",
    "I ask whether the eastern stair is finally clear of watchers.",
    "I ask her to recall the promise I made about returning before dawn.",
    "I ask which message tonight carried no blue seal and should be doubted.",
    "I tell Iria I will return the copied page to her by the west door.",
    "I ask whether the regent's envoy has reentered the palace gates.",
    "I ask Iria for the safe signal one last time before I leave the gallery.",
    "I ask which ally waits beyond the conservatory to carry our evidence.",
    "I tell Iria that if the bell rings twice, she must hide the compass and flee.",
    "I ask Iria what she will remember of this night when the regent is gone.",
)


@dataclass(frozen=True)
class StoryEvent:
    key: str
    definition_turn: int
    callback_turn: int
    term_groups: tuple[tuple[str, ...], ...]


STORY_EVENTS = (
    StoryEvent(
        key="before_dawn_promise",
        definition_turn=3,
        callback_turn=13,
        term_groups=(("promise", "promised"), ("return", "come back"), ("dawn",)),
    ),
    StoryEvent(
        key="silver_compass",
        definition_turn=21,
        callback_turn=31,
        term_groups=(("silver",), ("compass",), ("keep", "entrust", "hold")),
    ),
    StoryEvent(
        key="blue_seal_trust_rule",
        definition_turn=27,
        callback_turn=38,
        term_groups=(("blue",), ("wax",), ("seal",), ("trust", "message")),
    ),
    StoryEvent(
        key="key_hiding_place",
        definition_turn=34,
        callback_turn=45,
        term_groups=(("key",), ("third", "3rd"), ("rose",), ("pedestal",)),
    ),
    StoryEvent(
        key="three_tap_signal",
        definition_turn=40,
        callback_turn=50,
        term_groups=(("three", "3"), ("tap", "knock"), ("west",), ("door",)),
    ),
)

STRICT_WARNING_PREFIXES = {
    "critic": "critic skipped:",
    "memory_curation": "memory curation skipped:",
    "indexing": "memory indexing skipped:",
    "retrieval": "retrieval skipped:",
}
SECRET_KEY_FRAGMENTS = ("api_key", "apikey", "authorization", "password", "secret", "token")
NON_SECRET_PLACEHOLDERS = {"", "local", "replace_me"}
PROPER_NOUN_PATTERN = re.compile(r"\b[A-Z][a-z]{2,}\b")
KNOWN_PROPER_NOUNS = {
    "Iria",
    "Rose",
    "Gallery",
    "Winter",
    "Palace",
    "Regent",
    "Archivist",
}


class CheckpointError(AssertionError):
    pass


class HttpResponse(Protocol):
    def raise_for_status(self) -> None: ...

    def json(self) -> Any: ...


class HttpClient(Protocol):
    def get(self, url: str) -> HttpResponse: ...

    def post(self, url: str, *, json: Mapping[str, Any]) -> HttpResponse: ...

    def __enter__(self) -> HttpClient: ...

    def __exit__(self, *args: object) -> None: ...


@dataclass(frozen=True)
class EventAttribution:
    event_key: str
    query: str
    matching_memory_ids: tuple[str, ...]
    indexed_memory_ids: tuple[str, ...]
    selected_memory_ids: tuple[str, ...]
    selected_visibilities: tuple[str, ...]
    # Matching memories that were retrieved but not selected, with their overall
    # rank across the candidate set — the actionable "how far off" tuning signal.
    missed_memory_ids: tuple[str, ...] = ()
    missed_memory_ranks: tuple[int, ...] = ()


@dataclass(frozen=True)
class PersistenceInspection:
    persisted_turn_count: int
    persisted_memory_count: int
    canon_lore_count: int
    session_memory_count: int


Inspector = Callable[[str], PersistenceInspection]
EventInspector = Callable[[str, StoryEvent], EventAttribution]
ClientFactory = Callable[[], Any]


def resolve_turn_count(value: str | None, legacy_value: str | None = None) -> int:
    raw_value = value if value not in (None, "") else legacy_value
    if raw_value in (None, "", "0"):
        return DEFAULT_TURN_COUNT
    assert raw_value is not None
    try:
        turn_count = int(raw_value)
    except ValueError as exc:
        raise ValueError("LIVE_TURN_COUNT must be an integer from 5 through 100") from exc
    if not MIN_TURN_COUNT <= turn_count <= MAX_TURN_COUNT:
        raise ValueError("LIVE_TURN_COUNT must be an integer from 5 through 100")
    return turn_count


def conversation_messages(turn_count: int) -> tuple[str, ...]:
    if not MIN_TURN_COUNT <= turn_count <= MAX_TURN_COUNT:
        raise ValueError("turn_count must be from 5 through 100")
    return ROSE_GALLERY_MESSAGES[:turn_count]


def events_for_turn_count(turn_count: int) -> tuple[StoryEvent, ...]:
    return tuple(event for event in STORY_EVENTS if event.callback_turn <= turn_count)


def semantic_match(text: str, term_groups: Sequence[Sequence[str]]) -> bool:
    lowered = text.casefold()
    return all(any(term.casefold() in lowered for term in group) for group in term_groups)


def build_event_attribution(
    *,
    event: StoryEvent,
    query: str,
    memories: Sequence[MemoryEpisode],
    indexed_memory_ids: Sequence[str],
    selected: Sequence[Mapping[str, Any]],
    rejected: Sequence[Mapping[str, Any]] = (),
) -> EventAttribution:
    matching_ids = tuple(
        memory.id for memory in memories if semantic_match(memory.summary, event.term_groups)
    )
    matching_set = set(matching_ids)
    selected_memory_ids = tuple(
        str(item["id"])
        for item in selected
        if item.get("collection") == RagCollection.SESSION_MEMORY.value
        and str(item.get("id")) in matching_set
    )
    miss = summarize_retrieval_miss(
        expected_ids=matching_ids,
        selected=selected,
        rejected=rejected,
    )
    missed_memory_ranks = tuple(
        rank.overall_rank
        for rank in miss.ranks
        if not rank.selected and rank.overall_rank is not None
    )
    return EventAttribution(
        event_key=event.key,
        query=query,
        matching_memory_ids=matching_ids,
        indexed_memory_ids=tuple(
            memory_id for memory_id in matching_ids if memory_id in set(indexed_memory_ids)
        ),
        selected_memory_ids=selected_memory_ids,
        selected_visibilities=tuple(str(item.get("visibility")) for item in selected),
        missed_memory_ids=miss.missed_ids,
        missed_memory_ranks=missed_memory_ranks,
    )


def warning_counts(warnings: Sequence[Any]) -> dict[str, int]:
    counts = {category: 0 for category in STRICT_WARNING_PREFIXES}
    counts["other"] = 0
    for warning in warnings:
        text = str(warning)
        category = next(
            (
                name
                for name, prefix in STRICT_WARNING_PREFIXES.items()
                if text.startswith(prefix)
            ),
            "other",
        )
        counts[category] += 1
    return counts


def validate_warnings(
    warnings: Sequence[Any],
    *,
    strict: bool,
    turn_index: int,
) -> dict[str, int]:
    counts = warning_counts(warnings)
    structured_count = sum(counts[name] for name in STRICT_WARNING_PREFIXES)
    if strict and structured_count:
        raise CheckpointError(f"checkpoint warnings on turn {turn_index}: {list(warnings)}")
    return counts


def run_checkpoint(
    *,
    client_factory: ClientFactory,
    inspector: Inspector,
    event_inspector: EventInspector,
    expected_model: str,
    turn_count: int,
    fail_on_structured_warnings: bool,
    monotonic: Callable[[], float] = time.monotonic,
    progress_writer: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    messages = conversation_messages(turn_count)
    event_by_callback = {
        event.callback_turn: event for event in events_for_turn_count(turn_count)
    }
    attributions: dict[str, EventAttribution] = {}
    turns: list[dict[str, Any]] = []
    configuration = {
        "turn_count": turn_count,
        "expected_model": expected_model,
        "fail_on_structured_warnings": fail_on_structured_warnings,
    }

    def write_progress() -> None:
        if progress_writer is None:
            return
        progress_writer(
            {
                "status": "in_progress",
                "configuration": dict(configuration),
                "turns": list(turns),
                "event_inspections": {
                    key: dict(attribution.__dict__)
                    for key, attribution in attributions.items()
                },
            }
        )

    with client_factory() as client:
        runtime_status = _get_json(client, "/runtime/status")
        _require(
            runtime_status.get("content_catalog_available") is True,
            "content catalog unavailable",
        )
        _require(
            runtime_status.get("local_provider_configured") is True,
            "local provider unavailable",
        )
        _require(runtime_status.get("retrieval_configured") is True, "retrieval unavailable")
        catalog = _get_json(client, "/content/catalog")
        world_ids = {
            world.get("id")
            for world in catalog.get("worlds", [])
            if isinstance(world, dict)
        }
        _require("demo_world" in world_ids, "demo_world missing from content catalog")
        session = _post_json(
            client,
            "/sessions",
            {
                "world_id": "demo_world",
                "scene_id": "rose-gallery",
                "active_persona_id": "archivist",
                "player_name": "Live Checkpoint",
            },
        )
        session_id = str(session["session_id"])

        for turn_index, prompt in enumerate(messages, start=1):
            event = event_by_callback.get(turn_index)
            if event is not None:
                attribution = event_inspector(session_id, event)
                _validate_attribution(attribution, strict=fail_on_structured_warnings)
                attributions[event.key] = attribution
                write_progress()

            started = monotonic()
            turn = _post_json(
                client,
                f"/sessions/{session_id}/turns",
                {"message": prompt, "request_cloud": False},
            )
            duration_seconds = monotonic() - started
            route = turn.get("route", {})
            warnings = turn.get("warnings", [])
            raw_response = turn.get("text")
            _require(
                isinstance(raw_response, str) and bool(raw_response.strip()),
                f"empty actor text on turn {turn_index}",
            )
            response = cast(str, raw_response)
            outcome = str(turn.get("outcome") or "success")
            _require(
                route.get("provider") == "local",
                f"unexpected route provider on turn {turn_index}: {route}",
            )
            _require(
                route.get("model") == expected_model,
                f"unexpected route model on turn {turn_index}: {route}",
            )
            counts = validate_warnings(
                warnings,
                strict=fail_on_structured_warnings,
                turn_index=turn_index,
            )
            turns.append(
                {
                    "turn_index": turn_index,
                    "prompt": prompt,
                    "response": response,
                    "outcome": outcome,
                    "response_chars": len(response),
                    "duration_seconds": round(duration_seconds, 3),
                    "route": route,
                    "finish_reason": turn.get("finish_reason"),
                    "memory_written": bool(turn.get("memory_written")),
                    "stage_timings": dict(turn.get("stage_timings") or {}),
                    "retrieval": turn.get("retrieval"),
                    "warning_counts": counts,
                    "warnings": list(warnings),
                }
            )
            write_progress()

        lookup = _get_json(client, f"/sessions/{session_id}")

    # Controlled-failure turns are persisted but excluded from the recent-dialogue
    # view, so the expected window is the last 8 *successful* prompts.
    successful_prompts = [turn["prompt"] for turn in turns if turn["outcome"] == "success"]
    expected_recent = list(successful_prompts[-min(len(successful_prompts), 8) :])
    actual_recent = [
        turn.get("user_message")
        for turn in lookup.get("recent_turns", [])
        if isinstance(turn, dict)
    ]
    _require(
        actual_recent == expected_recent,
        "session lookup did not return expected recent conversation",
    )

    inspection = inspector(session_id)
    _require(
        inspection.persisted_turn_count == turn_count,
        f"persisted-turn count mismatch: {inspection.persisted_turn_count} != {turn_count}",
    )
    _require(inspection.canon_lore_count >= 1, "Qdrant contains no ingested canon lore")
    _require(
        inspection.persisted_memory_count == inspection.session_memory_count,
        "SQLite/Qdrant session-memory count mismatch: "
        f"{inspection.persisted_memory_count} != {inspection.session_memory_count}",
    )

    for event in events_for_turn_count(turn_count):
        attribution = attributions[event.key]
        callback_response = turns[event.callback_turn - 1]["response"]
        turns[event.callback_turn - 1]["event_callback"] = event.key
        turns[event.callback_turn - 1]["callback_recalled"] = semantic_match(
            callback_response,
            event.term_groups,
        )

    total_warning_counts = {
        category: sum(turn["warning_counts"][category] for turn in turns)
        for category in (*STRICT_WARNING_PREFIXES, "other")
    }
    durations = [float(turn["duration_seconds"]) for turn in turns]
    stage_samples: dict[str, list[float]] = {}
    for turn in turns:
        for stage, seconds in turn["stage_timings"].items():
            stage_samples.setdefault(str(stage), []).append(float(seconds))
    stage_latency_means = {
        stage: round(statistics.mean(samples), 3)
        for stage, samples in sorted(stage_samples.items())
    }
    finish_reasons = Counter(
        str(turn["finish_reason"] or "unknown")
        for turn in turns
    )
    event_payloads = [
        {
            **attribution.__dict__,
            "extracted": bool(attribution.matching_memory_ids),
            "indexed": (
                bool(attribution.matching_memory_ids)
                and attribution.indexed_memory_ids == attribution.matching_memory_ids
            ),
            "selected": bool(attribution.selected_memory_ids),
            "recalled": bool(turns[event.callback_turn - 1]["callback_recalled"]),
        }
        for event in events_for_turn_count(turn_count)
        for attribution in (attributions[event.key],)
    ]
    return {
        "status": "pass",
        "configuration": dict(configuration),
        "runtime_status": runtime_status,
        "session": session,
        "turns": turns,
        "events": event_payloads,
        "persisted": {
            "turn_count": inspection.persisted_turn_count,
            "memory_count": inspection.persisted_memory_count,
        },
        "qdrant": {
            "canon_lore_count": inspection.canon_lore_count,
            "session_memory_count": inspection.session_memory_count,
        },
        "lookup_recent_turn_count": len(actual_recent),
        "warning_counts": total_warning_counts,
        "quality_metrics": {
            # Report-only: fail-closed turns are model quality, not an
            # infrastructure failure; the operator reads the number.
            "controlled_failure_count": sum(
                1 for turn in turns if turn["outcome"] != "success"
            ),
            "memory_extraction_misses": sum(not event["extracted"] for event in event_payloads),
            "callback_recall_misses": sum(not event["recalled"] for event in event_payloads),
            "retrieval_selection_misses": sum(
                bool(event["missed_memory_ids"]) for event in event_payloads
            ),
            "retrieval_miss_ranks": sorted(
                rank for event in event_payloads for rank in event["missed_memory_ranks"]
            ),
            "response_chars": [turn["response_chars"] for turn in turns],
            "novel_proper_noun_candidates": sorted(
                {
                    candidate
                    for turn in turns
                    for candidate in PROPER_NOUN_PATTERN.findall(turn["response"])
                    if candidate not in KNOWN_PROPER_NOUNS
                }
            ),
            "latency": {
                "total_seconds": round(sum(durations), 3),
                "p50_seconds": round(statistics.median(durations), 3),
                "p95_seconds": round(_percentile(durations, 0.95), 3),
            },
            "stage_latency_means": stage_latency_means,
            "finish_reason_distribution": dict(sorted(finish_reasons.items())),
        },
    }


def _validate_attribution(attribution: EventAttribution, *, strict: bool) -> None:
    if strict:
        _require(
            bool(attribution.matching_memory_ids),
            f"event {attribution.event_key} has no persisted matching memory",
        )
    if attribution.matching_memory_ids:
        _require(
            attribution.indexed_memory_ids == attribution.matching_memory_ids,
            f"event {attribution.event_key} has extracted memories missing from Qdrant",
        )
        if strict:
            _require(
                bool(attribution.selected_memory_ids),
                f"event {attribution.event_key} was not selected by callback retrieval",
            )
    _require(
        all(
            visibility == Visibility.PLAYER.value
            for visibility in attribution.selected_visibilities
        ),
        f"event {attribution.event_key} selected hidden memory",
    )


def inspect_live_state(
    *,
    database_path: Path,
    settings: Settings,
    session_id: str,
) -> PersistenceInspection:
    with sqlite3.connect(database_path) as connection:
        persisted_turn_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM turns WHERE session_id = ?",
                (session_id,),
            ).fetchone()[0]
        )
        persisted_memory_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM memory_episodes WHERE session_id = ?",
                (session_id,),
            ).fetchone()[0]
        )
    vector_store = QdrantVectorStore(url=settings.qdrant_url)
    return PersistenceInspection(
        persisted_turn_count=persisted_turn_count,
        persisted_memory_count=persisted_memory_count,
        canon_lore_count=_qdrant_count(vector_store, RagCollection.CANON_LORE),
        session_memory_count=_qdrant_count(
            vector_store,
            RagCollection.SESSION_MEMORY,
            session_id=session_id,
        ),
    )


def inspect_story_event(
    *,
    database_path: Path,
    settings: Settings,
    session_id: str,
    event: StoryEvent,
) -> EventAttribution:
    connection = connect_sqlite(database_path)
    try:
        memory_repository = SQLiteMemoryRepository(connection)
        memories = memory_repository.list_memories_for_session(session_id)
        recent_turns = SQLiteTurnRepository(connection).list_recent_turns(
            session_id,
            settings.recent_dialogue_turns,
        )
    finally:
        connection.close()
    loader = build_file_loader(settings.content_root)
    query = build_retrieval_query(
        user_message=ROSE_GALLERY_MESSAGES[event.callback_turn - 1],
        scene=loader.load_scene("rose-gallery"),
        persona=loader.load_persona("archivist"),
        recent_turns=recent_turns,
    )
    retrieval = build_actor_context_retriever(settings).retrieve_for_actor_with_diagnostics(
        query=query,
        lexical_query=ROSE_GALLERY_MESSAGES[event.callback_turn - 1],
        world_id="demo_world",
        session_id=session_id,
        persona_id="archivist",
        scene_id="rose-gallery",
        top_k=settings.rag_default_top_k,
    )
    diagnostics = retrieval.diagnostics.model_dump(mode="json")
    indexed_ids = _qdrant_memory_ids(
        QdrantVectorStore(url=settings.qdrant_url),
        session_id=session_id,
    )
    return build_event_attribution(
        event=event,
        query=query,
        memories=memories,
        indexed_memory_ids=indexed_ids,
        selected=diagnostics["selected"],
        rejected=diagnostics["rejected"],
    )


def redact_secrets(value: Any, *, secret_values: Sequence[str] = ()) -> Any:
    secrets = tuple(secret for secret in secret_values if secret)
    if isinstance(value, dict):
        return {
            key: (
                "***"
                if any(fragment in key.lower() for fragment in SECRET_KEY_FRAGMENTS)
                else redact_secrets(item, secret_values=secrets)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_secrets(item, secret_values=secrets) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_secrets(item, secret_values=secrets) for item in value)
    if isinstance(value, str):
        redacted = value
        for secret in secrets:
            redacted = redacted.replace(secret, "***")
        return redacted
    return value


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def write_reports(
    summary: Mapping[str, Any],
    *,
    json_path: Path,
    markdown_path: Path,
    secret_values: Sequence[str] = (),
) -> None:
    safe_summary = redact_secrets(dict(summary), secret_values=secret_values)
    write_json_atomic(json_path, safe_summary)
    lines = [
        "## Conversation Checkpoint",
        "",
        f"- status: {safe_summary['status']}",
        f"- turn_count: {safe_summary['configuration']['turn_count']}",
        f"- persisted_turn_count: {safe_summary['persisted']['turn_count']}",
        f"- persisted_memory_count: {safe_summary['persisted']['memory_count']}",
        f"- session_memory_count: {safe_summary['qdrant']['session_memory_count']}",
        f"- warning_counts: `{json.dumps(safe_summary['warning_counts'], sort_keys=True)}`",
        f"- quality_metrics: `{json.dumps(safe_summary['quality_metrics'], sort_keys=True)}`",
        "",
        "### Event Attribution",
        "",
        "```json",
        json.dumps(safe_summary["events"], indent=2, sort_keys=True),
        "```",
        "",
        "### Turns",
        "",
    ]
    for turn in safe_summary["turns"]:
        lines.extend(
            [
                f"#### Turn {turn['turn_index']}",
                "",
                f"- duration_seconds: {turn['duration_seconds']}",
                f"- stage_timings: `{json.dumps(turn['stage_timings'], sort_keys=True)}`",
                f"- finish_reason: {turn['finish_reason']}",
                f"- route: `{json.dumps(turn['route'], sort_keys=True)}`",
                f"- memory_written: {str(turn['memory_written']).lower()}",
                f"- warnings: `{json.dumps(turn['warnings'])}`",
                "",
                f"**Prompt:** {turn['prompt']}",
                "",
                f"**Response:** {turn['response']}",
                "",
            ]
        )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def _qdrant_count(
    vector_store: QdrantVectorStore,
    collection: RagCollection,
    *,
    session_id: str | None = None,
) -> int:
    if not vector_store.client.collection_exists(collection_name=collection.value):
        return 0
    count_filter = None
    if session_id is not None:
        from qdrant_client.http import models

        count_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="session_id",
                    match=models.MatchValue(value=session_id),
                )
            ]
        )
    result = vector_store.client.count(
        collection_name=collection.value,
        count_filter=count_filter,
        exact=True,
    )
    return int(result.count)


def _qdrant_memory_ids(
    vector_store: QdrantVectorStore,
    *,
    session_id: str,
) -> tuple[str, ...]:
    if not vector_store.client.collection_exists(
        collection_name=RagCollection.SESSION_MEMORY.value
    ):
        return ()
    from qdrant_client.http import models

    points, _ = vector_store.client.scroll(
        collection_name=RagCollection.SESSION_MEMORY.value,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="session_id",
                    match=models.MatchValue(value=session_id),
                )
            ]
        ),
        limit=1000,
        with_payload=True,
        with_vectors=False,
    )
    return tuple(
        str(point.payload["id"])
        for point in points
        if point.payload and isinstance(point.payload.get("id"), str)
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[index]


def _get_json(client: HttpClient, path: str) -> dict[str, Any]:
    response = client.get(path)
    response.raise_for_status()
    payload = response.json()
    _require(isinstance(payload, dict), f"expected JSON object from {path}")
    return cast(dict[str, Any], payload)


def _post_json(client: HttpClient, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    response = client.post(path, json=payload)
    response.raise_for_status()
    result = response.json()
    _require(isinstance(result, dict), f"expected JSON object from {path}")
    return cast(dict[str, Any], result)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckpointError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the live-stack conversation checkpoint.")
    parser.add_argument("--api-base-url", required=True)
    parser.add_argument("--expected-model", required=True)
    parser.add_argument("--database-path", type=Path, required=True)
    parser.add_argument("--turn-count", type=int, required=True)
    parser.add_argument("--fail-on-structured-warnings", choices=("0", "1"), required=True)
    parser.add_argument("--json-report", type=Path, required=True)
    parser.add_argument("--markdown-report", type=Path, required=True)
    args = parser.parse_args()

    settings = Settings(database_path=str(args.database_path))
    secret_values = tuple(
        value
        for value in (settings.local_llm_api_key, settings.cloud_llm_api_key)
        if value.strip().lower() not in NON_SECRET_PLACEHOLDERS
    )
    # Must exceed provider timeout x (1 + retries) or the client gives up on
    # turns the provider would still have salvaged (late 50-turn sessions
    # legitimately reach ~120s per LLM call).
    http_timeout = float(os.environ.get("LIVE_HTTP_TIMEOUT_SECONDS", "420"))
    summary = run_checkpoint(
        client_factory=lambda: httpx.Client(base_url=args.api_base_url, timeout=http_timeout),
        inspector=lambda session_id: inspect_live_state(
            database_path=args.database_path,
            settings=settings,
            session_id=session_id,
        ),
        event_inspector=lambda session_id, event: inspect_story_event(
            database_path=args.database_path,
            settings=settings,
            session_id=session_id,
            event=event,
        ),
        expected_model=args.expected_model,
        turn_count=args.turn_count,
        fail_on_structured_warnings=args.fail_on_structured_warnings == "1",
        progress_writer=lambda snapshot: write_json_atomic(
            args.json_report,
            redact_secrets(dict(snapshot), secret_values=secret_values),
        ),
    )
    write_reports(
        summary,
        json_path=args.json_report,
        markdown_path=args.markdown_report,
        secret_values=secret_values,
    )
    print(json.dumps(redact_secrets(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
