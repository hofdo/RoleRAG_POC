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
from app.domain import MemoryEpisode, StoredTurn, Visibility
from app.memory.consolidation import CONSOLIDATED_TAG, SUMMARY_TAG
from app.persistence import SQLiteMemoryRepository, SQLiteTurnRepository, connect_sqlite
from app.rag.models import RagCollection
from app.rag.retriever import build_retrieval_query
from app.rag.vector_store import _SENTINEL_PAYLOAD_KEY, QdrantVectorStore

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
    "I tell Iria I have hidden a folded note inside the hollow bookend on the third archive shelf.",
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
    "I press an amber ring into Iria's palm and tell her to wear it as a sign of my word.",
    "I test the west door with three soft taps and wait for her answer.",
    "I ask Iria to confirm the blue wax seal on the note just delivered.",
    "I ask how long the guards' midnight rotation leaves the corridor empty.",
    "I lift the third rose pedestal and check that the spare key remains hidden.",
    "I ask Iria which page of the ledger names the winter succession plot.",
    "I ask whether the silver compass still rests where I entrusted it.",
    "I ask her to describe the envoy's face when he first entered the gallery.",
    "I ask what sound the hidden archive latch makes when it gives way.",
    "I tell Iria I will copy a single page and leave the seal unbroken.",
    "I ask Iria whether she still wears the token I gave her that night.",
    "I ask whether the ballroom crowd has thinned enough to cross unseen.",
    "I study the envoy's discarded note and compare its hand to the ledger.",
    "I ask Iria to recall the exact rule we set for trusting any message.",
    "I ask whether the regent has summoned anyone to the gallery tonight.",
    "I tell Iria that if we are separated, I will wait for her at the old north stair.",
    "I ask Iria to repeat where I hid the spare archive key.",
    "I ask what distraction could clear the west corridor for a full minute.",
    "I tell Iria the three-tap signal still stands if we must part.",
    "I ask whether the archivists left any cipher beside the sealed ledger.",
    "I ask which servant can carry the copied page without drawing notice.",
    "I ask Iria how the regent first learned of the old archive's existence.",
    "I ask whether the clock's silenced chime was meant as a signal to someone.",
    "I ask her to name the courtier who changed their route after the envoy left.",
    "I tell Iria I trust only the blue-sealed messages from this point on.",
    "I ask Iria whether she remembers where I said I would wait if we were separated.",
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
    "I ask Iria whether she remembers where I hid that folded note.",
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
    # Long-gap probe: fact stated in the opening act, callback near the very end of a
    # 100-turn run — the "old durable fact still retrievable after 100 turns" case.
    StoryEvent(
        key="hollow_bookend_note",
        definition_turn=17,
        callback_turn=95,
        term_groups=(("bookend",), ("note",), ("shelf", "shelves")),
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
    # Late-recall probes (turns 51-100 were previously filler; a 100-turn run asserted
    # nothing past turn 50). Both setup and callback fall after turn 50.
    StoryEvent(
        key="amber_ring_token",
        definition_turn=55,
        callback_turn=65,
        # "symbol"/"promise"/"pledge"/"pact": observed extractor paraphrase of the scripted
        # "wear it as a sign of my word" (docs/22 P2.2 first-run finding, 2026-07-12) --
        # the memory read "symbol of a pact/promise" and the probe false-negatived.
        term_groups=(
            ("amber",),
            ("ring",),
            ("wear", "sign", "token", "symbol", "promise", "pledge", "pact"),
        ),
    ),
    StoryEvent(
        key="north_stair_rendezvous",
        definition_turn=70,
        callback_turn=80,
        # "rendezvous": same paraphrase-risk class as amber_ring_token's finding.
        term_groups=(("north",), ("stair", "stairs"), ("wait", "meet", "rendezvous")),
    ),
)

STRICT_WARNING_PREFIXES = {
    "critic": "critic skipped:",
    "memory_curation": "memory curation skipped:",
    "indexing": "memory indexing skipped:",
    "retrieval": "retrieval skipped:",
}
# Report-only (#69 evidence): the context-accounting warnings from
# app.orchestration.context_budget.context_preflight_warning /
# .context_usage_warning. Both literally start with "context preflight: " --
# context_usage_warning's post-hoc string reuses that prefix by name (see its
# docstring) -- so they are told apart by what follows it, not by the shared
# lead-in. Deliberately kept out of STRICT_WARNING_PREFIXES: these never gate
# fail_on_structured_warnings, and are zero on any run with
# MODEL_CONTEXT_WINDOW_TOKENS unset (the default), since neither function ever
# fires then.
CONTEXT_WARNING_PREFIXES = {
    "preflight": "context preflight: estimated",
    "actual": "context preflight: actual prompt_tokens",
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
    # Excludes consolidation-marked (CONSOLIDATED_TAG) rows (docs/22 P2, fixed
    # 2026-07-11) so this stays comparable to ``session_memory_count``: consolidation
    # is non-destructive in SQLite (the original rows stay, tagged, forever) but
    # unindexes them from Qdrant, so an unfiltered SQLite count would permanently
    # diverge from the Qdrant-indexed count starting at the first roll-up.
    persisted_memory_count: int
    canon_lore_count: int
    session_memory_count: int
    # Report-only (P2.2 evidence): SQLite rows tagged CONSOLIDATED_TAG / SUMMARY_TAG,
    # confirming consolidation actually fired during the run and how much it rolled
    # up. Counted separately from persisted_memory_count above (which excludes
    # CONSOLIDATED_TAG rows for the SQLite/Qdrant parity check) so this stays purely
    # additive evidence. Zero on any run with consolidation off
    # (memory_consolidation_threshold <= 0, the default) -- select_consolidatable
    # never marks anything then.
    consolidated_memory_count: int = 0
    consolidation_summary_count: int = 0


Inspector = Callable[[str], PersistenceInspection]
# Third argument (#80, docs/26 §8 Q4): the definition turn's ACTUAL persisted DB
# ordinal at the moment it succeeded, tracked by run_checkpoint across any
# consumed definition-turn retries -- None when that turn never succeeded (or
# the caller doesn't track it), in which case implementations fall back to the
# StoryEvent's raw scripted definition_turn step.
EventInspector = Callable[[str, StoryEvent, int | None], EventAttribution]
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
    definition_turn_id: int | None = None,
) -> EventAttribution:
    """Build attribution for one probe event.

    A memory counts toward ``matching_memory_ids`` if it phrasing-matches
    (``semantic_match``, unchanged) OR its ``source_turn_id`` equals
    ``definition_turn_id`` -- provenance-OR-phrasing, never provenance-only
    (docs/26 §4's correction to the paraphrase-fragility measurement gap, #76).
    ``definition_turn_id=None`` (the default) disables the provenance leg
    entirely, so callers that never resolve it -- and every pre-#76 caller --
    get exactly the old phrasing-only behavior.

    Multi-memory-turn caveat (docs/26 §3.1, §4): a definition turn that produces
    several memory episodes links ALL of them to the same source_turn_id. If NONE
    of them phrasing-match the probe's term groups, provenance alone cannot tell
    which episode is the probed fact -- every episode from that turn still counts
    as "matching". This is a deliberate, documented residual (docs/26 explicitly
    rejects provenance-ONLY attribution for exactly this reason, but ships
    OR-with-phrasing anyway as the bounded, auditable trade against the worse
    paraphrase-fragility false negative); see
    test_build_event_attribution_provenance_over_attributes_on_multi_memory_definition_turn.
    """
    matching_ids = tuple(
        memory.id
        for memory in memories
        if semantic_match(memory.summary, event.term_groups)
        or (definition_turn_id is not None and memory.source_turn_id == definition_turn_id)
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


def context_warning_counts(warnings: Sequence[Any]) -> dict[str, int]:
    """Report-only counterpart to warning_counts: counts the #69 context-accounting
    warnings (CONTEXT_WARNING_PREFIXES) by category. Unlike warning_counts, an
    unmatched warning increments nothing -- there is no "other" bucket, since this
    is purely additive evidence and never feeds validate_warnings/
    fail_on_structured_warnings."""
    counts = {category: 0 for category in CONTEXT_WARNING_PREFIXES}
    for warning in warnings:
        text = str(warning)
        category = next(
            (
                name
                for name, prefix in CONTEXT_WARNING_PREFIXES.items()
                if text.startswith(prefix)
            ),
            None,
        )
        if category is not None:
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
    definition_retries: int = 0,
    monotonic: Callable[[], float] = time.monotonic,
    progress_writer: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the scripted Rose Gallery checkpoint.

    ``definition_retries`` (#80, docs/26 §6 Stage 5 / §8 Q4): a harness-local
    scenario-semantics parameter -- deliberately NOT an app.config.Settings field
    (Q4: it scripts the harness's own message sequence, never engine behavior, so
    it is out of scope for the Settings/.env.example pairing test). At the
    default of 0 this function is byte-identical to the pre-#80 code path: the
    retry loop below never executes, so every probe's definition turn either
    succeeds or fails exactly as it always has, and #75's fail-fast fires
    unchanged.

    When > 0: a probe's DEFINITION turn ending in a non-success outcome gets the
    SAME scripted message re-sent, up to ``definition_retries`` times -- modeling
    a real player's natural retry on an errored turn (Q4), not a re-roll of the
    run. If a retry succeeds, the run continues; if the budget is exhausted with
    no success, the run fails fast naming both the original cause and the
    consumed retry (see ``_validate_definition_turn_outcome``).

    Each consumed retry inserts an EXTRA persisted DB turn, so scripted step N no
    longer reliably lands on DB ``turn_index`` N once any retry has fired earlier
    in the run (docs/26 §8 Q4's "scenario semantics" hazard). Two bookkeeping
    structures exist purely to keep every turn-number-keyed computation correct
    under that offset, instead of assuming step == index:

    - ``turn_by_step``: the CURRENT (latest-attempt) turn record for a given
      scripted step, keyed by the step itself -- never by position in ``turns``.
      ``turns`` stays the full chronological attempt log (it gains one extra,
      clearly labeled ``is_definition_retry``/``retry_attempt`` entry per
      consumed retry); indexing into it positionally by scripted step is exactly
      the hazard ``turn_by_step`` avoids. ``event_by_callback`` and
      ``event_by_definition_turn`` stay keyed by scripted step throughout, since
      that is what a StoryEvent's ``definition_turn``/``callback_turn`` mean.
    - ``definition_turn_db_index``: the ACTUAL persisted ``turns.turn_index`` at
      the moment an event's definition turn succeeded (whether on the first
      attempt or a consumed retry) -- threaded into ``event_inspector`` so #76's
      provenance attribution resolves to the SUCCESSFUL attempt's DB id, never
      the failed original's id and never a naively-assumed step==index value.
    """
    messages = conversation_messages(turn_count)
    event_by_callback = {
        event.callback_turn: event for event in events_for_turn_count(turn_count)
    }
    # #74 fail-fast: looked up once per turn below so a probe's DEFINITION turn
    # ending in a non-success outcome is named immediately instead of surfacing as
    # a misleading "no persisted matching memory" at the callback turn, 10+ turns
    # later (_validate_attribution, which only fires in strict mode).
    event_by_definition_turn = {
        event.definition_turn: event for event in events_for_turn_count(turn_count)
    }
    attributions: dict[str, EventAttribution] = {}
    turns: list[dict[str, Any]] = []
    turn_by_step: dict[int, dict[str, Any]] = {}
    definition_turn_db_index: dict[str, int] = {}
    definition_retries_used: dict[str, int] = {
        event.key: 0 for event in event_by_definition_turn.values()
    }
    persisted_turn_ordinal = 0
    configuration = {
        "turn_count": turn_count,
        "expected_model": expected_model,
        "fail_on_structured_warnings": fail_on_structured_warnings,
        "definition_retries_allowed": definition_retries,
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

        def send_turn(turn_index: int, prompt: str, *, retry_attempt: int) -> dict[str, Any]:
            nonlocal persisted_turn_ordinal
            label = (
                f"turn {turn_index}"
                if retry_attempt == 0
                else f"turn {turn_index} (definition retry {retry_attempt})"
            )
            started = monotonic()
            turn = _post_json(
                client,
                f"/sessions/{session_id}/turns",
                {"message": prompt},
            )
            duration_seconds = monotonic() - started
            # Every POST persists a turn regardless of outcome (StoredTurn rows
            # are SUCCESS or CONTROLLED_FAILURE, never absent), so this ordinal
            # tracks the real DB turn_index of the row this call just created.
            persisted_turn_ordinal += 1
            route = turn.get("route", {})
            warnings = turn.get("warnings", [])
            raw_response = turn.get("text")
            _require(
                isinstance(raw_response, str) and bool(raw_response.strip()),
                f"empty actor text on {label}",
            )
            response = cast(str, raw_response)
            outcome = str(turn.get("outcome") or "success")
            _require(
                route.get("provider") == "local",
                f"unexpected route provider on {label}: {route}",
            )
            _require(
                route.get("model") == expected_model,
                f"unexpected route model on {label}: {route}",
            )
            counts = validate_warnings(
                warnings,
                strict=fail_on_structured_warnings,
                turn_index=turn_index,
            )
            turn_record = {
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
                # Lane A canon-pinning evidence (docs/26 §3.3, #78): already
                # returned by CreateTurnResponse on every turn but not
                # previously surfaced here -- needed for docs/25 Phase E's
                # "never drops to 0 once established" read (#80).
                "standing_facts_count": turn.get("standing_facts_count"),
                "standing_facts_chars": turn.get("standing_facts_chars"),
                "warning_counts": counts,
                "warnings": list(warnings),
                # #80 labeling (docs/26 §8 Q4): a consumed retry is a real extra
                # persisted DB row, auditable here rather than silently folded
                # into the original entry -- "scenario variant, not a re-roll".
                "is_definition_retry": retry_attempt > 0,
                "retry_attempt": retry_attempt,
            }
            turns.append(turn_record)
            return turn_record

        for turn_index, prompt in enumerate(messages, start=1):
            event = event_by_callback.get(turn_index)
            if event is not None:
                attribution = event_inspector(
                    session_id, event, definition_turn_db_index.get(event.key)
                )
                _validate_attribution(attribution, strict=fail_on_structured_warnings)
                attributions[event.key] = attribution
                write_progress()

            turn_record = send_turn(turn_index, prompt, retry_attempt=0)
            turn_by_step[turn_index] = turn_record
            write_progress()

            definition_event = event_by_definition_turn.get(turn_index)
            original_outcome = turn_record["outcome"]
            retries_used = 0
            if definition_event is not None:
                while (
                    turn_record["outcome"] != "success" and retries_used < definition_retries
                ):
                    retries_used += 1
                    turn_record = send_turn(turn_index, prompt, retry_attempt=retries_used)
                    turn_by_step[turn_index] = turn_record
                    write_progress()
                if retries_used:
                    definition_retries_used[definition_event.key] = retries_used
                if turn_record["outcome"] == "success":
                    definition_turn_db_index[definition_event.key] = persisted_turn_ordinal

            _validate_definition_turn_outcome(
                turn_index=turn_index,
                outcome=turn_record["outcome"],
                event_by_definition_turn=event_by_definition_turn,
                strict=fail_on_structured_warnings,
                original_outcome=original_outcome,
                retries_used=retries_used,
            )

        lookup = _get_json(client, f"/sessions/{session_id}")

    # Controlled-failure turns are persisted but excluded from the recent-dialogue
    # view, so the expected window is the last 8 *successful* prompts. This
    # already tolerates consumed retries unmodified: a retried step contributes
    # its FAILED attempt (outcome != success, filtered out here) and, if the
    # retry succeeded, exactly one "success" entry with the same prompt text --
    # matching what the real recent-turns view shows.
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
    # #80: each consumed retry is one real extra persisted DB row, so the
    # scripted turn_count alone under-counts once any retry has fired.
    total_definition_retries_consumed = sum(definition_retries_used.values())
    expected_persisted_turn_count = turn_count + total_definition_retries_consumed
    _require(
        inspection.persisted_turn_count == expected_persisted_turn_count,
        f"persisted-turn count mismatch: {inspection.persisted_turn_count} != "
        f"{expected_persisted_turn_count} (turn_count={turn_count}, "
        f"definition_retries_consumed={total_definition_retries_consumed})",
    )
    _require(inspection.canon_lore_count >= 1, "Qdrant contains no ingested canon lore")
    _require(
        inspection.persisted_memory_count == inspection.session_memory_count,
        "SQLite/Qdrant session-memory count mismatch: "
        f"{inspection.persisted_memory_count} != {inspection.session_memory_count}",
    )

    for event in events_for_turn_count(turn_count):
        attribution = attributions[event.key]
        # Step-keyed, not positional (#80): callback turns are never themselves
        # retried, so turn_by_step[event.callback_turn] is unambiguous even after
        # an earlier definition-turn retry has made turns[event.callback_turn - 1]
        # point at the wrong list position.
        callback_turn_record = turn_by_step[event.callback_turn]
        callback_turn_record["event_callback"] = event.key
        callback_turn_record["callback_recalled"] = semantic_match(
            callback_turn_record["response"],
            event.term_groups,
        )

    total_warning_counts = {
        category: sum(turn["warning_counts"][category] for turn in turns)
        for category in (*STRICT_WARNING_PREFIXES, "other")
    }
    # Report-only (#69 evidence): aggregated across the same per-turn "warnings"
    # already collected above (turn.get("warnings", []) from each turn's API
    # response, the source total_warning_counts reads too) -- kept separate from
    # warning_counts/STRICT_WARNING_PREFIXES so it never affects the
    # fail_on_structured_warnings gate.
    context_warning_totals = context_warning_counts(
        [warning for turn in turns for warning in turn["warnings"]]
    )
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
            "recalled": bool(turn_by_step[event.callback_turn]["callback_recalled"]),
            "definition_retries_used": definition_retries_used[event.key],
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
            "consolidated_memory_count": inspection.consolidated_memory_count,
            "consolidation_summary_count": inspection.consolidation_summary_count,
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
            # Report-only (#69 evidence): 0 on any run with MODEL_CONTEXT_WINDOW_TOKENS
            # unset (the default) -- see CONTEXT_WARNING_PREFIXES.
            "context_preflight_warning_count": context_warning_totals["preflight"],
            "context_actual_warning_count": context_warning_totals["actual"],
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
            # #80 (docs/26 §6 Stage 5 / §8 Q4): raw facts only -- the naive
            # independence multiplication was rejected, so the harness does not
            # compute a survival-rate estimate here. The owner derives the
            # measured delta from definition_retries_used/outcome across >= 2
            # live runs, per Stage 5.
            "definition_retries_allowed": definition_retries,
            "definition_retries_used": dict(definition_retries_used),
        },
    }


def _validate_definition_turn_outcome(
    *,
    turn_index: int,
    outcome: str,
    event_by_definition_turn: Mapping[int, StoryEvent],
    strict: bool,
    original_outcome: str | None = None,
    retries_used: int = 0,
) -> None:
    """#74 precision fix, not a loosening: when a probe's DEFINITION turn ends in a
    non-success outcome (invariant #4 fail-closed -- e.g. a critic rejection), the
    probed fact never entered the world, so it is vacuous to keep running the probe
    forward to its callback turn. Fail fast here, naming the real cause, instead of
    marching on and reporting a misleading "no persisted matching memory" at the
    callback turn, 10+ turns later (docs/25 Phase D run 4).

    Gated on ``strict`` because the later check this replaces
    (``_validate_attribution``'s ``matching_memory_ids`` requirement) only ever
    raised in strict mode -- a report-only (non-strict) run never failed on this
    class of miss before, and still does not; this only relocates and renames an
    existing strict-mode failure to the turn that actually caused it.

    #80 (docs/26 §8 Q4): ``outcome`` is the FINAL outcome after any consumed
    definition-turn retries -- if a retry succeeded, ``outcome == "success"`` and
    this still returns without raising (the run continues, per Q4's "retry
    succeeds" case). ``retries_used == 0`` (the default, and what every
    ``definition_retries=0`` call passes) reproduces the original single-cause
    message byte-for-byte. Only when a consumed retry ALSO ends non-success
    (budget exhausted) does the message additionally name the original cause and
    how many retries were spent, per Q4's "budget exhausted" case.
    """
    if not strict:
        return
    event = event_by_definition_turn.get(turn_index)
    if event is None or outcome == "success":
        return
    if retries_used:
        raise CheckpointError(
            f"definition turn {turn_index} for event {event.key} ended in "
            f"{original_outcome}; {retries_used} definition-turn retry(ies) consumed "
            f"and still ended in {outcome}: fact never entered the world — probe vacuous"
        )
    raise CheckpointError(
        f"definition turn {turn_index} for event {event.key} ended in {outcome}: "
        "fact never entered the world — probe vacuous"
    )


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
        memory_tag_rows = connection.execute(
            "SELECT tags_json FROM memory_episodes WHERE session_id = ?",
            (session_id,),
        ).fetchall()
    # Consolidation (docs/22 C2/P3) marks originals CONSOLIDATED_TAG and unindexes them
    # from Qdrant, but never deletes the SQLite row -- so counting every row here would
    # diverge from the Qdrant-indexed session_memory_count as soon as one roll-up runs
    # (docs/22 P2, fixed 2026-07-11). Excluding consolidated-marked rows keeps the two
    # sides comparable.
    parsed_tags = [json.loads(tags_json) for (tags_json,) in memory_tag_rows]
    persisted_memory_count = sum(1 for tags in parsed_tags if CONSOLIDATED_TAG not in tags)
    # Report-only (P2.2 evidence): counted separately from the parity-preserving
    # persisted_memory_count above -- see PersistenceInspection's docstring comment.
    consolidated_memory_count = sum(1 for tags in parsed_tags if CONSOLIDATED_TAG in tags)
    consolidation_summary_count = sum(1 for tags in parsed_tags if SUMMARY_TAG in tags)
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
        consolidated_memory_count=consolidated_memory_count,
        consolidation_summary_count=consolidation_summary_count,
    )


def _turn_id_for_index(turns: Sequence[StoredTurn], turn_index: int) -> int | None:
    """Resolve a 1-based ordinal ``turn_index`` to the persisted ``turns.id``
    primary key, for provenance-based attribution (#76).

    ``turn_index`` must be the ACTUAL persisted DB ordinal, not necessarily a
    StoryEvent's raw ``definition_turn`` scripted-step number (#80, docs/26 §8
    Q4): once a definition-turn retry has fired anywhere earlier in a run, DB
    turn_index drifts away from scripted step number by however many retries
    were consumed before it (every POST -- success or controlled_failure --
    persists one row, so each consumed retry inserts exactly one extra row).
    ``inspect_story_event``'s ``definition_turn_index`` parameter is what tracks
    and supplies the correct value; passing the raw scripted step instead is only
    correct when no retry has fired before it -- true for every call when
    ``definition_retries=0``, the default.

    Direct SQLite lookup, not an HTTP round-trip: inspect_story_event already reads
    the session's turns straight from the database (see its ``recent_turns`` call
    just below), and ``turns.id`` is an internal primary key the HTTP API
    deliberately never exposes (CreateTurnResponse/RecentTurnResponse/
    TurnDetailResponse all expose ``turn_index`` only) -- so this stays consistent
    with that boundary instead of adding a new API surface just for a diagnostics
    script. None if no turn was ever persisted under that index (degrades to
    phrasing-only matching for that event rather than raising).
    """
    for turn in turns:
        if turn.turn_index == turn_index:
            return turn.id
    return None


def inspect_story_event(
    *,
    database_path: Path,
    settings: Settings,
    session_id: str,
    event: StoryEvent,
    definition_turn_index: int | None = None,
) -> EventAttribution:
    """Inspect one probe event's retrieval/attribution state at its callback turn.

    ``definition_turn_index`` (#80, docs/26 §8 Q4): the definition turn's ACTUAL
    persisted DB ordinal, tracked by ``run_checkpoint`` across any consumed
    definition-turn retries and passed through as ``event_inspector``'s third
    argument. ``None`` (the default -- every pre-#80 caller, and any ad hoc/
    offline invocation that hasn't tracked it, e.g. a manual owner-side
    ``inspect_story_event`` call per docs/25) falls back to
    ``event.definition_turn`` (the raw scripted step), which is only correct
    when no retry has fired earlier in the run -- exactly reproducing pre-#80
    behavior at ``definition_retries=0``.
    """
    connection = connect_sqlite(database_path)
    try:
        memory_repository = SQLiteMemoryRepository(connection)
        memories = memory_repository.list_memories_for_session(session_id)
        turn_repository = SQLiteTurnRepository(connection)
        recent_turns = turn_repository.list_recent_turns(
            session_id,
            settings.recent_dialogue_turns,
        )
        resolved_definition_turn_index = (
            definition_turn_index
            if definition_turn_index is not None
            else event.definition_turn
        )
        definition_turn_id = _turn_id_for_index(
            turn_repository.list_all_turns(session_id),
            resolved_definition_turn_index,
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
        definition_turn_id=definition_turn_id,
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
        # #80 (docs/26 §8 Q4): a consumed definition-turn retry is a distinct
        # header, not silently merged into the original attempt's section --
        # "auditable, docs/25 forensics style", matching the raw JSON's
        # is_definition_retry/retry_attempt labeling.
        header = f"#### Turn {turn['turn_index']}"
        if turn.get("is_definition_retry"):
            header += f" (definition retry {turn.get('retry_attempt')})"
        lines.extend(
            [
                header,
                "",
                f"- outcome: {turn['outcome']}",
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
    """Count points in ``collection``, excluding the P1.4 embedding-model fingerprint
    sentinel meta point (docs/22 P4, fixed 2026-07-11).

    The sentinel is a real point ``ensure_collection`` writes into every collection it
    touches, so an unscoped ``client.count()`` (the ``session_id is None`` canon-lore
    path) would count it too -- inflating ``canon_lore_count`` by 1 and letting the
    checkpoint's ``>= 1 lore`` gate pass on a stamped-but-otherwise-empty collection,
    diverging from ``InMemoryVectorStore`` (whose fingerprint lives in a separate dict,
    never a counted entry). Reuses the same ``must_not`` exclusion
    ``QdrantVectorStore._build_qdrant_filter`` applies on every search path.
    """
    if not vector_store.client.collection_exists(collection_name=collection.value):
        return 0
    from qdrant_client.http import models

    must: list[Any] = []
    if session_id is not None:
        must.append(
            models.FieldCondition(
                key="session_id",
                match=models.MatchValue(value=session_id),
            )
        )
    count_filter = models.Filter(
        must=must,
        must_not=[
            models.FieldCondition(
                key=_SENTINEL_PAYLOAD_KEY,
                match=models.MatchValue(value=True),
            )
        ],
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
    parser.add_argument(
        "--definition-retries",
        type=int,
        default=0,
        help=(
            "Harness-local scenario-semantics knob (#80, docs/26 Stage 5 / §8 Q4): "
            "resend a probe's failed DEFINITION turn message up to N times before "
            "failing fast. Default 0 reproduces prior behavior byte-identically."
        ),
    )
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
        event_inspector=lambda session_id, event, definition_turn_index: inspect_story_event(
            database_path=args.database_path,
            settings=settings,
            session_id=session_id,
            event=event,
            definition_turn_index=definition_turn_index,
        ),
        expected_model=args.expected_model,
        turn_count=args.turn_count,
        fail_on_structured_warnings=args.fail_on_structured_warnings == "1",
        definition_retries=args.definition_retries,
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
