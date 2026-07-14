"""Stage 0 write-path benchmark (docs/26 Stage 0 "Instruments first", backlog #75).

A scripted multi-turn transcript through the real ``TurnMemoryStage``, exercising the
REAL deterministic extractor (``app.memory.deterministic_extractor``) and REAL
write-dedup (``app.orchestration.stages.memory_dedup``) with only the curator's LLM
call faked -- task-aware, keyed on a marker in the completed-turn message, mirroring
``app.evals.memory_continuity._TaskAwareFakeProvider``. No engine code is
reimplemented here: this module drives the production ``TurnMemoryStage`` directly, so
a regression in either real component fails this eval.

Proves (a) a player-stated durable fact (promise/entrusted class) survives
extraction + write-dedup into a persisted candidate set, and (b)/(c) the #72
compass/dawn adversarial pair -- pinned verbatim from
``tests/unit/orchestration/stages/test_memory_dedup.py`` and
``tests/unit/test_deterministic_extractor.py`` -- still behaves exactly as #72 left
it at the ``TurnMemoryStage`` integration level: the distinct terse compass-gift
candidate SURVIVES the 0.75 write-dedup threshold, a near-verbatim restatement of it
is DROPPED, and the drop is auditable (the warning names the dropped summary). Also
proves (d) the Stage 2 best-match tag/importance fold (docs/26 §3.2, backlog #77): a
deterministic candidate covered (at the separate, unchanged 0.5 coverage-drop
threshold) by this turn's own curated summary donates its tag and importance=4 onto
that summary instead of being silently discarded.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from app.agents.memory_curator import MemoryCurator
from app.domain import MemoryEpisode
from app.evals.fixtures import build_eval_fixture
from app.llm.provider import LlmProvider, LlmRequest, LlmResponse
from app.memory import MemoryEpisodeStore
from app.memory.deterministic_extractor import DETERMINISTIC_EVENT_IMPORTANCE
from app.orchestration.stages.memory import TurnMemoryStage
from app.orchestration.stages.routing import TurnRoutingStage
from app.persistence import SQLiteMemoryRepository, SQLiteSessionRepository
from app.persistence.sqlite import connect_sqlite, initialize_database

# Pinned verbatim from #72 (tests/unit/orchestration/stages/test_memory_dedup.py,
# tests/unit/test_deterministic_extractor.py) -- do not reword. These strings are the
# exact adversarial pair the WRITE_DEDUP_COVERAGE_THRESHOLD=0.75 split was tuned
# against; a reworded copy would silently stop pinning the live finding.
_DAWN_PROMISE_SUMMARY = (
    "The player promised to return to the archive before dawn, provided "
    "Iria Vale keeps the archive door unbarred."
)
_COMPASS_GIFT_SUMMARY = "The player gave Iria Vale a silver compass to keep until his return."
_COMPASS_RESTATEMENT_SUMMARY = "The player gave Iria Vale a silver compass to keep."

# First-person, explicit-trigger phrasing so the REAL deterministic extractor fires
# (app.memory.deterministic_extractor._PROMISE_PATTERN + _DEADLINE_PATTERN).
_PROMISE_TURN_MESSAGE = "I promise to guard the eastern vault before midnight."
_PROMISE_FRAMED_SUMMARY = f'The player stated: "{_PROMISE_TURN_MESSAGE}"'
_ASSISTANT_MESSAGE = "Iria considers this quietly."

# Stage 2 (#77, docs/26 §3.2): a turn where BOTH sides fire for real -- the message
# matches a real deterministic trigger (a distinct bridge/sunrise event, so it cannot
# collide with the vault/dawn/compass facts above at either coverage threshold; see
# the scores recorded below) while the curator, keyed on its own marker, curates a
# paraphrase of the SAME event that omits the tag entirely and under-scores its
# importance. This is what proves the coverage-drop fold, not just the write-dedup
# path proven by (b)/(c) above.
_FOLD_TARGET_MARKER = "FOLD_TARGET_TURN"
_FOLD_TARGET_MESSAGE = (
    f"{_FOLD_TARGET_MARKER}. I promise to meet at the northern bridge before sunrise."
)
_FOLD_TARGET_FRAMED_SUMMARY = (
    'The player stated: "I promise to meet at the northern bridge before sunrise."'
)
# Coverage of the deterministic candidate above against this curated paraphrase is
# 0.83 (well clear of the 0.5 coverage-drop threshold) via best_covering_summary;
# coverage of this same paraphrase against the three PRE-EXISTING persisted summaries
# above is 0.0 at the (separate, unchanged) 0.75 write-dedup threshold, so the fold
# survives into a persisted memory rather than being write-deduped away.
_FOLD_CURATOR_SUMMARY = "The player swore to meet at the northern bridge before sunrise."

# Markers select the fake curator's scripted response (mirrors memory_continuity's
# _EARLY_DIALOGUE_MARKER pattern); none of the first three match the extractor's
# promise/entrust/agreement/deadline patterns, so fallback_candidates stays empty on
# turns 2-4 and the curator alone controls what gets persisted there. The fold-target
# marker is the one exception: its message deliberately DOES match a trigger pattern.
_SEED_DAWN_PROMISE_MARKER = "SEED_DAWN_PROMISE_TURN"
_COMPASS_GIFT_MARKER = "COMPASS_GIFT_TURN"
_COMPASS_RESTATEMENT_MARKER = "COMPASS_RESTATEMENT_TURN"

_DECLINE_JSON = json.dumps(
    {"write_memory": False, "memories": [], "reason": "Deterministic extractor covers this."}
)


class MemoryWriteLifecycleResult(BaseModel):
    passed: bool
    checks: dict[str, bool] = Field(default_factory=dict)


def _curated_json(summary: str, *, importance: int, tags: list[str]) -> str:
    return json.dumps(
        {
            "write_memory": True,
            "memories": [
                {
                    "summary": summary,
                    "visibility": "player",
                    "importance": importance,
                    "tags": tags,
                    "scene_id": "eval-rose-gallery",
                    "actor_id": "iria-vale",
                }
            ],
            "reason": "Scripted memory_write_lifecycle fixture turn.",
        }
    )


class _ScriptedCuratorProvider(LlmProvider):
    """Memory-extraction-only fake: response chosen by a marker substring in the
    completed-turn message. The first (marker-free) turn falls through to decline,
    so its persisted memory comes entirely from the REAL deterministic extractor's
    fallback path, not from this fake."""

    def __init__(self, scripted: dict[str, str]) -> None:
        self._scripted = scripted
        self.requests: list[LlmRequest] = []

    async def generate(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        completed_turn = request.messages[-1].content
        for marker, response_text in self._scripted.items():
            if marker in completed_turn:
                return LlmResponse(
                    text=response_text, provider="fake", model=request.model, finish_reason="stop"
                )
        return LlmResponse(
            text=_DECLINE_JSON, provider="fake", model=request.model, finish_reason="stop"
        )


def evaluate_memory_write_lifecycle() -> MemoryWriteLifecycleResult:
    return asyncio.run(_evaluate_memory_write_lifecycle())


async def _evaluate_memory_write_lifecycle() -> MemoryWriteLifecycleResult:
    fixture = build_eval_fixture()
    provider = _ScriptedCuratorProvider(
        {
            _SEED_DAWN_PROMISE_MARKER: _curated_json(
                _DAWN_PROMISE_SUMMARY, importance=3, tags=["promise"]
            ),
            _COMPASS_GIFT_MARKER: _curated_json(
                _COMPASS_GIFT_SUMMARY, importance=2, tags=["entrusted"]
            ),
            _COMPASS_RESTATEMENT_MARKER: _curated_json(
                _COMPASS_RESTATEMENT_SUMMARY, importance=2, tags=["entrusted"]
            ),
            # Stage 2 (#77): curator paraphrases the fold-target event but forgets
            # the tag and under-scores importance -- the fold must repair both.
            _FOLD_TARGET_MARKER: _curated_json(
                _FOLD_CURATOR_SUMMARY, importance=2, tags=["reaction"]
            ),
        }
    )
    routing_stage = TurnRoutingStage(
        local_model=fixture.memory_route.model,
        cloud_model=fixture.cloud_route.model,
        local_max_tokens=fixture.local_route.max_tokens,
        local_structured_max_tokens=fixture.memory_route.max_tokens,
        cloud_max_tokens=fixture.cloud_route.max_tokens,
        local_temperature=fixture.local_route.temperature,
        cloud_temperature=fixture.cloud_route.temperature,
    )

    with tempfile.TemporaryDirectory(prefix="rolerag-memory-write-lifecycle-") as temp_dir:
        connection = connect_sqlite(Path(temp_dir) / "memory-write-lifecycle.db")
        initialize_database(connection)
        SQLiteSessionRepository(connection).create_session(fixture.session)
        memory_store = MemoryEpisodeStore(memory_repository=SQLiteMemoryRepository(connection))
        stage = TurnMemoryStage(
            provider=provider,
            memory_store=memory_store,
            memory_curator=MemoryCurator(),
            memory_indexer=None,
            routing_stage=routing_stage,
            gating="always",
            # Byte-identical no-op (>=1.0): this transcript targets the lexical
            # coverage dedup (is_covered_by_summaries) #72 tuned, not the separate
            # embedding-based semantic dedup.
            write_dedup_cosine_threshold=1.0,
        )

        turn_messages = (
            _PROMISE_TURN_MESSAGE,
            f"{_SEED_DAWN_PROMISE_MARKER}: I ask Iria to keep the archive door "
            "unbarred while I am away.",
            f"{_COMPASS_GIFT_MARKER}: I give Iria Vale a silver compass and ask "
            "her to keep it until I return.",
            f"{_COMPASS_RESTATEMENT_MARKER}: I check that Iria Vale still has "
            "the silver compass I gave her.",
            _FOLD_TARGET_MESSAGE,
        )
        turn_results = [
            await stage.run(
                session=fixture.session,
                scene=fixture.scene,
                persona=fixture.primary_persona,
                user_message=message,
                assistant_message=_ASSISTANT_MESSAGE,
                retrieval_confidence=None,
                scene_complexity=1,
            )
            for message in turn_messages
        ]
        memories = memory_store.list_memories_for_session(fixture.session.id)
        connection.close()

    promise_turn, _seed_turn, compass_turn, restatement_turn, fold_target_turn = turn_results
    promise_memory = _find_memory(memories, _PROMISE_FRAMED_SUMMARY)
    compass_memory = _find_memory(memories, _COMPASS_GIFT_SUMMARY)
    restatement_memory = _find_memory(memories, _COMPASS_RESTATEMENT_SUMMARY)
    dropped_warning = next(
        (
            warning
            for warning in restatement_turn.warnings
            if warning.startswith("memory dedup dropped")
        ),
        None,
    )

    # Stage 2 (#77, docs/26 §3.2 best-match tag/importance fold): the fold-target
    # turn's message matches a real deterministic trigger AND the (fake) curator
    # independently curates a paraphrase of the same event that carries neither the
    # tag nor importance=4. The curator-coverage-drop site folds the deterministic
    # candidate onto that curated summary instead of discarding it, so the persisted
    # memory is the curator's paraphrase (_FOLD_CURATOR_SUMMARY, not the deterministic
    # framing) enriched with the donated tag(s) and importance.
    fold_target_memory = _find_memory(memories, _FOLD_CURATOR_SUMMARY)
    fold_warning = next(
        (
            warning
            for warning in fold_target_turn.warnings
            if warning.startswith("deterministic candidate folded")
        ),
        None,
    )

    # --- Stage 3 extension point (#78, docs/26 Section 3.3 Lane A tag-eligible
    # canon pinning): once CANON_TAG_PINNING widens build_standing_facts
    # eligibility, add assertions here that promise_memory/compass_memory appear
    # in canon_builder.build_standing_facts(memories, ...) output (standing-facts
    # inclusion) with the flag on, and that the output stays byte-identical with
    # the flag off. Not implemented yet -- Stage 0 is instrumentation only. ---

    checks = {
        "promise_turn_writes_memory": promise_turn.memory_written is True,
        "durable_fact_survives_extraction_and_dedup": (
            promise_memory is not None
            and "promise" in promise_memory.tags
            and "deadline" in promise_memory.tags
            and promise_memory.importance == DETERMINISTIC_EVENT_IMPORTANCE
        ),
        "dawn_promise_seed_persisted": _find_memory(memories, _DAWN_PROMISE_SUMMARY) is not None,
        "distinct_compass_candidate_survives_write_dedup": (
            compass_turn.memory_written is True and compass_memory is not None
        ),
        "near_verbatim_restatement_dropped_by_write_dedup": (
            restatement_turn.memory_written is False and restatement_memory is None
        ),
        "dropped_restatement_is_auditable": (
            dropped_warning is not None and "silver compass to keep" in dropped_warning
        ),
        "deterministic_tag_and_importance_fold_onto_curated_summary": (
            fold_target_turn.memory_written is True
            and fold_target_memory is not None
            and "promise" in fold_target_memory.tags
            and "deadline" in fold_target_memory.tags
            and "reaction" in fold_target_memory.tags  # curator's own tag survives
            and fold_target_memory.importance == DETERMINISTIC_EVENT_IMPORTANCE
            # Folded, not duplicated: the deterministic extractor's own framed text
            # must NOT also appear as a separate persisted memory.
            and _find_memory(memories, _FOLD_TARGET_FRAMED_SUMMARY) is None
        ),
        "fold_is_auditable": (
            fold_warning is not None and "northern bridge before sunrise" in fold_warning
        ),
    }
    return MemoryWriteLifecycleResult(passed=all(checks.values()), checks=checks)


def _find_memory(memories: list[MemoryEpisode], summary: str) -> MemoryEpisode | None:
    return next((memory for memory in memories if memory.summary == summary), None)
