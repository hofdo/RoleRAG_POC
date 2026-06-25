"""Deterministic retrieval-miss eval — rank/score floor guard for durable memories.

`event_key_retrieval` already verifies that each seeded event lands at rank 1 for
its callback query. This eval is the regression backstop: it ranks each durable
memory across the *full* candidate set (selected ∪ rejected) via
`summarize_retrieval_miss` and asserts the memory stays above a rank floor and an
adjusted-score floor. A future dedup/boost/recency tradeoff that pushes a wanted
memory below the cut — even by one rank — fails here.

The `min_adjusted_score` floor is empirical. With the deterministic
EventKeywordEmbeddingProvider the baseline adjusted scores are::

    before_dawn_promise   1   0.892350   (observed minimum)
    silver_compass        1   1.115000
    blue_seal_trust_rule  1   1.139597
    key_hiding_place      1   1.189597
    three_tap_signal      1   1.309427

The floor is set just below the observed minimum (0.88, ~1.4% under 0.892350) so
the eval passes today but bites on a meaningful score degradation. If the
embedding provider changes, re-run the baseline and reset the floor.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.diagnostics.retrieval_miss import summarize_retrieval_miss
from app.evals.event_key_retrieval import SEEDED_EVENTS, seed_event_retrieval_context
from app.evals.fixtures import EvalFixture
from app.rag.retriever import build_retrieval_query

_MIN_ADJUSTED_SCORE = 0.88


class RetrievalMissResult(BaseModel):
    passed: bool
    checks: dict[str, bool] = Field(default_factory=dict)


def evaluate_retrieval_miss(
    fixture: EvalFixture,
    *,
    top_k: int = 5,
    max_rank: int = 3,
    min_adjusted_score: float = _MIN_ADJUSTED_SCORE,
) -> RetrievalMissResult:
    context = seed_event_retrieval_context(fixture, db_name="retrieval-miss")
    retriever = context.retriever
    event_memory_ids = context.event_memory_ids
    connection = context.connection

    checks: dict[str, bool] = {}
    for event in SEEDED_EVENTS:
        result = retriever.retrieve_for_actor_with_diagnostics(
            query=build_retrieval_query(
                user_message=event.callback_message,
                scene=fixture.scene,
                persona=fixture.primary_persona,
                recent_turns=(),
            ),
            lexical_query=event.callback_message,
            world_id=fixture.world.id,
            session_id=fixture.session.id,
            persona_id=fixture.primary_persona.id,
            scene_id=fixture.scene.id,
            top_k=top_k,
        )
        memory_id = event_memory_ids[event.key]
        report = summarize_retrieval_miss(
            expected_ids=[memory_id],
            selected=[chunk.model_dump() for chunk in result.diagnostics.selected],
            rejected=[chunk.model_dump() for chunk in result.diagnostics.rejected],
        )
        rank = report.ranks[0]
        checks[f"{event.key}_present"] = memory_id not in report.absent_ids
        checks[f"{event.key}_selected"] = memory_id not in report.missed_ids
        checks[f"{event.key}_within_rank_floor"] = (
            rank.overall_rank is not None and rank.overall_rank <= max_rank
        )
        checks[f"{event.key}_above_score_floor"] = (
            rank.adjusted_score is not None and rank.adjusted_score >= min_adjusted_score
        )
    connection.close()

    return RetrievalMissResult(passed=all(checks.values()), checks=checks)
