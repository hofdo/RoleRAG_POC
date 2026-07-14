from __future__ import annotations

import math
from datetime import UTC, datetime

from app.domain import MemoryEpisode, Visibility
from app.memory.consolidation import CONSOLIDATED_TAG
from app.rag.lexical import (
    LexicalHit,
    idf_over,
    score_memories_lexical,
)


def _memory(
    memory_id: str,
    summary: str,
    *,
    importance: int = 2,
    visibility: Visibility = Visibility.PLAYER,
    tags: list[str] | None = None,
    created_at: datetime | None = None,
) -> MemoryEpisode:
    return MemoryEpisode(
        id=memory_id,
        session_id="session",
        scene_id="scene",
        summary=summary,
        importance=importance,
        visibility=visibility,
        tags=tags or [],
        created_at=created_at,
    )


def test_idf_over_is_log_n_over_df() -> None:
    # 4 docs: "keep" in all (df=4 -> idf 0, frame vocabulary is free), each rare
    # term in exactly one doc (df=1 -> idf log(4)).
    docs = {
        "m1": {"blue", "seal", "keep"},
        "m2": {"wax", "keep"},
        "m3": {"gate", "keep"},
        "m4": {"keep"},
    }
    idf = idf_over(docs)

    assert idf["keep"] == 0.0
    assert idf["blue"] == math.log(4 / 1)
    assert idf["seal"] == math.log(4 / 1)
    assert idf["wax"] == math.log(4 / 1)


def test_idf_over_empty_pool_is_empty() -> None:
    assert idf_over({}) == {}


def test_score_is_summed_idf_of_matched_terms() -> None:
    memories = [
        _memory("m1", "blue seal keep"),
        _memory("m2", "wax keep"),
        _memory("m3", "gate keep"),
        _memory("m4", "keep"),
    ]

    hits = score_memories_lexical(query_text="blue seal", memories=memories)

    # Only m1 shares content terms with the query; blue and seal each occur in one
    # of four docs -> score = 2 * log(4). "keep" (in every doc) would contribute 0.
    assert len(hits) == 1
    assert hits[0].memory_id == "m1"
    assert hits[0].score == 2 * math.log(4)
    assert hits[0].matched == ("blue", "seal")


def test_frame_vocabulary_term_scores_zero_but_still_hits() -> None:
    # "keep" is in every doc (idf 0). A query that only shares "keep" still returns
    # hits (non-empty intersection) but at score 0 -- the no-floor behavior docs/26
    # fix 2 documents: without min_slice_score, weak matches still qualify.
    memories = [
        _memory("m1", "blue keep"),
        _memory("m2", "wax keep"),
    ]

    hits = score_memories_lexical(query_text="keep", memories=memories)

    assert {hit.memory_id for hit in hits} == {"m1", "m2"}
    assert all(hit.score == 0.0 for hit in hits)


def test_ordering_tie_breaks_by_score_then_importance_then_recency_then_id() -> None:
    older = datetime(2026, 1, 1, tzinfo=UTC)
    newer = datetime(2026, 6, 1, tzinfo=UTC)
    memories = [
        _memory("a", "signal alpha", importance=2, created_at=newer),
        _memory("b", "signal beta", importance=2, created_at=older),
        _memory("c", "signal gamma", importance=5, created_at=None),
        _memory("d", "signal delta", importance=2, created_at=None),
        _memory("e", "filler one", importance=1),
        _memory("f", "filler two", importance=1),
    ]

    hits = score_memories_lexical(query_text="signal", memories=memories)

    # signal is in 4 of 6 docs, so all four matches share one equal score. The
    # deterministic tie-break then orders them: c wins on importance (5 > 2); among
    # the importance-2 trio, newer created_at wins (a before b), and the None-created
    # d sorts last (oldest). filler docs never match.
    ordered = [hit.memory_id for hit in hits]
    assert ordered == ["c", "a", "b", "d"]
    assert all(hit.score == math.log(6 / 4) for hit in hits)


def test_none_created_at_ties_break_by_id_not_raising() -> None:
    # Two None-created memories with identical score+importance: the sort must not
    # raise on the None created_at hazard and must fall through to the id tie-break.
    memories = [
        _memory("zeta", "signal one", importance=2, created_at=None),
        _memory("alpha", "signal two", importance=2, created_at=None),
        _memory("filler", "nothing here", importance=1),
    ]

    hits = score_memories_lexical(query_text="signal", memories=memories)

    assert [hit.memory_id for hit in hits] == ["alpha", "zeta"]


def test_pool_excludes_non_player_visibility() -> None:
    memories = [
        _memory("visible", "blue seal", visibility=Visibility.PLAYER),
        _memory("gm", "blue seal", visibility=Visibility.GM),
        _memory("private", "blue seal", visibility=Visibility.CHARACTER_PRIVATE),
    ]

    hits = score_memories_lexical(query_text="blue seal", memories=memories)

    assert [hit.memory_id for hit in hits] == ["visible"]


def test_pool_excludes_consolidated_originals() -> None:
    # docs/26 fix 1: a consolidation-retired original must never be resurrected into
    # the prompt via the lexical path.
    memories = [
        _memory("live", "blue seal", tags=["rule"]),
        _memory("retired", "blue seal", tags=["rule", CONSOLIDATED_TAG]),
    ]

    hits = score_memories_lexical(query_text="blue seal", memories=memories)

    assert [hit.memory_id for hit in hits] == ["live"]


def test_tags_contribute_to_doc_terms() -> None:
    # A term present only in a memory's tags (not its summary) still matches.
    memories = [
        _memory("tagged", "a quiet evening", tags=["compass"]),
        _memory("plain", "a quiet evening", tags=[]),
    ]

    hits = score_memories_lexical(query_text="compass", memories=memories)

    assert [hit.memory_id for hit in hits] == ["tagged"]
    # Query and tag both pass through the shared content_terms stemmer
    # ("compass" -> "compas"), so they match on the stemmed form.
    assert "compas" in hits[0].matched


def test_empty_pool_and_no_match_return_empty() -> None:
    assert score_memories_lexical(query_text="anything", memories=[]) == []
    assert (
        score_memories_lexical(
            query_text="nonmatching", memories=[_memory("m", "totally different words")]
        )
        == []
    )


def test_lexical_hit_is_frozen() -> None:
    hit = LexicalHit(memory_id="m", score=1.0, importance=2, created_ordinal=0.0, matched=("x",))
    try:
        hit.score = 2.0  # type: ignore[misc]
    except AttributeError:
        return
    raise AssertionError("LexicalHit should be frozen")
