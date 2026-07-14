from __future__ import annotations

from datetime import UTC, datetime

from app.domain import MemoryEpisode, RetrievedChunk, Visibility
from app.orchestration.context_budget import ContextBudget, select_retrieved_chunks_for_prompt
from app.rag.diagnostics import ChunkRetrievalDiagnostic, RetrievalDiagnostics
from app.rag.lexical import LexicalHit
from app.rag.models import RagCollection
from app.rag.ranking import (
    SLICE_CONFIDENCE_EQUIVALENT,
    RetrievalRankingContext,
    SliceQuotas,
    apply_lexical_slice_quotas,
    rerank_chunks,
    slice_aware_confidence,
)


def _chunk(chunk_id: str, *, score: float, text: str | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        source=f"memory_episode:{chunk_id}",
        source_type="session_memory",
        # Distinct text per id so select_retrieved_chunks_for_prompt's N3 text-dedup
        # does not collapse the fixture chunks into one.
        text=text or f"distinct fact for {chunk_id}",
        score=score,
        visibility=Visibility.PLAYER,
        session_id="session",
    )


def _diag(chunk: RetrievedChunk, *, rank: int | None) -> ChunkRetrievalDiagnostic:
    # Mirrors the real rerank contract: adjusted == original + sum(applied_boosts).
    return ChunkRetrievalDiagnostic(
        id=chunk.id,
        source=chunk.source,
        source_type=chunk.source_type,
        collection=RagCollection.SESSION_MEMORY,
        visibility=chunk.visibility,
        tags=list(chunk.tags),
        original_score=chunk.score,
        adjusted_score=chunk.score + 0.08,
        applied_boosts={"collection": 0.08},
        selected_rank=rank,
    )


def _diagnostics(chunks: list[RetrievedChunk]) -> RetrievalDiagnostics:
    return RetrievalDiagnostics(
        query="q",
        selected=[_diag(chunk, rank=i) for i, chunk in enumerate(chunks, start=1)],
        rejected=[],
    )


def _hit(memory_id: str, *, score: float, importance: int = 2) -> LexicalHit:
    return LexicalHit(
        memory_id=memory_id,
        score=score,
        importance=importance,
        created_ordinal=0.0,
        matched=("rule", "trust"),
    )


def _memory(memory_id: str, *, summary: str = "a stored fact") -> MemoryEpisode:
    return MemoryEpisode(
        id=memory_id,
        session_id="session",
        scene_id="scene",
        summary=summary,
        importance=2,
        visibility=Visibility.PLAYER,
        tags=["rule"],
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


# --- Golden byte-identity: quotas=0 is a no-op against the real rerank path -------


def test_quota_zero_is_byte_identical_to_rerank_output() -> None:
    candidates = [
        (RagCollection.SESSION_MEMORY, _chunk("m1", score=0.61, text="blue seal rule")),
        (RagCollection.SESSION_MEMORY, _chunk("m2", score=0.55, text="a market stall")),
        (RagCollection.CANON_LORE, _chunk("lore1", score=0.72, text="the keep's history")),
        (RagCollection.PERSONA_MEMORY, _chunk("p1", score=0.40, text="iria is wary")),
    ]
    context = RetrievalRankingContext(
        query="what rule did we agree", session_id="session", persona_id="persona"
    )
    chunks, diagnostics = rerank_chunks(context=context, candidates=candidates, top_k=13)

    # Even with lexical hits present, quota 0 must not touch the rerank output.
    result = apply_lexical_slice_quotas(
        chunks=chunks,
        diagnostics=diagnostics,
        lexical_hits=[_hit("m1", score=9.9)],
        memories_by_id={"m1": _memory("m1")},
        quotas=SliceQuotas(lexical=0),
        prompt_window=5,
    )

    assert result.chunks == chunks
    assert result.diagnostics == diagnostics
    assert result.slice_member_ids == frozenset()


def test_empty_lexical_hits_leave_chunks_unchanged() -> None:
    chunks = [_chunk("a", score=0.6), _chunk("b", score=0.5)]
    diagnostics = _diagnostics(chunks)

    result = apply_lexical_slice_quotas(
        chunks=chunks,
        diagnostics=diagnostics,
        lexical_hits=[],
        memories_by_id={},
        quotas=SliceQuotas(lexical=2),
        prompt_window=5,
    )

    assert result.chunks == chunks
    assert result.diagnostics == diagnostics
    assert result.slice_member_ids == frozenset()


# --- Reorder / injection semantics ------------------------------------------------


def test_slice_member_already_in_prompt_window_evicts_no_dense_chunk() -> None:
    # 6 dense chunks; the lexical hit's memory is at dense position 3 (index 2),
    # already inside the top-5 prompt window. Guaranteeing it must only reorder it to
    # the front -- the set of the top-5 must be unchanged (no extra dense eviction).
    chunks = [_chunk(cid, score=score) for cid, score in [
        ("d0", 0.60), ("d1", 0.55), ("hit", 0.50), ("d3", 0.45), ("d4", 0.40), ("d5", 0.35),
    ]]
    diagnostics = _diagnostics(chunks)
    before = {c.id for c in select_retrieved_chunks_for_prompt(chunks, budget=ContextBudget())}

    result = apply_lexical_slice_quotas(
        chunks=chunks,
        diagnostics=diagnostics,
        lexical_hits=[_hit("hit", score=8.0)],
        memories_by_id={"hit": _memory("hit")},
        quotas=SliceQuotas(lexical=1),
        prompt_window=5,
    )

    assert result.chunks[0].id == "hit"  # promoted to the front
    after = {
        c.id for c in select_retrieved_chunks_for_prompt(result.chunks, budget=ContextBudget())
    }
    assert after == before  # already-present member did not evict an extra dense chunk
    assert result.slice_member_ids == frozenset({"hit"})
    assert result.chunks.count(next(c for c in result.chunks if c.id == "hit")) == 1


def test_slice_member_deeper_than_window_is_pulled_into_prompt() -> None:
    # 13 dense chunks; the lexical hit is at dense position 12 (index 11), far below
    # the top-5 window -- exactly docs/26's "fused position 11" case. It must reach
    # the actor prompt after the reorder.
    chunks = [_chunk(f"d{i}", score=1.0 - i * 0.05) for i in range(13)]
    chunks[11] = _chunk("deep", score=0.45)
    diagnostics = _diagnostics(chunks)

    assert "deep" not in {
        c.id for c in select_retrieved_chunks_for_prompt(chunks, budget=ContextBudget())
    }

    result = apply_lexical_slice_quotas(
        chunks=chunks,
        diagnostics=diagnostics,
        lexical_hits=[_hit("deep", score=8.0)],
        memories_by_id={"deep": _memory("deep")},
        quotas=SliceQuotas(lexical=1),
        prompt_window=5,
    )

    selected = {
        c.id for c in select_retrieved_chunks_for_prompt(result.chunks, budget=ContextBudget())
    }
    assert "deep" in selected
    assert result.slice_member_ids == frozenset({"deep"})


def test_non_dense_hit_is_injected_from_memory_preserving_boost_identity() -> None:
    chunks = [_chunk(f"d{i}", score=0.6 - i * 0.05) for i in range(5)]
    diagnostics = _diagnostics(chunks)

    result = apply_lexical_slice_quotas(
        chunks=chunks,
        diagnostics=diagnostics,
        lexical_hits=[_hit("injected", score=7.5)],
        memories_by_id={"injected": _memory("injected", summary="the blue seal rule")},
        quotas=SliceQuotas(lexical=1),
        prompt_window=5,
    )

    injected = result.chunks[0]
    assert injected.id == "injected"
    assert injected.text == "the blue seal rule"
    assert injected.score == 0.0
    assert injected.source == "memory_episode:injected"
    assert injected.source_type == RagCollection.SESSION_MEMORY.value

    entry = next(e for e in result.diagnostics.selected if e.id == "injected")  # type: ignore[union-attr]
    assert entry.original_score == 0.0
    assert entry.adjusted_score == 0.0
    assert entry.applied_boosts == {}
    assert entry.slice_score == 7.5
    assert entry.slice_guaranteed is True
    assert entry.slice_matched_terms == ["rule", "trust"]
    # The dedicated slice_score field keeps the additive identity intact for every
    # chunk (fix 3): adjusted == original + sum(applied_boosts).
    for e in result.diagnostics.selected:  # type: ignore[union-attr]
        assert e.adjusted_score == e.original_score + sum(e.applied_boosts.values())


def test_dense_slice_winner_keeps_real_scores_and_gains_slice_labels() -> None:
    chunks = [_chunk("hit", score=0.30), _chunk("other", score=0.55)]
    diagnostics = _diagnostics(chunks)

    result = apply_lexical_slice_quotas(
        chunks=chunks,
        diagnostics=diagnostics,
        lexical_hits=[_hit("hit", score=9.0)],
        memories_by_id={"hit": _memory("hit")},
        quotas=SliceQuotas(lexical=1),
        prompt_window=5,
    )

    entry = next(e for e in result.diagnostics.selected if e.id == "hit")  # type: ignore[union-attr]
    assert entry.original_score == 0.30  # real vector score preserved
    assert entry.applied_boosts == {"collection": 0.08}  # real boosts preserved
    assert entry.slice_score == 9.0
    assert entry.slice_guaranteed is True


def test_no_double_appearance_when_hit_is_already_dense() -> None:
    chunks = [_chunk("a", score=0.6), _chunk("hit", score=0.5), _chunk("b", score=0.4)]
    diagnostics = _diagnostics(chunks)

    result = apply_lexical_slice_quotas(
        chunks=chunks,
        diagnostics=diagnostics,
        lexical_hits=[_hit("hit", score=8.0)],
        memories_by_id={"hit": _memory("hit")},
        quotas=SliceQuotas(lexical=1),
        prompt_window=5,
    )

    ids = [c.id for c in result.chunks]
    assert ids.count("hit") == 1
    assert sorted(ids) == ["a", "b", "hit"]
    # Diagnostics selected ranks are a clean 1..N over the reordered list.
    assert [e.selected_rank for e in result.diagnostics.selected] == [1, 2, 3]  # type: ignore[union-attr]


def test_injected_hit_beyond_rerank_window_is_dropped_from_rejected() -> None:
    chunks = [_chunk("a", score=0.6)]
    diagnostics = RetrievalDiagnostics(
        query="q",
        selected=[_diag(chunks[0], rank=1)],
        rejected=[_diag(_chunk("deep", score=0.1), rank=None)],
    )

    result = apply_lexical_slice_quotas(
        chunks=chunks,
        diagnostics=diagnostics,
        lexical_hits=[_hit("deep", score=8.0)],
        memories_by_id={"deep": _memory("deep")},
        quotas=SliceQuotas(lexical=1),
        prompt_window=5,
    )

    # "deep" is now a guaranteed selected member; it must not also linger in rejected.
    assert [e.id for e in result.diagnostics.rejected] == []  # type: ignore[union-attr]
    assert "deep" in {e.id for e in result.diagnostics.selected}  # type: ignore[union-attr]


# --- min_slice_score floor (fix 2) ------------------------------------------------


def test_min_slice_score_floor_filters_weak_hits_when_set() -> None:
    chunks = [_chunk("d0", score=0.6)]
    diagnostics = _diagnostics(chunks)
    hits = [_hit("strong", score=3.0), _hit("weak", score=1.0)]

    result = apply_lexical_slice_quotas(
        chunks=chunks,
        diagnostics=diagnostics,
        lexical_hits=hits,
        memories_by_id={"strong": _memory("strong"), "weak": _memory("weak")},
        quotas=SliceQuotas(lexical=2, min_slice_score=2.0),
        prompt_window=5,
    )

    assert result.slice_member_ids == frozenset({"strong"})


def test_min_slice_score_none_applies_no_floor() -> None:
    chunks = [_chunk("d0", score=0.6)]
    diagnostics = _diagnostics(chunks)
    hits = [_hit("strong", score=3.0), _hit("weak", score=0.001)]

    result = apply_lexical_slice_quotas(
        chunks=chunks,
        diagnostics=diagnostics,
        lexical_hits=hits,
        memories_by_id={"strong": _memory("strong"), "weak": _memory("weak")},
        quotas=SliceQuotas(lexical=2, min_slice_score=None),
        prompt_window=5,
    )

    assert result.slice_member_ids == frozenset({"strong", "weak"})


def test_quota_caps_number_of_guaranteed_members() -> None:
    chunks = [_chunk("d0", score=0.6)]
    diagnostics = _diagnostics(chunks)
    hits = [_hit(f"h{i}", score=5.0 - i) for i in range(4)]

    result = apply_lexical_slice_quotas(
        chunks=chunks,
        diagnostics=diagnostics,
        lexical_hits=hits,
        memories_by_id={f"h{i}": _memory(f"h{i}") for i in range(4)},
        quotas=SliceQuotas(lexical=2),
        prompt_window=5,
    )

    assert result.slice_member_ids == frozenset({"h0", "h1"})


# --- slice-aware confidence -------------------------------------------------------


def test_confidence_no_slice_matches_plain_max() -> None:
    chunks = [_chunk("a", score=0.31), _chunk("b", score=0.62)]
    assert slice_aware_confidence(chunks, frozenset()) == 0.62


def test_confidence_slice_only_rescue_is_not_low_confidence() -> None:
    # An injected chunk carries score 0.0; without the slice-equivalent floor the
    # turn would read as 0.0 (low confidence) on exactly the turn the slice rescued.
    injected = _chunk("hit", score=0.0)
    assert slice_aware_confidence([injected], frozenset({"hit"})) == SLICE_CONFIDENCE_EQUIVALENT
    assert SLICE_CONFIDENCE_EQUIVALENT >= 0.45  # clears the default low_retrieval_confidence


def test_confidence_dense_slice_member_keeps_higher_real_score() -> None:
    strong = _chunk("hit", score=0.9)
    assert slice_aware_confidence([strong], frozenset({"hit"})) == 0.9


def test_confidence_ignores_non_player_chunks() -> None:
    gm = RetrievedChunk(
        id="gm", source="s", source_type="t", text="x", score=0.99, visibility=Visibility.GM
    )
    player = _chunk("p", score=0.2)
    assert slice_aware_confidence([gm, player], frozenset()) == 0.2


def test_confidence_empty_is_zero() -> None:
    assert slice_aware_confidence([], frozenset()) == 0.0


def test_confidence_preserves_negative_max_without_slice() -> None:
    # Plain max over player scores, byte-identical to the pre-Lane-B behavior even
    # for negative cosine scores (not floored at 0.0).
    chunks = [_chunk("a", score=-0.2), _chunk("b", score=-0.5)]
    assert slice_aware_confidence(chunks, frozenset()) == -0.2


# --- SliceQuotas dataclass --------------------------------------------------------


def test_slice_quotas_defaults_and_frozen() -> None:
    quotas = SliceQuotas()
    assert quotas.lexical == 0
    assert quotas.min_slice_score is None
    try:
        quotas.lexical = 2  # type: ignore[misc]
    except AttributeError:
        return
    raise AssertionError("SliceQuotas should be frozen")
