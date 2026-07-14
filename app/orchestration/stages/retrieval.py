from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.domain import (
    RetrievalCandidateDiagnostic,
    RetrievedChunk,
    TurnInput,
    TurnRetrievalDiagnostics,
)
from app.orchestration.context_budget import ContextBudget
from app.orchestration.stages.session import LoadedTurnContext
from app.rag.diagnostics import ChunkRetrievalDiagnostic, RetrievalDiagnostics, RetrievalResult
from app.rag.lexical import LexicalHit, score_memories_lexical
from app.rag.ranking import (
    DEFAULT_SLICE_QUOTAS,
    SliceQuotas,
    apply_lexical_slice_quotas,
    slice_aware_confidence,
)
from app.rag.retriever import build_retrieval_query


class ActorContextRetrieving(Protocol):
    def retrieve_for_actor(
        self,
        *,
        query: str,
        world_id: str,
        session_id: str,
        persona_id: str,
        scene_id: str | None = None,
        top_k: int,
        lexical_query: str | None = None,
    ) -> list[RetrievedChunk]: ...


@dataclass(frozen=True)
class RetrievalStageResult:
    chunks: tuple[RetrievedChunk, ...]
    confidence: float | None
    warnings: tuple[str, ...]
    diagnostics: TurnRetrievalDiagnostics | None = None


class TurnRetrievalStage:
    def __init__(
        self,
        *,
        actor_context_retriever: ActorContextRetrieving | None,
        context_budget: ContextBudget,
        slice_quotas: SliceQuotas = DEFAULT_SLICE_QUOTAS,
    ) -> None:
        self.actor_context_retriever = actor_context_retriever
        self.context_budget = context_budget
        # Lane B lexical slice quotas (docs/26 §3.4, #79). Default is a byte-identical
        # no-op: quota 0 means the scorer never runs and the reorder returns its input
        # unchanged, so this whole stage reproduces the pre-Lane-B behavior exactly.
        self.slice_quotas = slice_quotas

    def run(
        self,
        *,
        turn_input: TurnInput,
        context: LoadedTurnContext,
    ) -> RetrievalStageResult:
        if self.actor_context_retriever is None:
            return RetrievalStageResult(chunks=(), confidence=None, warnings=())

        # Lane B: score the lexical slice from the session memories already loaded
        # for the Standing-facts block -- BEFORE the retriever call (fix 4) and
        # reusing context.session_memories, so no extra query runs and a later
        # dense/Qdrant failure cannot take the lexical leg down with it. A scorer
        # exception degrades Lane B to a no-op (dense-only) with a warning; it never
        # fails the turn. When the quota is off the scorer does not run at all.
        lexical_hits: tuple[LexicalHit, ...] = ()
        warnings: tuple[str, ...] = ()
        if self.slice_quotas.lexical > 0:
            try:
                lexical_hits = tuple(
                    score_memories_lexical(
                        query_text=turn_input.message,
                        memories=context.session_memories,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - Lane B is fail-open (dense-only)
                warnings = (f"lexical slice scorer skipped: {exc}",)

        memories_by_id = {memory.id: memory for memory in context.session_memories}
        query = build_retrieval_query(
            user_message=turn_input.message,
            scene=context.scene,
            persona=context.persona,
            recent_turns=context.recent_turns,
        )
        try:
            # Over-fetch by the standing-facts count: a durable fact pinned into the
            # Standing-facts block also wins rerank, so it is dropped from the prompt's
            # retrieved set (docs/22 C1, via select_retrieved_chunks_for_prompt). Fetching
            # the extra chunks means that exclusion recovers the freed slot with a distinct
            # fact instead of shrinking the retrieved block below budget.
            #
            # Also over-fetch by up to retrieved_chunks - 1 for read-time normalized-text
            # dedup (docs/22 N3, review round: the original N3 landing under-sized this).
            # select_retrieved_chunks_for_prompt drops later chunks whose normalized text
            # duplicates an earlier one, and rerank_chunks truncates its candidate pool to
            # exactly top_k *before* that dedup ever runs -- so a fixed-size window has no
            # replacement candidate available once a duplicate is dropped, and the block
            # silently shrinks. This margin covers the specific worst case it was sized
            # for: every slot but one in the first `retrieved_chunks` window is a
            # duplicate of an earlier one (retrieved_chunks - 1 wasted slots total,
            # matching the reviewer's repro). It is a bounded heuristic, not a universal
            # guarantee -- one fact repeated more times than the margin (heavy
            # near-duplicate pileup of a single memory, exactly the accumulation N3 exists
            # to catch) can still push genuinely distinct chunks out of the truncated
            # top_k window entirely, under-filling the block below budget even though
            # those chunks exist upstream (docs/22 P6, honesty note added 2026-07-11).
            # That residual case is a write-side/consolidation concern (semantic
            # write-dedup, or C2 folding the backlog), not something a larger fixed
            # over-fetch margin can close -- deliberately not addressed here. Deterministic
            # and bounded; does not change ranking or truncate anything the prompt step
            # wasn't already going to drop.
            top_k = (
                self.context_budget.retrieved_chunks
                + len(context.standing_facts)
                + max(self.context_budget.retrieved_chunks - 1, 0)
            )
            chunks, rag_diagnostics = self._retrieve(
                query=query,
                lexical_query=turn_input.message,
                world_id=context.session.world_id,
                session_id=context.session.id,
                persona_id=context.persona.id,
                scene_id=context.session.active_scene_id,
                top_k=top_k,
            )
        except Exception as exc:
            # Dense/Qdrant failure. Lane B survives (fix 4): if the lexical slice found
            # anything, degrade to a lexical-only prompt -- strictly better than today's
            # empty-chunks fallback -- instead of losing the lexical leg with the dense one.
            if self.slice_quotas.lexical > 0 and lexical_hits:
                fallback = apply_lexical_slice_quotas(
                    chunks=(),
                    diagnostics=None,
                    lexical_hits=lexical_hits,
                    memories_by_id=memories_by_id,
                    quotas=self.slice_quotas,
                    prompt_window=self.context_budget.retrieved_chunks,
                )
                if fallback.chunks:
                    return RetrievalStageResult(
                        chunks=tuple(fallback.chunks),
                        confidence=slice_aware_confidence(
                            fallback.chunks, fallback.slice_member_ids
                        ),
                        warnings=(*warnings, f"retrieval degraded to lexical slice: {exc}"),
                        diagnostics=None,
                    )
            return RetrievalStageResult(
                chunks=(),
                confidence=None,
                warnings=(*warnings, f"retrieval skipped: {exc}"),
            )

        # Dense retrieval succeeded. Apply the Lane B reorder on the FINAL prompt
        # window. This too is fail-open: a reorder exception degrades to dense-only
        # with a warning rather than failing the turn (quota-off is a no-op).
        try:
            slice_result = apply_lexical_slice_quotas(
                chunks=chunks,
                diagnostics=rag_diagnostics,
                lexical_hits=lexical_hits,
                memories_by_id=memories_by_id,
                quotas=self.slice_quotas,
                prompt_window=self.context_budget.retrieved_chunks,
            )
        except Exception as exc:  # noqa: BLE001 - Lane B is fail-open (dense-only)
            warnings = (*warnings, f"lexical slice reorder skipped: {exc}")
            slice_result = apply_lexical_slice_quotas(
                chunks=chunks,
                diagnostics=rag_diagnostics,
                lexical_hits=(),
                memories_by_id=memories_by_id,
                quotas=self.slice_quotas,
                prompt_window=self.context_budget.retrieved_chunks,
            )

        turn_diagnostics = (
            _to_turn_diagnostics(slice_result.diagnostics)
            if slice_result.diagnostics is not None
            else None
        )
        return RetrievalStageResult(
            chunks=tuple(slice_result.chunks),
            confidence=slice_aware_confidence(slice_result.chunks, slice_result.slice_member_ids),
            warnings=warnings,
            diagnostics=turn_diagnostics,
        )

    def _retrieve(
        self,
        *,
        query: str,
        lexical_query: str,
        world_id: str,
        session_id: str,
        persona_id: str,
        scene_id: str | None,
        top_k: int,
    ) -> tuple[list[RetrievedChunk], RetrievalDiagnostics | None]:
        retriever = self.actor_context_retriever
        assert retriever is not None
        with_diagnostics = getattr(retriever, "retrieve_for_actor_with_diagnostics", None)
        if callable(with_diagnostics):
            result: RetrievalResult = with_diagnostics(
                query=query,
                lexical_query=lexical_query,
                world_id=world_id,
                session_id=session_id,
                persona_id=persona_id,
                scene_id=scene_id,
                top_k=top_k,
            )
            return list(result.chunks), result.diagnostics
        chunks = retriever.retrieve_for_actor(
            query=query,
            lexical_query=lexical_query,
            world_id=world_id,
            session_id=session_id,
            persona_id=persona_id,
            scene_id=scene_id,
            top_k=top_k,
        )
        return chunks, None


def _to_turn_diagnostics(diagnostics: RetrievalDiagnostics) -> TurnRetrievalDiagnostics:
    return TurnRetrievalDiagnostics(
        query=diagnostics.query,
        selected=[_to_candidate(entry) for entry in diagnostics.selected],
        rejected=[_to_candidate(entry) for entry in diagnostics.rejected],
    )


def _to_candidate(entry: ChunkRetrievalDiagnostic) -> RetrievalCandidateDiagnostic:
    return RetrievalCandidateDiagnostic(
        id=entry.id,
        source=entry.source,
        source_type=entry.source_type,
        collection=entry.collection.value,
        visibility=entry.visibility,
        tags=entry.tags,
        original_score=entry.original_score,
        adjusted_score=entry.adjusted_score,
        applied_boosts=entry.applied_boosts,
        selected_rank=entry.selected_rank,
        slice_score=entry.slice_score,
        slice_matched_terms=entry.slice_matched_terms,
        slice_guaranteed=entry.slice_guaranteed,
    )
