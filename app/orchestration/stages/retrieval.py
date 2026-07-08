from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.domain import (
    RetrievalCandidateDiagnostic,
    RetrievedChunk,
    TurnInput,
    TurnRetrievalDiagnostics,
)
from app.domain.visibility import Visibility
from app.orchestration.context_budget import ContextBudget
from app.orchestration.stages.session import LoadedTurnContext
from app.rag.diagnostics import ChunkRetrievalDiagnostic, RetrievalDiagnostics, RetrievalResult
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
    ) -> None:
        self.actor_context_retriever = actor_context_retriever
        self.context_budget = context_budget

    def run(
        self,
        *,
        turn_input: TurnInput,
        context: LoadedTurnContext,
    ) -> RetrievalStageResult:
        if self.actor_context_retriever is None:
            return RetrievalStageResult(chunks=(), confidence=None, warnings=())

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
            chunks, diagnostics = self._retrieve(
                query=query,
                lexical_query=turn_input.message,
                world_id=context.session.world_id,
                session_id=context.session.id,
                persona_id=context.persona.id,
                scene_id=context.session.active_scene_id,
                top_k=self.context_budget.retrieved_chunks + len(context.standing_facts),
            )
            scores = [
                chunk.score for chunk in chunks if chunk.visibility == Visibility.PLAYER
            ]
            confidence = max(scores) if scores else 0.0
            return RetrievalStageResult(
                chunks=tuple(chunks),
                confidence=confidence,
                warnings=(),
                diagnostics=diagnostics,
            )
        except Exception as exc:
            return RetrievalStageResult(
                chunks=(),
                confidence=None,
                warnings=(f"retrieval skipped: {exc}",),
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
    ) -> tuple[list[RetrievedChunk], TurnRetrievalDiagnostics | None]:
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
            return list(result.chunks), _to_turn_diagnostics(result.diagnostics)
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
    )
