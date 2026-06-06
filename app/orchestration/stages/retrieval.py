from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.domain import RetrievedChunk, TurnInput
from app.domain.visibility import Visibility
from app.orchestration.context_budget import ContextBudget
from app.orchestration.stages.session import LoadedTurnContext
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
    ) -> list[RetrievedChunk]: ...


@dataclass(frozen=True)
class RetrievalStageResult:
    chunks: tuple[RetrievedChunk, ...]
    confidence: float | None
    warnings: tuple[str, ...]


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
            chunks = self.actor_context_retriever.retrieve_for_actor(
                query=query,
                world_id=context.session.world_id,
                session_id=context.session.id,
                persona_id=context.persona.id,
                scene_id=context.session.active_scene_id,
                top_k=self.context_budget.retrieved_chunks,
            )
            scores = [
                chunk.score for chunk in chunks if chunk.visibility == Visibility.PLAYER
            ]
            confidence = max(scores) if scores else 0.0
            return RetrievalStageResult(
                chunks=tuple(chunks),
                confidence=confidence,
                warnings=(),
            )
        except Exception as exc:
            return RetrievalStageResult(
                chunks=(),
                confidence=None,
                warnings=(f"retrieval skipped: {exc}",),
            )
