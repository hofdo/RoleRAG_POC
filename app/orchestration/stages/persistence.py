from __future__ import annotations

from dataclasses import dataclass

from app.domain import SessionState, StoredTurn, TurnOutcome
from app.llm.router import ModelRoute
from app.persistence.repositories import SessionRepository, TurnRepository


@dataclass(frozen=True)
class PersistenceStageResult:
    turn: StoredTurn


class TurnPersistenceStage:
    def __init__(
        self,
        *,
        session_repository: SessionRepository,
        turn_repository: TurnRepository,
    ) -> None:
        self.session_repository = session_repository
        self.turn_repository = turn_repository

    def run(
        self,
        *,
        session: SessionState,
        user_message: str,
        assistant_message: str,
        route: ModelRoute,
        outcome: TurnOutcome = TurnOutcome.SUCCESS,
    ) -> PersistenceStageResult:
        turn = self.turn_repository.append_turn(
            session_id=session.id,
            scene_id=session.active_scene_id,
            persona_id=session.active_persona_id,
            user_message=user_message,
            assistant_message=assistant_message,
            route=route,
            outcome=outcome,
        )
        self.session_repository.update_session_activity(
            session.id,
            updated_at=turn.created_at,
        )
        return PersistenceStageResult(turn=turn)
