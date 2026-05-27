from __future__ import annotations

from app.domain import StoredTurn
from app.persistence.repositories import TurnRepository


class RecentDialogueStore:
    def __init__(self, *, turn_repository: TurnRepository, recent_turns: int) -> None:
        self.turn_repository = turn_repository
        self.recent_turns = recent_turns

    def load_recent_dialogue(self, session_id: str) -> list[StoredTurn]:
        return self.turn_repository.list_recent_turns(session_id, self.recent_turns)
