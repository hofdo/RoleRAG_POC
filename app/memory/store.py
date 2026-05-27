from __future__ import annotations

from app.domain import MemoryCandidate, MemoryEpisode, StoredTurn
from app.persistence.repositories import MemoryRepository, TurnRepository


class RecentDialogueStore:
    def __init__(self, *, turn_repository: TurnRepository, recent_turns: int) -> None:
        self.turn_repository = turn_repository
        self.recent_turns = recent_turns

    def load_recent_dialogue(self, session_id: str) -> list[StoredTurn]:
        return self.turn_repository.list_recent_turns(session_id, self.recent_turns)


class MemoryEpisodeStore:
    def __init__(self, *, memory_repository: MemoryRepository) -> None:
        self.memory_repository = memory_repository

    def persist_memories(
        self,
        *,
        session_id: str,
        memories: list[MemoryCandidate],
    ) -> list[MemoryEpisode]:
        return self.memory_repository.append_memories(
            session_id=session_id,
            memories=memories,
        )
