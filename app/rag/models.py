from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from app.domain import Visibility


class RagCollection(str, Enum):
    CANON_LORE = "canon_lore"
    SESSION_MEMORY = "session_memory"
    PERSONA_MEMORY = "persona_memory"


class RagChunk(BaseModel):
    id: str
    source: str
    source_type: str
    text: str = Field(min_length=1)
    visibility: Visibility
    tags: list[str] = Field(default_factory=list)
    world_id: str | None = None
    scene_id: str | None = None
    persona_id: str | None = None
    session_id: str | None = None
    actor_id: str | None = None
    importance: int | None = Field(default=None, ge=1, le=5)


class RetrievalFilter(BaseModel):
    allowed_visibilities: list[Visibility] = Field(min_length=1)
    world_id: str | None = None
    scene_id: str | None = None
    persona_id: str | None = None
    session_id: str | None = None
    tags: list[str] = Field(default_factory=list)

    @classmethod
    def player_visible(
        cls,
        *,
        world_id: str | None = None,
        scene_id: str | None = None,
        persona_id: str | None = None,
        session_id: str | None = None,
        tags: list[str] | None = None,
    ) -> "RetrievalFilter":
        return cls(
            allowed_visibilities=[Visibility.PLAYER],
            world_id=world_id,
            scene_id=scene_id,
            persona_id=persona_id,
            session_id=session_id,
            tags=tags or [],
        )
