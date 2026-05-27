from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.domain.visibility import Visibility
from app.llm.router import ModelRoute


class PersonaCard(BaseModel):
    id: str
    name: str
    role: Literal["narrator", "npc", "companion", "antagonist"]
    public_description: str
    private_description: str | None = None
    speaking_style: str
    values: list[str] = Field(default_factory=list)
    fears: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    secrets: list[str] = Field(default_factory=list)
    forbidden_knowledge: list[str] = Field(default_factory=list)
    relationships: dict[str, str] = Field(default_factory=dict)


class SceneState(BaseModel):
    id: str
    title: str
    location: str
    current_time: str | None = None
    active_personas: list[str] = Field(default_factory=list)
    player_visible_summary: str
    gm_private_summary: str | None = None
    open_conflicts: list[str] = Field(default_factory=list)
    active_quests: list[str] = Field(default_factory=list)
    recent_events: list[str] = Field(default_factory=list)


class SessionState(BaseModel):
    id: str
    world_id: str
    active_scene_id: str
    player_name: str
    recent_turn_ids: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class MemoryEpisode(BaseModel):
    id: str
    session_id: str
    scene_id: str
    actor_id: str | None = None
    summary: str
    importance: int = Field(ge=1, le=5)
    visibility: Visibility
    tags: list[str] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    id: str
    source: str
    text: str
    score: float
    visibility: Visibility
    tags: list[str] = Field(default_factory=list)
    source_type: str | None = None
    world_id: str | None = None
    scene_id: str | None = None
    persona_id: str | None = None
    session_id: str | None = None


class TurnInput(BaseModel):
    session_id: str
    message: str
    active_persona_id: str


class TurnResult(BaseModel):
    text: str
    route: ModelRoute
    memory_written: bool = False
    warnings: list[str] = Field(default_factory=list)


class CriticResult(BaseModel):
    accepted: bool
    issues: list[str] = Field(default_factory=list)
    repair_instruction: str | None = None
