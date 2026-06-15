from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

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
    active_persona_id: str
    player_name: str
    content_root: str = "data"
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
    created_at: datetime | None = None


class CanonFact(BaseModel):
    id: str
    session_id: str
    text: str
    created_at: datetime | None = None


class MemoryCandidate(BaseModel):
    summary: str
    visibility: Visibility
    importance: int = Field(ge=1, le=5)
    tags: list[str] = Field(default_factory=list)
    scene_id: str | None = None
    actor_id: str | None = None


class MemoryCuratorResult(BaseModel):
    write_memory: bool
    memories: list[MemoryCandidate] = Field(default_factory=list)
    reason: str

    @model_validator(mode="after")
    def validate_write_memory_has_candidates(self) -> "MemoryCuratorResult":
        # A grammar-constrained local model can claim write_memory=true while
        # emitting no candidates; treat that contradiction as a decline rather
        # than failing the whole curation (the deterministic extractor still
        # preserves explicit player commitments).
        if self.write_memory and not self.memories:
            object.__setattr__(self, "write_memory", False)
        return self


class RetrievedChunk(BaseModel):
    id: str
    source: str
    source_type: str
    text: str
    score: float
    visibility: Visibility
    tags: list[str] = Field(default_factory=list)
    world_id: str | None = None
    scene_id: str | None = None
    persona_id: str | None = None
    session_id: str | None = None
    actor_id: str | None = None
    importance: int | None = Field(default=None, ge=1, le=5)
    created_at: datetime | None = None


class TurnInput(BaseModel):
    session_id: str
    message: str
    active_persona_id: str | None = None
    user_requested_cloud: bool = False
    cloud_confirmed: bool = False
    force_local: bool = False


class StoredTurn(BaseModel):
    id: int
    session_id: str
    turn_index: int
    scene_id: str
    persona_id: str
    user_message: str
    assistant_message: str
    route: ModelRoute
    created_at: datetime


class TurnOutcome(str, Enum):
    SUCCESS = "success"
    CONTROLLED_FAILURE = "controlled_failure"
    CONFIRMATION_REQUIRED = "confirmation_required"


class CriticStatus(str, Enum):
    """How critic validation concluded for the returned text.

    SKIPPED means the text was never successfully validated: the critic errored,
    or the turn failed before critique ran.
    """

    ACCEPTED = "accepted"
    REPAIRED = "repaired"
    REJECTED = "rejected"
    SKIPPED = "skipped"


class RetrievalCandidateDiagnostic(BaseModel):
    """Metadata-only ranking record for one retrieval candidate; never carries chunk text."""

    id: str
    source: str
    source_type: str
    collection: str
    visibility: Visibility
    tags: list[str] = Field(default_factory=list)
    original_score: float
    adjusted_score: float
    applied_boosts: dict[str, float] = Field(default_factory=dict)
    selected_rank: int | None = Field(default=None, ge=1)


class TurnRetrievalDiagnostics(BaseModel):
    query: str
    selected: list[RetrievalCandidateDiagnostic] = Field(default_factory=list)
    rejected: list[RetrievalCandidateDiagnostic] = Field(default_factory=list)


class TurnResult(BaseModel):
    text: str
    route: ModelRoute
    finish_reason: str | None = None
    memory_written: bool = False
    critic_status: CriticStatus = CriticStatus.SKIPPED
    warnings: list[str] = Field(default_factory=list)
    retrieval: TurnRetrievalDiagnostics | None = None
    stage_timings: dict[str, float] = Field(default_factory=dict)
    outcome: TurnOutcome = Field(default=TurnOutcome.SUCCESS, exclude=True)


class CriticResult(BaseModel):
    accepted: bool
    issues: list[str] = Field(default_factory=list)
    repair_instruction: str | None = None
