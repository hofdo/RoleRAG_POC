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


class TurnOutcome(str, Enum):
    SUCCESS = "success"
    CONTROLLED_FAILURE = "controlled_failure"
    CONFIRMATION_REQUIRED = "confirmation_required"


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
    diagnostics: "TurnDiagnostics | None" = None
    # CONFIRMATION_REQUIRED turns are never persisted; stored rows are
    # SUCCESS or CONTROLLED_FAILURE.
    outcome: TurnOutcome = TurnOutcome.SUCCESS


class CriticStatus(str, Enum):
    """How critic validation concluded for the returned text.

    SKIPPED means the critic did not run -- either the gate judged the turn low-risk (and the
    draft is served unvalidated by design), or there was no draft to validate (CONFIRMATION_REQUIRED
    or a generation failure before critique). A critic that ERRORS no longer maps to SKIPPED: it
    fails the turn closed (REJECTED + CONTROLLED_FAILURE) rather than serving unvalidated text.
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


class TurnDiagnostics(BaseModel):
    """Persisted diagnostic record for one completed turn; mirrors the live TurnResult."""

    retrieval: TurnRetrievalDiagnostics | None = None
    stage_timings: dict[str, float] = Field(default_factory=dict)
    critic_status: CriticStatus
    finish_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)
    memory_written: bool


class DeferredMemoryJob(BaseModel):
    """Inputs the memory stage needs when it runs after the response is sent.

    scene_id/persona_id pin the job to the scene and persona the turn was actually
    generated under -- NOT whatever the session's live fields say when the job runs.
    A scene or persona switch that lands between the response and the deferred job
    executing must not cause the job to curate/attribute memories to the new
    scene/persona instead of the turn's original one.
    """

    session_id: str
    turn_id: int
    scene_id: str
    persona_id: str
    user_message: str
    assistant_message: str
    retrieval_confidence: float | None
    scene_complexity: int


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
    deferred_memory: "DeferredMemoryJob | None" = Field(default=None, exclude=True)


class TurnError(BaseModel):
    """Structured view of a turn warning: a consumer can branch on category/stage instead of
    string-matching free-form text. Derived from the warning strings (which are preserved), so
    this adds structure without changing how stages report problems."""

    category: str  # kind of problem: degraded, gated, fallback, security, provider_unavailable...
    stage: str  # pipeline stage: retrieval, generation, critique, repair, memory, containment...
    message: str  # the original warning text
    suggestion: str | None = None  # optional remediation hint


class CriticResult(BaseModel):
    accepted: bool
    issues: list[str] = Field(default_factory=list)
    repair_instruction: str | None = None


# StoredTurn forward-references TurnDiagnostics (defined later); resolve it now.
StoredTurn.model_rebuild()
