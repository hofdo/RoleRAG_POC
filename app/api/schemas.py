"""Pydantic request/response models for the HTTP API boundary.

Defines the wire shapes ``app.api.routes`` validates and serializes: session
create/get, scene update, turn create/detail, last-turn deletion, session
memories, eval-run summaries, and the error envelope. Request models pin
``extra="forbid"`` and field bounds so malformed input fails with a 422 before
reaching the orchestrator; response models are deliberately narrower than the
domain objects, omitting hidden persona/scene content. These types are the
source of the generated OpenAPI schema.
"""

from __future__ import annotations

from datetime import datetime
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from app.domain import TurnError, TurnRetrievalDiagnostics

ErrorLocation: TypeAlias = str | int


class ErrorDetail(BaseModel):
    loc: list[ErrorLocation]
    type: str
    message: str


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[ErrorDetail]


class ErrorResponse(BaseModel):
    error: ErrorBody


class CreateSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    world_id: str = Field(min_length=1, max_length=200)
    scene_id: str = Field(min_length=1, max_length=200)
    player_name: str = Field(min_length=1, max_length=200)
    active_persona_id: str = Field(min_length=1, max_length=200)
    provider: str = Field(default="local", pattern="^(local|cloud)$")
    # Additive (#16 follow-up): mirrors the CLI's `--skip-lore-ingest`. Default False preserves
    # the new auto-ingest-on-create behavior for callers (the SPA included) that don't set it.
    skip_lore_ingest: bool = False


class CreateSessionResponse(BaseModel):
    session_id: str
    world_id: str
    active_scene_id: str
    active_persona_id: str
    provider: str
    # Additive (#16 follow-up): best-effort scenario-lore auto-ingest failures degrade to a
    # warning here instead of failing session creation (parity with the CLI's `start-session`,
    # which prints the same warning). Empty when ingest was skipped or succeeded.
    warnings: list[str] = Field(default_factory=list)


class UpdateSceneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_id: str = Field(min_length=1, max_length=200)


class RecentSessionResponse(BaseModel):
    session_id: str
    world_id: str
    active_scene_id: str
    active_persona_id: str
    player_name: str
    created_at: datetime
    updated_at: datetime


class RecentSessionsResponse(BaseModel):
    sessions: list[RecentSessionResponse]


class CatalogWorldResponse(BaseModel):
    id: str
    name: str
    default_scene_id: str
    scene_ids: list[str]
    persona_ids: list[str]


class CatalogSceneResponse(BaseModel):
    id: str
    title: str
    location: str
    player_visible_summary: str
    active_personas: list[str]


class CatalogPersonaResponse(BaseModel):
    id: str
    name: str
    role: str
    public_description: str
    speaking_style: str


class ContentCatalogResponse(BaseModel):
    worlds: list[CatalogWorldResponse]
    scenes: list[CatalogSceneResponse]
    personas: list[CatalogPersonaResponse]


class RuntimeStatusResponse(BaseModel):
    app_name: str
    app_version: str
    environment: str
    cloud_mode: str
    retrieval_configured: bool
    content_catalog_available: bool
    local_provider_configured: bool
    cloud_provider_configured: bool


class CreateTurnRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=4000)
    active_persona_id: str | None = Field(default=None, min_length=1, max_length=200)


class RouteResponse(BaseModel):
    provider: str
    model: str
    reason: str


class RetrievalCandidateResponse(BaseModel):
    id: str
    source: str
    source_type: str
    collection: str
    visibility: str
    tags: list[str] = Field(default_factory=list)
    original_score: float
    adjusted_score: float
    applied_boosts: dict[str, float] = Field(default_factory=dict)
    selected_rank: int | None = None
    # Lane B lexical slice labels (docs/26 §3.4, #79); additive, no-slice defaults so
    # a slice pick is never a silent cosine mystery in the inspection payload.
    slice_score: float | None = None
    slice_matched_terms: list[str] = Field(default_factory=list)
    slice_guaranteed: bool = False


class RetrievalDiagnosticsResponse(BaseModel):
    query: str
    selected: list[RetrievalCandidateResponse] = Field(default_factory=list)
    rejected: list[RetrievalCandidateResponse] = Field(default_factory=list)


def to_retrieval_diagnostics_response(
    diagnostics: TurnRetrievalDiagnostics | None,
) -> RetrievalDiagnosticsResponse | None:
    if diagnostics is None:
        return None
    return RetrievalDiagnosticsResponse.model_validate(diagnostics.model_dump(mode="json"))


class CreateTurnResponse(BaseModel):
    status: str = "completed"
    outcome: str = "success"
    text: str
    route: RouteResponse
    finish_reason: str | None = None
    memory_written: bool
    critic_status: str
    warnings: list[str]
    errors: list[TurnError] = Field(default_factory=list)
    retrieval: RetrievalDiagnosticsResponse | None = None
    stage_timings: dict[str, float] = Field(default_factory=dict)
    # Token usage from the generation that produced `text` (repair generation's usage
    # when a repair ran, otherwise the initial actor generation's). None when no
    # generation completed or the provider reported no usage (#69).
    token_usage: dict[str, int] | None = None
    # Standing-facts (Lane A canon pinning) diagnostics (docs/26 §3.3, #78): count and
    # total char length of this turn's pinned Standing-facts block, populated every
    # turn. Mirrors token_usage's optional/additive contract.
    standing_facts_count: int | None = None
    standing_facts_chars: int | None = None


class StreamTextPayload(BaseModel):
    text: str


class StreamFinalPayload(BaseModel):
    route: RouteResponse
    finish_reason: str | None = None
    memory_written: bool
    critic_status: str
    warnings: list[str]
    errors: list[TurnError] = Field(default_factory=list)
    retrieval: RetrievalDiagnosticsResponse | None = None
    stage_timings: dict[str, float] = Field(default_factory=dict)
    token_usage: dict[str, int] | None = None
    standing_facts_count: int | None = None
    standing_facts_chars: int | None = None


class StreamFailurePayload(StreamFinalPayload):
    text: str
    outcome: str = "controlled_failure"


class StreamStagePayload(BaseModel):
    stage: str


class StreamErrorPayload(BaseModel):
    code: str
    message: str
    status: int


class RecentTurnResponse(BaseModel):
    turn_index: int
    user_message: str
    assistant_message: str
    created_at: datetime


class DeleteLastTurnResponse(BaseModel):
    session_id: str
    deleted_turn_index: int
    user_message: str
    deleted_memory_count: int


class MemoryEpisodeResponse(BaseModel):
    id: str
    scene_id: str
    actor_id: str | None = None
    summary: str
    importance: int
    visibility: str
    tags: list[str] = Field(default_factory=list)


class SessionMemoriesResponse(BaseModel):
    session_id: str
    memories: list[MemoryEpisodeResponse]


class AddCanonFactRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)


class CanonFactResponse(BaseModel):
    id: str
    text: str


class CanonFactsResponse(BaseModel):
    session_id: str
    facts: list[CanonFactResponse]


class GetSessionResponse(BaseModel):
    session_id: str
    world_id: str
    active_scene_id: str
    active_persona_id: str
    recent_turns: list[RecentTurnResponse]


class TurnDetailResponse(BaseModel):
    turn_index: int
    outcome: str = "success"
    scene_id: str
    persona_id: str
    user_message: str
    assistant_message: str
    route: RouteResponse
    created_at: datetime
    finish_reason: str | None = None
    memory_written: bool
    critic_status: str
    warnings: list[str]
    errors: list[TurnError] = Field(default_factory=list)
    stage_timings: dict[str, float] = Field(default_factory=dict)
    retrieval: RetrievalDiagnosticsResponse | None = None
    token_usage: dict[str, int] | None = None
    standing_facts_count: int | None = None
    standing_facts_chars: int | None = None


class EvalRunSummary(BaseModel):
    id: str
    status: str
    turn_count: int
    recall_misses: int
    extraction_misses: int
    retrieval_misses: int
    total_seconds: float
    p50_seconds: float
    p95_seconds: float
    warning_total: int


class EvalRunsResponse(BaseModel):
    results_dir: str
    runs: list[EvalRunSummary] = Field(default_factory=list)


class SessionTurnDetailsResponse(BaseModel):
    session_id: str
    turns: list[TurnDetailResponse] = Field(default_factory=list)
