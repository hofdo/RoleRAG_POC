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


class CreateSessionResponse(BaseModel):
    session_id: str
    world_id: str
    active_scene_id: str
    active_persona_id: str


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
    request_cloud: bool = False
    cloud_confirmed: bool = False
    force_local: bool = False


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
    text: str
    route: RouteResponse
    finish_reason: str | None = None
    memory_written: bool
    critic_status: str
    warnings: list[str]
    errors: list[TurnError] = Field(default_factory=list)
    retrieval: RetrievalDiagnosticsResponse | None = None
    stage_timings: dict[str, float] = Field(default_factory=dict)


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


class StreamFailurePayload(StreamFinalPayload):
    text: str


class StreamConfirmationPayload(BaseModel):
    status: str = "confirmation_required"
    route: RouteResponse
    warnings: list[str]
    errors: list[TurnError] = Field(default_factory=list)


class RecentTurnResponse(BaseModel):
    turn_index: int
    user_message: str
    assistant_message: str
    created_at: datetime


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
