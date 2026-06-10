from __future__ import annotations

from datetime import datetime
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from app.domain import TurnRetrievalDiagnostics

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

    world_id: str = Field(min_length=1)
    scene_id: str = Field(min_length=1)
    player_name: str = Field(min_length=1)
    active_persona_id: str = Field(min_length=1)


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

    message: str = Field(min_length=1)
    active_persona_id: str | None = Field(default=None, min_length=1)
    request_cloud: bool = False


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
    text: str
    route: RouteResponse
    finish_reason: str | None = None
    memory_written: bool
    warnings: list[str]
    retrieval: RetrievalDiagnosticsResponse | None = None


class StreamTextPayload(BaseModel):
    text: str


class StreamFinalPayload(BaseModel):
    route: RouteResponse
    finish_reason: str | None = None
    memory_written: bool
    warnings: list[str]
    retrieval: RetrievalDiagnosticsResponse | None = None


class StreamFailurePayload(StreamFinalPayload):
    text: str


class RecentTurnResponse(BaseModel):
    turn_index: int
    user_message: str
    assistant_message: str
    created_at: datetime


class GetSessionResponse(BaseModel):
    session_id: str
    world_id: str
    active_scene_id: str
    active_persona_id: str
    recent_turns: list[RecentTurnResponse]
