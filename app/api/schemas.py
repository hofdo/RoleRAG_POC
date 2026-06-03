from __future__ import annotations

from datetime import datetime
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict, Field

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


class CreateTurnRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1)
    active_persona_id: str | None = Field(default=None, min_length=1)
    request_cloud: bool = False


class RouteResponse(BaseModel):
    provider: str
    model: str
    reason: str


class CreateTurnResponse(BaseModel):
    text: str
    route: RouteResponse
    memory_written: bool
    warnings: list[str]


class StreamTextPayload(BaseModel):
    text: str


class StreamFinalPayload(BaseModel):
    route: RouteResponse
    memory_written: bool
    warnings: list[str]


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
