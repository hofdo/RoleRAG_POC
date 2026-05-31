from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    world_id: str = Field(min_length=1)
    scene_id: str = Field(min_length=1)
    player_name: str = Field(min_length=1)
    active_persona_id: str = Field(min_length=1)


class CreateSessionResponse(BaseModel):
    session_id: str
    world_id: str
    active_scene_id: str
    active_persona_id: str


class CreateTurnRequest(BaseModel):
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
