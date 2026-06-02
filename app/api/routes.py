from __future__ import annotations

from collections.abc import Generator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, status

from app.api.errors import ApiError
from app.api.schemas import (
    CreateSessionRequest,
    CreateSessionResponse,
    CreateTurnRequest,
    CreateTurnResponse,
    ErrorResponse,
    GetSessionResponse,
    RecentTurnResponse,
    RouteResponse,
)
from app.composition import AppServices, build_services
from app.config import Settings, get_settings
from app.domain import TurnInput, TurnResult
from app.persistence import DataFileNotFoundError, DataValidationError, SessionNotFoundError

router = APIRouter()

ERROR_400_RESPONSE: dict[int | str, dict[str, Any]] = {400: {"model": ErrorResponse}}
ERROR_404_RESPONSE: dict[int | str, dict[str, Any]] = {404: {"model": ErrorResponse}}
ERROR_422_RESPONSE: dict[int | str, dict[str, Any]] = {422: {"model": ErrorResponse}}


def get_read_services(
    settings: Annotated[Settings, Depends(get_settings)],
) -> Generator[AppServices, None, None]:
    services = build_services(settings, enable_retrieval=False)
    try:
        yield services
    finally:
        services.close()


def get_turn_services(
    settings: Annotated[Settings, Depends(get_settings)],
) -> Generator[AppServices, None, None]:
    services = build_services(settings, enable_retrieval=True)
    try:
        yield services
    finally:
        services.close()


@router.post(
    "/sessions",
    response_model=CreateSessionResponse,
    status_code=status.HTTP_201_CREATED,
    responses={**ERROR_400_RESPONSE, **ERROR_422_RESPONSE},
)
def create_session(
    request: CreateSessionRequest,
    services: Annotated[AppServices, Depends(get_read_services)],
) -> CreateSessionResponse:
    try:
        session = services.orchestrator.create_session(
            world_id=request.world_id,
            scene_id=request.scene_id,
            active_persona_id=request.active_persona_id,
            player_name=request.player_name,
        )
    except (DataFileNotFoundError, DataValidationError, ValueError) as exc:
        raise ApiError(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="invalid_session_request",
            message=_safe_request_error_message(exc),
        ) from exc
    return CreateSessionResponse(
        session_id=session.id,
        world_id=session.world_id,
        active_scene_id=session.active_scene_id,
        active_persona_id=session.active_persona_id,
    )


@router.post(
    "/sessions/{session_id}/turns",
    response_model=CreateTurnResponse,
    responses={**ERROR_400_RESPONSE, **ERROR_404_RESPONSE, **ERROR_422_RESPONSE},
)
async def create_turn(
    session_id: str,
    request: CreateTurnRequest,
    services: Annotated[AppServices, Depends(get_turn_services)],
) -> CreateTurnResponse:
    try:
        result = await services.orchestrator.run_turn(
            turn_input=TurnInput(
                session_id=session_id,
                message=request.message,
                active_persona_id=request.active_persona_id,
                user_requested_cloud=request.request_cloud,
            )
        )
    except SessionNotFoundError as exc:
        raise ApiError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="session_not_found",
            message=str(exc),
        ) from exc
    except (DataFileNotFoundError, DataValidationError, ValueError) as exc:
        raise ApiError(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="invalid_turn_request",
            message=_safe_request_error_message(exc),
        ) from exc
    return _to_turn_response(result)


@router.get(
    "/sessions/{session_id}",
    response_model=GetSessionResponse,
    responses=ERROR_404_RESPONSE,
)
def get_session(
    session_id: str,
    services: Annotated[AppServices, Depends(get_read_services)],
) -> GetSessionResponse:
    try:
        session = services.orchestrator.resume_session(session_id)
    except SessionNotFoundError as exc:
        raise ApiError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="session_not_found",
            message=str(exc),
        ) from exc
    recent_turns = [
        RecentTurnResponse(
            turn_index=turn.turn_index,
            user_message=turn.user_message,
            assistant_message=turn.assistant_message,
            created_at=turn.created_at,
        )
        for turn in services.recent_dialogue_store.load_recent_dialogue(session.id)
    ]
    return GetSessionResponse(
        session_id=session.id,
        world_id=session.world_id,
        active_scene_id=session.active_scene_id,
        active_persona_id=session.active_persona_id,
        recent_turns=recent_turns,
    )


def _to_turn_response(result: TurnResult) -> CreateTurnResponse:
    return CreateTurnResponse(
        text=result.text,
        route=RouteResponse(
            provider=result.route.provider.value,
            model=result.route.model,
            reason=result.route.reason,
        ),
        memory_written=result.memory_written,
        warnings=result.warnings,
    )


def _safe_request_error_message(
    exc: DataFileNotFoundError | DataValidationError | ValueError,
) -> str:
    if isinstance(exc, DataFileNotFoundError):
        return f"Unknown {exc.entity_type} id: {exc.entity_id}"
    if isinstance(exc, DataValidationError):
        return f"Invalid {exc.entity_type} id: {exc.entity_id}"
    return str(exc)
