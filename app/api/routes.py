from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Generator
from contextlib import suppress
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Response, status
from fastapi.responses import StreamingResponse

from app import __version__
from app.api.errors import ApiError
from app.api.schemas import (
    AddCanonFactRequest,
    CanonFactResponse,
    CanonFactsResponse,
    CatalogPersonaResponse,
    CatalogSceneResponse,
    CatalogWorldResponse,
    ContentCatalogResponse,
    CreateSessionRequest,
    CreateSessionResponse,
    CreateTurnRequest,
    CreateTurnResponse,
    ErrorResponse,
    EvalRunsResponse,
    EvalRunSummary,
    GetSessionResponse,
    MemoryEpisodeResponse,
    RecentSessionResponse,
    RecentSessionsResponse,
    RecentTurnResponse,
    RouteResponse,
    RuntimeStatusResponse,
    SessionMemoriesResponse,
    SessionTurnDetailsResponse,
    TurnDetailResponse,
    to_retrieval_diagnostics_response,
)
from app.api.sse import build_turn_stream_frames, serialize_error_frame, serialize_stage_frame
from app.composition import AppServices, build_file_loader, build_services
from app.config import Settings, get_settings, is_usable_cloud_api_key
from app.diagnostics.eval_runs import load_eval_run, load_eval_runs
from app.domain import SessionState, StoredTurn, TurnInput, TurnOutcome, TurnResult
from app.llm.provider import ProviderTimeoutError, ProviderUnavailableError
from app.orchestration.turn_errors import classify_warnings
from app.persistence import (
    ContentCatalogError,
    DataFileNotFoundError,
    DataValidationError,
    SessionNotFoundError,
)

router = APIRouter()
RECENT_SESSIONS_LIMIT = 10

ERROR_400_RESPONSE: dict[int | str, dict[str, Any]] = {400: {"model": ErrorResponse}}
ERROR_404_RESPONSE: dict[int | str, dict[str, Any]] = {404: {"model": ErrorResponse}}
ERROR_422_RESPONSE: dict[int | str, dict[str, Any]] = {422: {"model": ErrorResponse}}
ERROR_503_RESPONSE: dict[int | str, dict[str, Any]] = {503: {"model": ErrorResponse}}
ERROR_504_RESPONSE: dict[int | str, dict[str, Any]] = {504: {"model": ErrorResponse}}
SSE_200_RESPONSE: dict[int | str, dict[str, Any]] = {
    200: {
        "description": "Buffered validated turn events",
        "content": {"text/event-stream": {"schema": {"type": "string"}}},
    }
}


def _has_text(value: str) -> bool:
    return bool(value.strip())


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


@router.get("/runtime/status", response_model=RuntimeStatusResponse)
def get_runtime_status(
    settings: Annotated[Settings, Depends(get_settings)],
) -> RuntimeStatusResponse:
    try:
        build_file_loader(settings.content_root).load_catalog()
        content_catalog_available = True
    except ContentCatalogError:
        content_catalog_available = False

    return RuntimeStatusResponse(
        app_name="rolerag-poc",
        app_version=__version__,
        environment=settings.app_env,
        cloud_mode=settings.cloud_mode.value,
        retrieval_configured=_has_text(settings.qdrant_url) and _has_text(settings.embedding_model),
        content_catalog_available=content_catalog_available,
        local_provider_configured=(
            _has_text(settings.local_llm_base_url)
            and _has_text(settings.local_llm_api_key)
            and _has_text(settings.local_llm_model)
        ),
        cloud_provider_configured=(
            _has_text(settings.cloud_llm_base_url)
            and _has_text(settings.cloud_llm_model)
            and is_usable_cloud_api_key(settings.cloud_llm_api_key)
        ),
    )


@router.get("/diagnostics/eval-runs", response_model=EvalRunsResponse)
def get_eval_runs() -> EvalRunsResponse:
    # Read-only: scans EVAL_RESULTS_DIR for per-run conversation-checkpoint.json artifacts.
    base, runs = load_eval_runs()
    return EvalRunsResponse(
        results_dir=str(base),
        runs=[EvalRunSummary(**run) for run in runs],
    )


@router.get("/diagnostics/eval-runs/{run_id}")
def get_eval_run(run_id: str) -> dict[str, Any]:
    # Read-only: full conversation-checkpoint.json for one run (drill-down).
    payload = load_eval_run(run_id)
    if payload is None:
        raise ApiError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="eval_run_not_found",
            message=f"Unknown eval run: {run_id}",
        )
    return payload


@router.get(
    "/content/catalog",
    response_model=ContentCatalogResponse,
    responses=ERROR_400_RESPONSE,
)
def get_content_catalog(
    settings: Annotated[Settings, Depends(get_settings)],
) -> ContentCatalogResponse:
    try:
        catalog = build_file_loader(settings.content_root).load_catalog()
    except ContentCatalogError as exc:
        raise ApiError(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="invalid_content_catalog",
            message=exc.message,
        ) from exc
    return ContentCatalogResponse(
        worlds=[
            CatalogWorldResponse(
                id=world.id,
                name=world.name,
                default_scene_id=world.default_scene_id,
                scene_ids=world.scene_ids,
                persona_ids=world.persona_ids,
            )
            for world in catalog.worlds
        ],
        scenes=[
            CatalogSceneResponse(
                id=scene.id,
                title=scene.title,
                location=scene.location,
                player_visible_summary=scene.player_visible_summary,
                active_personas=scene.active_personas,
            )
            for scene in catalog.scenes
        ],
        personas=[
            CatalogPersonaResponse(
                id=persona.id,
                name=persona.name,
                role=persona.role,
                public_description=persona.public_description,
                speaking_style=persona.speaking_style,
            )
            for persona in catalog.personas
        ],
    )


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


@router.get("/sessions", response_model=RecentSessionsResponse)
def list_recent_sessions(
    services: Annotated[AppServices, Depends(get_read_services)],
) -> RecentSessionsResponse:
    return RecentSessionsResponse(
        sessions=[
            _to_recent_session_response(session)
            for session in services.session_repository.list_recent_sessions(
                RECENT_SESSIONS_LIMIT
            )
        ]
    )


@router.post(
    "/sessions/{session_id}/turns",
    response_model=CreateTurnResponse,
    responses={
        **ERROR_400_RESPONSE,
        **ERROR_404_RESPONSE,
        **ERROR_422_RESPONSE,
        **ERROR_503_RESPONSE,
        **ERROR_504_RESPONSE,
    },
)
async def create_turn(
    session_id: str,
    request: CreateTurnRequest,
    services: Annotated[AppServices, Depends(get_turn_services)],
) -> CreateTurnResponse:
    return _to_turn_response(await _run_turn(session_id, request, services))


@router.post(
    "/sessions/{session_id}/turns/stream",
    response_class=StreamingResponse,
    responses={**SSE_200_RESPONSE, **ERROR_422_RESPONSE},
)
async def stream_turn(
    session_id: str,
    request: CreateTurnRequest,
    services: Annotated[AppServices, Depends(get_turn_services)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> StreamingResponse:
    queue: asyncio.Queue[str] = asyncio.Queue()

    def on_stage(stage: str) -> None:
        queue.put_nowait(serialize_stage_frame(stage))

    async def event_stream() -> AsyncIterator[str]:
        turn = asyncio.create_task(
            _run_turn(session_id, request, services, on_stage=on_stage)
        )
        frame: asyncio.Task[str] | None = None
        try:
            try:
                while not turn.done():
                    frame = asyncio.create_task(queue.get())
                    done, _ = await asyncio.wait(
                        {frame, turn}, return_when=asyncio.FIRST_COMPLETED
                    )
                    if frame in done:
                        yield frame.result()
                    else:
                        frame.cancel()
                        with suppress(asyncio.CancelledError):
                            await frame
                    frame = None
                while not queue.empty():
                    yield queue.get_nowait()
                result = await turn
            except ApiError as exc:
                # The HTTP status is already 200 once streaming starts; errors must
                # travel as a terminal frame instead.
                yield serialize_error_frame(
                    code=exc.code, message=exc.message, status=exc.status_code
                )
                return
            for out in build_turn_stream_frames(
                result, text_chunk_chars=settings.sse_text_chunk_chars
            ):
                yield out
        finally:
            # If the client disconnects (or any other early exit closes this
            # generator) before the turn finished, cancel it so it doesn't run
            # orphaned while still holding request-scoped services (e.g. the
            # SQLite connection).
            if frame is not None and not frame.done():
                frame.cancel()
                with suppress(asyncio.CancelledError):
                    await frame
            if not turn.done():
                turn.cancel()
                with suppress(asyncio.CancelledError):
                    await turn

    return StreamingResponse(event_stream(), media_type="text/event-stream")


async def _run_turn(
    session_id: str,
    request: CreateTurnRequest,
    services: AppServices,
    *,
    on_stage: Callable[[str], None] | None = None,
) -> TurnResult:
    try:
        result = await services.orchestrator.run_turn(
            turn_input=TurnInput(
                session_id=session_id,
                message=request.message,
                active_persona_id=request.active_persona_id,
                user_requested_cloud=request.request_cloud,
                cloud_confirmed=request.cloud_confirmed,
                force_local=request.force_local,
            ),
            on_stage=on_stage,
        )
    except SessionNotFoundError as exc:
        raise ApiError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="session_not_found",
            message=str(exc),
        ) from exc
    except ProviderTimeoutError as exc:
        raise ApiError(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            code="provider_timeout",
            message=str(exc),
        ) from exc
    except ProviderUnavailableError as exc:
        raise ApiError(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            code="provider_unavailable",
            message=str(exc),
        ) from exc
    except (DataFileNotFoundError, DataValidationError, ValueError) as exc:
        raise ApiError(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="invalid_turn_request",
            message=_safe_request_error_message(exc),
        ) from exc
    return result


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


@router.get(
    "/sessions/{session_id}/turns/{turn_index}",
    response_model=TurnDetailResponse,
    responses=ERROR_404_RESPONSE,
)
def get_turn_detail(
    session_id: str,
    turn_index: int,
    services: Annotated[AppServices, Depends(get_read_services)],
) -> TurnDetailResponse:
    _require_session(services, session_id)
    if services.turn_repository is None:
        raise RuntimeError("Read services must include a turn repository")
    stored_turn = next(
        (
            turn
            for turn in services.turn_repository.list_all_turns(session_id)
            if turn.turn_index == turn_index
        ),
        None,
    )
    if stored_turn is None:
        raise ApiError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="turn_not_found",
            message=f"Unknown turn index: {turn_index}",
        )
    return _to_turn_detail_response(stored_turn)


@router.get(
    "/sessions/{session_id}/turn-details",
    response_model=SessionTurnDetailsResponse,
    responses=ERROR_404_RESPONSE,
)
def get_session_turn_details(
    session_id: str,
    services: Annotated[AppServices, Depends(get_read_services)],
) -> SessionTurnDetailsResponse:
    # One call returns every stored turn's full diagnostics, so Analytics/Inspector don't
    # fan out N requests over /turns/{i}.
    _require_session(services, session_id)
    if services.turn_repository is None:
        raise RuntimeError("Read services must include a turn repository")
    turns = sorted(
        services.turn_repository.list_all_turns(session_id),
        key=lambda turn: turn.turn_index,
    )
    return SessionTurnDetailsResponse(
        session_id=session_id,
        turns=[_to_turn_detail_response(turn) for turn in turns],
    )


@router.get(
    "/sessions/{session_id}/memories",
    response_model=SessionMemoriesResponse,
    responses=ERROR_404_RESPONSE,
)
def get_session_memories(
    session_id: str,
    services: Annotated[AppServices, Depends(get_read_services)],
) -> SessionMemoriesResponse:
    if services.session_repository.get_session(session_id) is None:
        raise ApiError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="session_not_found",
            message=f"Unknown session id: {session_id}",
        )
    # The memory viewer is a single-user authoring surface: it intentionally
    # returns ALL visibilities (PLAYER/GM/CHARACTER_PRIVATE) so the author can
    # inspect their own GM-only notes. Actor-facing leakage is prevented by
    # retrieval/prompt visibility filtering, not here. Pinned by
    # test_get_session_memories_returns_all_visibilities.
    episodes = (
        services.memory_repository.list_memories_for_session(session_id)
        if services.memory_repository is not None
        else []
    )
    return SessionMemoriesResponse(
        session_id=session_id,
        memories=[
            MemoryEpisodeResponse(
                id=episode.id,
                scene_id=episode.scene_id,
                actor_id=episode.actor_id,
                summary=episode.summary,
                importance=episode.importance,
                visibility=episode.visibility.value,
                tags=episode.tags,
            )
            for episode in episodes
        ],
    )


@router.get(
    "/sessions/{session_id}/canon",
    response_model=CanonFactsResponse,
    responses=ERROR_404_RESPONSE,
)
def get_session_canon(
    session_id: str,
    services: Annotated[AppServices, Depends(get_read_services)],
) -> CanonFactsResponse:
    _require_session(services, session_id)
    facts = (
        services.canon_repository.list_canon_facts(session_id)
        if services.canon_repository is not None
        else []
    )
    return CanonFactsResponse(
        session_id=session_id,
        facts=[CanonFactResponse(id=fact.id, text=fact.text) for fact in facts],
    )


@router.post(
    "/sessions/{session_id}/canon",
    response_model=CanonFactResponse,
    status_code=status.HTTP_201_CREATED,
    responses=ERROR_404_RESPONSE,
)
def add_session_canon(
    session_id: str,
    request: AddCanonFactRequest,
    services: Annotated[AppServices, Depends(get_read_services)],
) -> CanonFactResponse:
    _require_session(services, session_id)
    if services.canon_repository is None:
        raise ApiError(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            code="canon_unavailable",
            message="Canon repository is unavailable.",
        )
    fact = services.canon_repository.add_canon_fact(session_id=session_id, text=request.text)
    return CanonFactResponse(id=fact.id, text=fact.text)


@router.delete(
    "/sessions/{session_id}/canon/{fact_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses=ERROR_404_RESPONSE,
)
def delete_session_canon(
    session_id: str,
    fact_id: str,
    services: Annotated[AppServices, Depends(get_read_services)],
) -> Response:
    _require_session(services, session_id)
    deleted = (
        services.canon_repository.delete_canon_fact(session_id=session_id, fact_id=fact_id)
        if services.canon_repository is not None
        else False
    )
    if not deleted:
        raise ApiError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="canon_fact_not_found",
            message=f"Unknown canon fact id: {fact_id}",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _require_session(services: AppServices, session_id: str) -> None:
    if services.session_repository.get_session(session_id) is None:
        raise ApiError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="session_not_found",
            message=f"Unknown session id: {session_id}",
        )


def _to_recent_session_response(session: SessionState) -> RecentSessionResponse:
    if session.created_at is None or session.updated_at is None:
        raise RuntimeError("Persisted sessions must include timestamps")
    return RecentSessionResponse(
        session_id=session.id,
        world_id=session.world_id,
        active_scene_id=session.active_scene_id,
        active_persona_id=session.active_persona_id,
        player_name=session.player_name,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


def _to_turn_response(result: TurnResult) -> CreateTurnResponse:
    return CreateTurnResponse(
        status=(
            "confirmation_required"
            if result.outcome == TurnOutcome.CONFIRMATION_REQUIRED
            else "completed"
        ),
        text=result.text,
        route=RouteResponse(
            provider=result.route.provider.value,
            model=result.route.model,
            reason=result.route.reason,
        ),
        finish_reason=result.finish_reason,
        memory_written=result.memory_written,
        critic_status=result.critic_status.value,
        warnings=result.warnings,
        errors=classify_warnings(result.warnings),
        retrieval=to_retrieval_diagnostics_response(result.retrieval),
        stage_timings=result.stage_timings,
    )


def _to_turn_detail_response(turn: StoredTurn) -> TurnDetailResponse:
    diagnostics = turn.diagnostics
    return TurnDetailResponse(
        turn_index=turn.turn_index,
        scene_id=turn.scene_id,
        persona_id=turn.persona_id,
        user_message=turn.user_message,
        assistant_message=turn.assistant_message,
        route=RouteResponse(
            provider=turn.route.provider.value,
            model=turn.route.model,
            reason=turn.route.reason,
        ),
        created_at=turn.created_at,
        finish_reason=diagnostics.finish_reason if diagnostics is not None else None,
        memory_written=diagnostics.memory_written if diagnostics is not None else False,
        critic_status=diagnostics.critic_status.value if diagnostics is not None else "",
        warnings=diagnostics.warnings if diagnostics is not None else [],
        errors=classify_warnings(diagnostics.warnings if diagnostics is not None else []),
        stage_timings=diagnostics.stage_timings if diagnostics is not None else {},
        retrieval=(
            to_retrieval_diagnostics_response(diagnostics.retrieval)
            if diagnostics is not None
            else None
        ),
    )


def _safe_request_error_message(
    exc: DataFileNotFoundError | DataValidationError | ValueError,
) -> str:
    if isinstance(exc, DataFileNotFoundError):
        return f"Unknown {exc.entity_type} id: {exc.entity_id}"
    if isinstance(exc, DataValidationError):
        return f"Invalid {exc.entity_type} id: {exc.entity_id}"
    return str(exc)
