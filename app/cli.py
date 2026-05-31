from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer

from app.composition import (
    AppServices,
    build_actor_context_retriever,
    build_cloud_provider,
    build_critic_agent,
    build_embedding_provider,
    build_file_loader,
    build_local_provider,
    build_memory_curator,
    build_vector_store,
    redact_settings,
)
from app.config import Settings, get_settings
from app.domain import TurnInput, TurnResult, Visibility
from app.llm.router import ModelTask, choose_route
from app.memory import MemoryEpisodeStore, RecentDialogueStore
from app.orchestration.turn_orchestrator import TurnOrchestrator
from app.persistence import (
    DataFileNotFoundError,
    DataValidationError,
    SessionNotFoundError,
    SQLiteMemoryRepository,
    SQLiteSessionRepository,
    SQLiteTurnRepository,
    connect_sqlite,
    initialize_database,
)
from app.rag import (
    ChunkingConfig,
    IngestionRequest,
    RagCollection,
    ingest_document,
)

app = typer.Typer(help="RoleRAG CLI")


async def _run_turn(
    *,
    services: AppServices,
    turn_input: TurnInput,
) -> TurnResult:
    return await services.orchestrator.run_turn(turn_input=turn_input)


_redact_settings = redact_settings
_build_local_provider = build_local_provider
_build_cloud_provider = build_cloud_provider
_build_critic_agent = build_critic_agent
_build_file_loader = build_file_loader
_build_memory_curator = build_memory_curator
_build_embedding_provider = build_embedding_provider
_build_vector_store = build_vector_store
_build_actor_context_retriever = build_actor_context_retriever


def _build_services(settings: Settings, *, enable_retrieval: bool) -> AppServices:
    connection = connect_sqlite(settings.database_path)
    initialize_database(connection)
    session_repository = SQLiteSessionRepository(connection)
    turn_repository = SQLiteTurnRepository(connection)
    memory_repository = SQLiteMemoryRepository(connection)
    recent_dialogue_store = RecentDialogueStore(
        turn_repository=turn_repository,
        recent_turns=settings.recent_dialogue_turns,
    )
    orchestrator = TurnOrchestrator(
        loader=_build_file_loader(),
        provider=_build_local_provider(settings),
        cloud_provider=_build_cloud_provider(settings),
        critic_agent=_build_critic_agent(),
        session_repository=session_repository,
        turn_repository=turn_repository,
        recent_dialogue_store=recent_dialogue_store,
        memory_store=MemoryEpisodeStore(memory_repository=memory_repository),
        memory_curator=_build_memory_curator(),
        actor_context_retriever=(
            _build_actor_context_retriever(settings) if enable_retrieval else None
        ),
        retrieval_top_k=settings.rag_default_top_k,
        max_retrieved_chunk_chars=settings.rag_max_retrieved_chunk_chars,
        local_model=settings.local_llm_model,
        cloud_model=settings.cloud_llm_model,
        local_max_tokens=settings.local_llm_max_tokens,
        cloud_max_tokens=settings.cloud_llm_max_tokens,
        local_temperature=settings.local_llm_temperature,
        cloud_temperature=settings.cloud_llm_temperature,
        cloud_mode=settings.cloud_mode,
    )
    return AppServices(
        connection=connection,
        orchestrator=orchestrator,
        recent_dialogue_store=recent_dialogue_store,
    )


@app.command()
def config() -> None:
    settings = get_settings()
    typer.echo(json.dumps(_redact_settings(settings), indent=2, sort_keys=True))


@app.command()
def start_session(
    world_id: Annotated[str, typer.Option(help="World identifier")] = "demo_world",
    scene_id: Annotated[str, typer.Option(help="Scene identifier")] = "rose-gallery",
    active_persona_id: Annotated[
        str,
        typer.Option(help="Active demo persona identifier"),
    ] = "archivist",
    player_name: Annotated[str, typer.Option(help="Player name")] = "Player",
    session_id: Annotated[
        str | None,
        typer.Option(help="Optional explicit session identifier"),
    ] = None,
) -> None:
    settings = get_settings()
    services = _build_services(settings, enable_retrieval=False)
    try:
        session = services.orchestrator.create_session(
            world_id=world_id,
            scene_id=scene_id,
            active_persona_id=active_persona_id,
            player_name=player_name,
            session_id=session_id,
        )
    except (DataFileNotFoundError, DataValidationError, ValueError) as exc:
        services.close()
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    services.close()
    typer.echo(json.dumps(session.model_dump(mode="json"), indent=2, sort_keys=True))


@app.command()
def resume(
    session_id: Annotated[str, typer.Option(help="Session identifier")],
) -> None:
    settings = get_settings()
    services = _build_services(settings, enable_retrieval=False)
    try:
        session = services.orchestrator.resume_session(session_id)
    except SessionNotFoundError as exc:
        services.close()
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    services.close()
    typer.echo(json.dumps(session.model_dump(mode="json"), indent=2, sort_keys=True))


@app.command()
def route(
    task: Annotated[ModelTask, typer.Option(help="Model task to route")],
    failed_local_attempts: Annotated[int, typer.Option(min=0)] = 0,
    scene_complexity: Annotated[int, typer.Option(min=1)] = 1,
    retrieval_confidence: Annotated[float | None, typer.Option(min=0.0, max=1.0)] = None,
    request_cloud: Annotated[bool, typer.Option(help="Simulate an explicit cloud request")] = False,
) -> None:
    settings = get_settings()
    chosen_route = choose_route(
        task=task,
        cloud_mode=settings.cloud_mode,
        local_model=settings.local_llm_model,
        cloud_model=settings.cloud_llm_model,
        local_max_tokens=settings.local_llm_max_tokens,
        cloud_max_tokens=settings.cloud_llm_max_tokens,
        local_temperature=settings.local_llm_temperature,
        cloud_temperature=settings.cloud_llm_temperature,
        failed_local_attempts=failed_local_attempts,
        retrieval_confidence=retrieval_confidence,
        scene_complexity=scene_complexity,
        user_requested_cloud=request_cloud,
    )
    typer.echo(json.dumps(chosen_route.model_dump(), indent=2, sort_keys=True))


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(help="Markdown or text document to ingest")],
    visibility: Annotated[Visibility, typer.Option(help="Required chunk visibility")] ,
    source_type: Annotated[str, typer.Option(help="Document source type label")] ,
    collection: Annotated[
        RagCollection,
        typer.Option(help="Target vector collection"),
    ] = RagCollection.CANON_LORE,
    tag: Annotated[list[str] | None, typer.Option(help="Repeatable chunk tag")] = None,
    world_id: Annotated[str | None, typer.Option(help="Optional world scope")] = None,
    scene_id: Annotated[str | None, typer.Option(help="Optional scene scope")] = None,
    persona_id: Annotated[str | None, typer.Option(help="Optional persona scope")] = None,
    session_id: Annotated[str | None, typer.Option(help="Optional session scope")] = None,
) -> None:
    settings = get_settings()
    try:
        result = ingest_document(
            IngestionRequest(
                path=path,
                collection=collection,
                source_type=source_type,
                visibility=visibility,
                tags=tag or [],
                world_id=world_id,
                scene_id=scene_id,
                persona_id=persona_id,
                session_id=session_id,
            ),
            embedding_provider=_build_embedding_provider(settings),
            vector_store=_build_vector_store(settings),
            chunking_config=ChunkingConfig(
                chunk_size_chars=settings.rag_chunk_size_chars,
                chunk_overlap_chars=settings.rag_chunk_overlap_chars,
            ),
        )
    except (FileNotFoundError, ImportError, ValueError) as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))


@app.command()
def turn(
    message: Annotated[str, typer.Option(help="Player message for the demo turn")],
    session_id: Annotated[str, typer.Option(help="Session identifier")],
    request_cloud: Annotated[
        bool,
        typer.Option(help="Request cloud quality for this turn"),
    ] = False,
) -> None:
    settings = get_settings()
    services = _build_services(settings, enable_retrieval=True)
    turn_input = TurnInput(
        session_id=session_id,
        message=message,
        user_requested_cloud=request_cloud,
    )
    try:
        result = asyncio.run(
            _run_turn(
                services=services,
                turn_input=turn_input,
            )
        )
    except (DataFileNotFoundError, DataValidationError, SessionNotFoundError, ValueError) as exc:
        services.close()
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    services.close()
    for warning in result.warnings:
        typer.echo(f"Warning: {warning}", err=True)
    typer.echo(result.text)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
