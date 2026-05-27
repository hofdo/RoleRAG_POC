from __future__ import annotations

import asyncio
import json
from typing import Annotated

import typer

from app.agents import MemoryCurator
from app.config import Settings, get_settings
from app.domain import TurnInput, TurnResult
from app.llm.openai_compatible import OpenAICompatibleProvider
from app.llm.provider import LlmProvider
from app.llm.router import ModelTask, choose_route
from app.memory import MemoryEpisodeStore, RecentDialogueStore
from app.orchestration.turn_orchestrator import TurnOrchestrator
from app.persistence import (
    DataFileNotFoundError,
    DataValidationError,
    FileDataLoader,
    SessionNotFoundError,
    SQLiteMemoryRepository,
    SQLiteSessionRepository,
    SQLiteTurnRepository,
    connect_sqlite,
    initialize_database,
)

app = typer.Typer(help="RoleRAG CLI")


def _redact_settings(settings: Settings) -> dict[str, object]:
    values = settings.model_dump()
    values["local_llm_api_key"] = "***"
    values["cloud_llm_api_key"] = "***"
    return values


def _build_local_provider(settings: Settings) -> LlmProvider:
    return OpenAICompatibleProvider(
        provider_name="local",
        base_url=settings.local_llm_base_url,
        api_key=settings.local_llm_api_key,
    )


def _build_file_loader() -> FileDataLoader:
    return FileDataLoader()


def _build_orchestrator(settings: Settings, provider: LlmProvider) -> TurnOrchestrator:
    connection = connect_sqlite(settings.database_path)
    initialize_database(connection)
    turn_repository = SQLiteTurnRepository(connection)
    memory_repository = SQLiteMemoryRepository(connection)
    return TurnOrchestrator(
        loader=_build_file_loader(),
        provider=provider,
        session_repository=SQLiteSessionRepository(connection),
        turn_repository=turn_repository,
        recent_dialogue_store=RecentDialogueStore(
            turn_repository=turn_repository,
            recent_turns=settings.recent_dialogue_turns,
        ),
        memory_store=MemoryEpisodeStore(memory_repository=memory_repository),
        memory_curator=MemoryCurator(),
        local_model=settings.local_llm_model,
        cloud_model=settings.cloud_llm_model,
        local_max_tokens=settings.local_llm_max_tokens,
        cloud_max_tokens=settings.cloud_llm_max_tokens,
        local_temperature=settings.local_llm_temperature,
        cloud_temperature=settings.cloud_llm_temperature,
        cloud_mode=settings.cloud_mode,
    )


async def _run_turn(
    *,
    orchestrator: TurnOrchestrator,
    turn_input: TurnInput,
) -> TurnResult:
    return await orchestrator.run_turn(turn_input=turn_input)


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
    provider = _build_local_provider(settings)
    orchestrator = _build_orchestrator(settings, provider)
    try:
        session = orchestrator.create_session(
            world_id=world_id,
            scene_id=scene_id,
            active_persona_id=active_persona_id,
            player_name=player_name,
            session_id=session_id,
        )
    except (DataFileNotFoundError, DataValidationError, ValueError) as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(session.model_dump(mode="json"), indent=2, sort_keys=True))


@app.command()
def resume(
    session_id: Annotated[str, typer.Option(help="Session identifier")],
) -> None:
    settings = get_settings()
    provider = _build_local_provider(settings)
    orchestrator = _build_orchestrator(settings, provider)
    try:
        session = orchestrator.resume_session(session_id)
    except SessionNotFoundError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(session.model_dump(mode="json"), indent=2, sort_keys=True))


@app.command()
def route(
    task: Annotated[ModelTask, typer.Option(help="Model task to route")],
    failed_local_attempts: Annotated[int, typer.Option(min=0)] = 0,
    scene_complexity: Annotated[int, typer.Option(min=1)] = 1,
    retrieval_confidence: Annotated[float | None, typer.Option(min=0.0, max=1.0)] = None,
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
    )
    typer.echo(json.dumps(chosen_route.model_dump(), indent=2, sort_keys=True))


@app.command()
def turn(
    message: Annotated[str, typer.Option(help="Player message for the demo turn")],
    session_id: Annotated[str, typer.Option(help="Session identifier")],
) -> None:
    settings = get_settings()
    provider = _build_local_provider(settings)
    orchestrator = _build_orchestrator(settings, provider)
    turn_input = TurnInput(
        session_id=session_id,
        message=message,
    )
    try:
        result = asyncio.run(
            _run_turn(
                orchestrator=orchestrator,
                turn_input=turn_input,
            )
        )
    except (DataFileNotFoundError, DataValidationError, SessionNotFoundError, ValueError) as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    typer.echo(result.text)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
