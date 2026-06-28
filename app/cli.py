from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, Protocol, cast

import typer

from app import __version__
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
from app.content import (
    ContentValidationReport,
    ScenarioTemplateResult,
    create_scenario_template,
    validate_content,
)
from app.content.validator import ContentValidationStatus
from app.diagnostics import (
    RuntimeDiagnosticsReport,
    SmokeRunSummary,
    build_runtime_diagnostics,
    run_smoke,
)
from app.domain import TurnInput, TurnOutcome, TurnResult, Visibility
from app.llm.provider import ProviderTimeoutError, ProviderUnavailableError
from app.llm.router import ModelRoute, ModelTask, choose_route
from app.memory import MemoryEpisodeStore, MemoryIndexer, RecentDialogueStore
from app.orchestration.turn_orchestrator import TurnOrchestrator, TurnOrchestratorConfig
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
    RetrievalResult,
    build_retrieval_query,
    ingest_document,
    ingest_lore_manifest,
)

app = typer.Typer(help="RoleRAG CLI")


class _SupportsDropCollection(Protocol):
    def drop_collection(self, collection: RagCollection) -> None: ...


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


def _auto_ingest_scenario_lore(settings: Settings, content_root: Path) -> None:
    """Best-effort: index the scenario's lore so retrieval works without a separate ingest step.

    Never blocks session creation. A scenario with no manifest is silent; an unreachable vector
    store or missing embedding backend only warns (run ingest-scenario-lore later). Idempotent,
    so re-running start-session re-indexes the same lore without duplicating it.
    """
    if not (content_root / "documents" / "manifest.json").exists():
        return  # no lore manifest -> nothing to index; skip building embedding/vector providers
    try:
        results = ingest_lore_manifest(
            content_root,
            embedding_provider=_build_embedding_provider(settings),
            vector_store=_build_vector_store(settings),
            chunking_config=ChunkingConfig(
                chunk_size_chars=settings.rag_chunk_size_chars,
                chunk_overlap_chars=settings.rag_chunk_overlap_chars,
            ),
        )
    except Exception as exc:  # degrade gracefully: a session must still be creatable offline
        # Includes a manifest that references a missing document -- warn rather than swallow,
        # so a typo'd lore path does not silently leave retrieval empty. (The no-manifest case
        # already returned above.)
        typer.secho(
            f"warning: scenario lore auto-ingest skipped: {exc}",
            fg=typer.colors.YELLOW,
            err=True,
        )
        return
    total = sum(result.chunk_count for result in results)
    typer.secho(
        f"auto-ingested {total} lore chunk(s) from {len(results)} document(s)",
        fg=typer.colors.GREEN,
        err=True,
    )


def _build_services(
    settings: Settings,
    *,
    enable_retrieval: bool,
    content_root: Path | str | None = None,
) -> AppServices:
    resolved_content_root = content_root or settings.content_root
    connection = connect_sqlite(settings.database_path)
    initialize_database(connection)
    session_repository = SQLiteSessionRepository(connection)
    turn_repository = SQLiteTurnRepository(connection)
    memory_repository = SQLiteMemoryRepository(connection)
    memory_store = MemoryEpisodeStore(memory_repository=memory_repository)
    recent_dialogue_store = RecentDialogueStore(
        turn_repository=turn_repository,
        recent_turns=settings.recent_dialogue_turns,
    )
    embedding_provider = _build_embedding_provider(settings) if enable_retrieval else None
    vector_store = _build_vector_store(settings) if enable_retrieval else None
    orchestrator = TurnOrchestrator(
        loader=_build_file_loader(resolved_content_root),
        loader_factory=_build_file_loader,
        provider=_build_local_provider(settings),
        cloud_provider=_build_cloud_provider(settings),
        critic_agent=_build_critic_agent(settings),
        session_repository=session_repository,
        turn_repository=turn_repository,
        recent_dialogue_store=recent_dialogue_store,
        memory_store=memory_store,
        memory_curator=_build_memory_curator(settings),
        memory_indexer=(
            MemoryIndexer(
                memory_store=memory_store,
                embedding_provider=embedding_provider,
                vector_store=vector_store,
            )
            if embedding_provider is not None and vector_store is not None
            else None
        ),
        actor_context_retriever=(
            _build_actor_context_retriever(
                settings,
                embedding_provider=embedding_provider,
                vector_store=vector_store,
            )
            if embedding_provider is not None and vector_store is not None
            else None
        ),
        config=TurnOrchestratorConfig(
            content_root=str(resolved_content_root),
            retrieval_top_k=settings.rag_default_top_k,
            max_retrieved_chunk_chars=settings.rag_max_retrieved_chunk_chars,
            local_model=settings.local_llm_model,
            cloud_model=settings.cloud_llm_model,
            local_max_tokens=settings.local_llm_max_tokens,
            cloud_max_tokens=settings.cloud_llm_max_tokens,
            local_temperature=settings.local_llm_temperature,
            cloud_temperature=settings.cloud_llm_temperature,
            cloud_mode=settings.cloud_mode,
        ),
    )
    return AppServices(
        connection=connection,
        session_repository=session_repository,
        orchestrator=orchestrator,
        recent_dialogue_store=recent_dialogue_store,
    )


@app.command()
def config() -> None:
    settings = get_settings()
    typer.echo(json.dumps(_redact_settings(settings), indent=2, sort_keys=True))


@app.command()
def health() -> None:
    settings = get_settings()
    typer.echo(
        json.dumps(
            {
                "name": "rolerag-poc",
                "status": "ok",
                "version": __version__,
                "settings": _redact_settings(settings),
            },
            indent=2,
            sort_keys=True,
        )
    )


def run_doctor(
    *,
    settings: Settings,
    check_qdrant: bool = False,
    check_local_provider: bool = False,
) -> RuntimeDiagnosticsReport:
    return build_runtime_diagnostics(
        settings,
        check_qdrant=check_qdrant,
        check_local_provider=check_local_provider,
    )


def _echo_json(payload: object) -> None:
    if hasattr(payload, "model_dump"):
        content = payload.model_dump(mode="json")
    else:
        content = payload
    typer.echo(json.dumps(content, indent=2, sort_keys=True))


def _status_exit_code(status: object) -> int:
    value = getattr(status, "value", status)
    return 1 if value == "fail" else 0


@app.command()
def doctor(
    check_qdrant: Annotated[
        bool,
        typer.Option(help="Verify Qdrant reachability"),
    ] = False,
    check_local_provider: Annotated[
        bool,
        typer.Option(help="Verify local provider reachability"),
    ] = False,
) -> None:
    settings = get_settings()
    report = run_doctor(
        settings=settings,
        check_qdrant=check_qdrant,
        check_local_provider=check_local_provider,
    )
    _echo_json(report)
    raise typer.Exit(code=_status_exit_code(report.status))


@app.command("smoke-run")
def smoke_run(
    real_runtime: Annotated[
        bool,
        typer.Option(help="Also run optional live Qdrant and local-provider checks"),
    ] = False,
) -> None:
    settings = get_settings()
    summary: SmokeRunSummary = run_smoke(settings=settings, real_runtime=real_runtime)
    _echo_json(summary)
    raise typer.Exit(code=_status_exit_code(summary.status))


def run_content_validation(
    *,
    content_root: Path,
    world_id: str | None = None,
) -> ContentValidationReport:
    return validate_content(content_root=content_root, world_id=world_id)


@app.command("validate-content")
def validate_content_command(
    content_root: Annotated[
        Path | None,
        typer.Option(
            help="Content root containing worlds, scenes, personas, and optional documents"
        ),
    ] = None,
    world_id: Annotated[
        str | None,
        typer.Option(help="Optional world id to focus world-graph checks on"),
    ] = None,
) -> None:
    settings = get_settings()
    report = run_content_validation(
        content_root=content_root or settings.content_root,
        world_id=world_id,
    )
    _echo_json(report)
    raise typer.Exit(code=_status_exit_code(report.status))


@app.command("create-scenario-template")
def create_scenario_template_command(
    output: Annotated[
        Path,
        typer.Option(help="Output directory for the standalone scenario pack"),
    ],
    name: Annotated[str | None, typer.Option(help="Optional scenario display name")] = None,
    overwrite: Annotated[
        bool,
        typer.Option(help="Overwrite generated files if the output exists"),
    ] = False,
) -> None:
    try:
        result: ScenarioTemplateResult = create_scenario_template(
            output=output,
            name=name,
            overwrite=overwrite,
        )
    except (FileExistsError, ValueError, OSError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    _echo_json(result)


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
    content_root: Annotated[
        Path | None,
        typer.Option(help="Content root containing scenario files"),
    ] = None,
    skip_lore_ingest: Annotated[
        bool,
        typer.Option("--skip-lore-ingest", help="Do not auto-index scenario lore on start"),
    ] = False,
) -> None:
    settings = get_settings()
    resolved_content_root = content_root or settings.content_root
    services = _build_services(
        settings,
        enable_retrieval=False,
        content_root=resolved_content_root,
    )
    try:
        session = services.orchestrator.create_session(
            world_id=world_id,
            scene_id=scene_id,
            active_persona_id=active_persona_id,
            player_name=player_name,
            session_id=session_id,
            content_root=str(resolved_content_root),
        )
    except (DataFileNotFoundError, DataValidationError, ValueError) as exc:
        services.close()
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    services.close()
    # Auto-index lore so retrieval works immediately; ingest-scenario-lore is no longer a
    # mandatory separate step (it stays available for re-indexing without starting a session).
    if not skip_lore_ingest:
        _auto_ingest_scenario_lore(settings, Path(resolved_content_root))
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
        typer.secho(str(exc), fg=typer.colors.RED)
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
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))


@app.command("ingest-scenario-lore")
def ingest_scenario_lore(
    content_root: Annotated[
        Path,
        typer.Option(help="Scenario pack content root containing documents/manifest.json"),
    ],
) -> None:
    settings = get_settings()
    try:
        validation_report = validate_content(content_root=content_root)
        if validation_report.status == ContentValidationStatus.FAIL:
            raise ValueError("content validation failed")
        results = ingest_lore_manifest(
            content_root,
            embedding_provider=_build_embedding_provider(settings),
            vector_store=_build_vector_store(settings),
            chunking_config=ChunkingConfig(
                chunk_size_chars=settings.rag_chunk_size_chars,
                chunk_overlap_chars=settings.rag_chunk_overlap_chars,
            ),
        )
    except (FileNotFoundError, ImportError, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "content_root": str(content_root),
                "documents": [
                    result.model_dump(mode="json")
                    for result in results
                ],
                "total_chunk_count": sum(result.chunk_count for result in results),
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.command()
def reindex_memories(
    session_id: Annotated[str, typer.Option(help="Session identifier")],
) -> None:
    settings = get_settings()
    connection = connect_sqlite(settings.database_path)
    initialize_database(connection)
    session_repository = SQLiteSessionRepository(connection)
    memory_store = MemoryEpisodeStore(
        memory_repository=SQLiteMemoryRepository(connection),
    )
    try:
        if session_repository.get_session(session_id) is None:
            raise SessionNotFoundError(session_id)
        result = MemoryIndexer(
            memory_store=memory_store,
            embedding_provider=_build_embedding_provider(settings),
            vector_store=_build_vector_store(settings),
        ).reindex_session(session_id)
    except Exception as exc:
        connection.close()
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    connection.close()
    typer.echo(
        json.dumps(
            {
                "indexed_count": result.indexed_count,
                "session_id": session_id,
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.command("retrieve-debug")
def retrieve_debug(
    session_id: Annotated[str, typer.Option(help="Session identifier")],
    query: Annotated[str, typer.Option(help="User query to inspect retrieval for")],
) -> None:
    settings = get_settings()
    services = _build_services(settings, enable_retrieval=False)
    try:
        session = services.orchestrator.resume_session(session_id)
        loader = services.orchestrator.loader_for_session(session)
        persona = loader.load_persona(session.active_persona_id)
        scene = loader.load_scene(session.active_scene_id)
        retrieval_query = build_retrieval_query(
            user_message=query,
            scene=scene,
            persona=persona,
            recent_turns=services.recent_dialogue_store.load_recent_dialogue(session.id),
        )
        retrieval_result: RetrievalResult = _build_actor_context_retriever(
            settings,
            embedding_provider=_build_embedding_provider(settings),
            vector_store=_build_vector_store(settings),
        ).retrieve_for_actor_with_diagnostics(
            query=retrieval_query,
            lexical_query=query,
            world_id=session.world_id,
            session_id=session.id,
            persona_id=persona.id,
            scene_id=scene.id,
            top_k=settings.rag_default_top_k,
        )
    except (
        DataFileNotFoundError,
        DataValidationError,
        ImportError,
        SessionNotFoundError,
        ValueError,
    ) as exc:
        services.close()
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    services.close()
    typer.echo(
        json.dumps(
            retrieval_result.diagnostics.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
        )
    )


@app.command("embedding-ab")
def embedding_ab(
    model: Annotated[
        list[str],
        typer.Option(
            help="FastEmbed model name to benchmark (repeatable). "
            "Models download on first use.",
        ),
    ],
    baseline: Annotated[
        str,
        typer.Option(
            help="Baseline FastEmbed model name (downloads on first use)",
        ),
    ] = "sentence-transformers/all-MiniLM-L6-v2",
) -> None:
    """Rank the seeded durable memories with each embedding model (LLM-free).

    For each model (the --baseline plus every --model) a real
    FastEmbedEmbeddingProvider is built and used to seed the study corpus, then
    per-event ranks, mean rank, and miss count are printed. Models are auto-
    downloaded by fastembed on first use, so the first run for a given model is slow.
    """
    from app.diagnostics.embedding_ab import measure_embedding_ranks
    from app.evals.event_key_retrieval import SEEDED_EVENTS
    from app.rag.embeddings import FastEmbedEmbeddingProvider

    model_names = [baseline, *model]
    event_keys = [event.key for event in SEEDED_EVENTS]
    rows: list[tuple[str, dict[str, int | None]]] = []
    for model_name in model_names:
        try:
            ranks = measure_embedding_ranks(
                embedding_provider=FastEmbedEmbeddingProvider(model_name=model_name),
            )
        except ImportError as exc:
            typer.echo(str(exc))
            raise typer.Exit(code=1) from exc
        rows.append((model_name, ranks))

    typer.echo(_render_embedding_ab_table(event_keys, rows))


def _render_embedding_ab_table(
    event_keys: Sequence[str],
    rows: Sequence[tuple[str, dict[str, int | None]]],
) -> str:
    def cell(value: int | None) -> str:
        return "miss" if value is None else str(value)

    headers = ["model", *event_keys, "mean rank", "miss count"]
    body: list[list[str]] = []
    for model_name, ranks in rows:
        present = [rank for rank in ranks.values() if rank is not None]
        mean_rank = f"{sum(present) / len(present):.2f}" if present else "n/a"
        miss_count = sum(1 for rank in ranks.values() if rank is None)
        body.append(
            [
                model_name,
                *(cell(ranks.get(key)) for key in event_keys),
                mean_rank,
                str(miss_count),
            ]
        )
    widths = [
        max(len(headers[col]), *(len(line[col]) for line in body)) if body else len(headers[col])
        for col in range(len(headers))
    ]

    def format_row(cells: Sequence[str]) -> str:
        return "| " + " | ".join(c.ljust(widths[col]) for col, c in enumerate(cells)) + " |"

    lines = [
        format_row(headers),
        "| " + " | ".join("-" * widths[col] for col in range(len(headers))) + " |",
        *(format_row(line) for line in body),
    ]
    return "\n".join(lines)


@app.command()
def turn(
    message: Annotated[str, typer.Option(help="Player message for the demo turn")],
    session_id: Annotated[str, typer.Option(help="Session identifier")],
    request_cloud: Annotated[
        bool,
        typer.Option(help="Request cloud quality for this turn"),
    ] = False,
    confirm_cloud: Annotated[
        bool,
        typer.Option(help="Pre-approve a confirmation-required cloud route"),
    ] = False,
    force_local: Annotated[
        bool,
        typer.Option(help="Decline cloud routing and answer locally"),
    ] = False,
) -> None:
    settings = get_settings()
    services = _build_services(settings, enable_retrieval=True)
    turn_input = TurnInput(
        session_id=session_id,
        message=message,
        user_requested_cloud=request_cloud,
        cloud_confirmed=confirm_cloud,
        force_local=force_local,
    )
    try:
        result = asyncio.run(
            _run_turn(
                services=services,
                turn_input=turn_input,
            )
        )
        if result.outcome == TurnOutcome.CONFIRMATION_REQUIRED:
            approved = typer.confirm(
                f"Route this turn to cloud model {result.route.model}? "
                f"Reason: {result.route.reason}",
                default=False,
            )
            result = asyncio.run(
                _run_turn(
                    services=services,
                    turn_input=turn_input.model_copy(
                        update=(
                            {"cloud_confirmed": True}
                            if approved
                            else {"force_local": True}
                        )
                    ),
                )
            )
    except (ProviderTimeoutError, ProviderUnavailableError) as exc:
        services.close()
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    except (DataFileNotFoundError, DataValidationError, SessionNotFoundError, ValueError) as exc:
        services.close()
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    services.close()
    for warning in result.warnings:
        typer.secho(f"Warning: {warning}", fg=typer.colors.YELLOW, err=True)
    typer.echo(result.text)


def _open_repositories() -> tuple[
    Any, SQLiteSessionRepository, SQLiteTurnRepository, SQLiteMemoryRepository
]:
    settings = get_settings()
    connection = connect_sqlite(settings.database_path)
    initialize_database(connection)
    return (
        connection,
        SQLiteSessionRepository(connection),
        SQLiteTurnRepository(connection),
        SQLiteMemoryRepository(connection),
    )


def _delete_session_vectors(session_id: str) -> None:
    settings = get_settings()
    try:
        vector_store = _build_vector_store(settings)
        vector_store.delete_session_points(RagCollection.SESSION_MEMORY, session_id)
    except Exception as exc:
        typer.secho(
            f"Warning: session vector cleanup skipped: {exc}", fg=typer.colors.YELLOW, err=True
        )


@app.command("list-sessions")
def list_sessions(
    limit: Annotated[int, typer.Option(help="Maximum sessions to list")] = 50,
) -> None:
    connection, sessions, turns, _ = _open_repositories()
    payload = {
        "sessions": [
            {
                "session_id": session.id,
                "world_id": session.world_id,
                "active_scene_id": session.active_scene_id,
                "active_persona_id": session.active_persona_id,
                "player_name": session.player_name,
                "turn_count": turns.count_turns(session.id),
                "updated_at": (
                    session.updated_at.isoformat() if session.updated_at else None
                ),
            }
            for session in sessions.list_recent_sessions(limit)
        ]
    }
    connection.close()
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("delete-session")
def delete_session(
    session_id: Annotated[str, typer.Option(help="Session identifier to delete")],
    yes: Annotated[bool, typer.Option("--yes", help="Skip confirmation prompt")] = False,
) -> None:
    if not yes:
        typer.confirm(
            f"Delete session {session_id} with all turns and memories?",
            abort=True,
        )
    connection, sessions, _, _ = _open_repositories()
    deleted = sessions.delete_session(session_id)
    connection.close()
    if not deleted:
        typer.echo(f"Unknown session id: {session_id}")
        raise typer.Exit(code=1)
    _delete_session_vectors(session_id)
    typer.echo(json.dumps({"deleted": session_id}))


@app.command("export-session")
def export_session(
    session_id: Annotated[str, typer.Option(help="Session identifier to export")],
    output: Annotated[Path, typer.Option(help="Output JSON file path")],
) -> None:
    connection, sessions, turns, memories = _open_repositories()
    session = sessions.get_session(session_id)
    if session is None:
        connection.close()
        typer.echo(f"Unknown session id: {session_id}")
        raise typer.Exit(code=1)
    envelope = {
        "format_version": 1,
        "session": session.model_dump(mode="json"),
        "turns": [turn.model_dump(mode="json") for turn in turns.list_all_turns(session_id)],
        "memory_episodes": [
            memory.model_dump(mode="json")
            for memory in memories.list_memories_for_session(session_id)
        ],
    }
    connection.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(envelope, indent=2, sort_keys=True), encoding="utf-8")
    typer.echo(json.dumps({"exported": session_id, "output": str(output)}))


@app.command("import-session")
def import_session(
    input_path: Annotated[Path, typer.Option("--input", help="Exported session JSON file")],
    new_id: Annotated[
        bool,
        typer.Option("--new-id", help="Import under a fresh session id"),
    ] = False,
) -> None:
    from uuid import uuid4

    from app.domain import MemoryCandidate, SessionState

    envelope = json.loads(input_path.read_text(encoding="utf-8"))
    if envelope.get("format_version") != 1:
        typer.echo(f"Unsupported export format: {envelope.get('format_version')}")
        raise typer.Exit(code=1)

    session = SessionState.model_validate(envelope["session"])
    if new_id:
        session = session.model_copy(update={"id": str(uuid4())})

    connection, sessions, turns, memories = _open_repositories()
    if sessions.get_session(session.id) is not None:
        connection.close()
        typer.echo(f"Session {session.id} already exists; use --new-id to import a copy.")
        raise typer.Exit(code=1)
    sessions.create_session(session)
    for raw_turn in envelope["turns"]:
        route = ModelRoute.model_validate(raw_turn["route"])
        turns.append_turn(
            session_id=session.id,
            scene_id=raw_turn["scene_id"],
            persona_id=raw_turn["persona_id"],
            user_message=raw_turn["user_message"],
            assistant_message=raw_turn["assistant_message"],
            route=route,
        )
    candidates = [
        MemoryCandidate(
            summary=raw["summary"],
            visibility=Visibility(raw["visibility"]),
            importance=raw["importance"],
            tags=list(raw.get("tags", [])),
            scene_id=raw.get("scene_id") or None,
            actor_id=raw.get("actor_id"),
        )
        for raw in envelope["memory_episodes"]
    ]
    if candidates:
        memories.append_memories(session_id=session.id, memories=candidates)
    connection.close()
    typer.echo(json.dumps({"session_id": session.id, "turns": len(envelope["turns"])}))
    typer.echo(
        "Run reindex-memories --session-id "
        f"{session.id} to rebuild the vector index.",
        err=True,
    )


@app.command("inspect-memories")
def inspect_memories(
    session_id: Annotated[str, typer.Option(help="Session identifier")],
    limit: Annotated[
        int | None,
        typer.Option(help="Maximum number of memories to show"),
    ] = None,
) -> None:
    connection, sessions, _, memories = _open_repositories()
    if sessions.get_session(session_id) is None:
        connection.close()
        typer.echo(f"Unknown session id: {session_id}")
        raise typer.Exit(code=1)
    episodes = memories.list_memories_for_session(session_id, limit=limit)
    connection.close()
    typer.echo(
        json.dumps(
            {
                "session_id": session_id,
                "memories": [
                    {
                        "id": episode.id,
                        "scene_id": episode.scene_id,
                        "actor_id": episode.actor_id,
                        "summary": episode.summary,
                        "importance": episode.importance,
                        "visibility": episode.visibility.value,
                        "tags": episode.tags,
                    }
                    for episode in episodes
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.command("turn-history")
def turn_history(
    session_id: Annotated[str, typer.Option(help="Session identifier")],
    turn: Annotated[
        int | None,
        typer.Option(help="Filter to a single turn_index"),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(help="Maximum number of turns to show"),
    ] = None,
) -> None:
    connection, sessions, turns, _ = _open_repositories()
    if sessions.get_session(session_id) is None:
        connection.close()
        typer.echo(f"Unknown session id: {session_id}")
        raise typer.Exit(code=1)
    stored_turns = turns.list_all_turns(session_id)
    if turn is not None:
        stored_turns = [item for item in stored_turns if item.turn_index == turn]
    if limit is not None:
        stored_turns = stored_turns[:limit]
    connection.close()
    typer.echo(
        json.dumps(
            {
                "session_id": session_id,
                "turns": [
                    {
                        "turn_index": item.turn_index,
                        "scene_id": item.scene_id,
                        "persona_id": item.persona_id,
                        "user_message": item.user_message,
                        "assistant_message": item.assistant_message,
                        "route": {
                            "provider": item.route.provider.value,
                            "model": item.route.model,
                            "reason": item.route.reason,
                        },
                        "created_at": item.created_at.isoformat(),
                        "diagnostics": (
                            item.diagnostics.model_dump(mode="json")
                            if item.diagnostics is not None
                            else None
                        ),
                    }
                    for item in stored_turns
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.command("reset-db")
def reset_db(
    yes: Annotated[bool, typer.Option("--yes", help="Skip confirmation prompt")] = False,
) -> None:
    if not yes:
        typer.confirm(
            "Delete ALL sessions, turns, and memories? This cannot be undone.",
            abort=True,
        )
    connection, sessions, _, _ = _open_repositories()
    session_ids = [session.id for session in sessions.list_recent_sessions(1_000_000)]
    connection.execute("DELETE FROM sessions")
    connection.commit()
    connection.close()
    for session_id in session_ids:
        _delete_session_vectors(session_id)
    typer.echo(json.dumps({"deleted_sessions": len(session_ids)}))


@app.command("reset-index")
def reset_index(
    collection: Annotated[
        str,
        typer.Option(
            help="Vector collection to drop: all|canon_lore|session_memory|persona_memory"
        ),
    ] = "all",
    yes: Annotated[bool, typer.Option("--yes", help="Skip confirmation prompt")] = False,
) -> None:
    if collection == "all":
        targets = list(RagCollection)
    else:
        try:
            targets = [RagCollection(collection)]
        except ValueError:
            typer.echo(
                f"Unknown collection: {collection} "
                "(choose all, canon_lore, session_memory, or persona_memory)"
            )
            raise typer.Exit(code=1) from None
    if not yes:
        typer.confirm(
            "Drop the selected vector index collection(s)? This cannot be undone.",
            abort=True,
        )
    settings = get_settings()
    vector_store = cast(_SupportsDropCollection, _build_vector_store(settings))
    for target in targets:
        vector_store.drop_collection(target)
    typer.echo(json.dumps({"dropped": [target.value for target in targets]}))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
