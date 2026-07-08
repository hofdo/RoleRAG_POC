"""Fake-provider ASGI app for the deterministic frontend<->backend contract test (#60).

Serves the *real* ``app.main:app`` — real routes, real error handlers, real SPA
mount — but replaces the service dependencies with a fully in-process fake stack:
the ``SmokeTestProvider`` canned actor/critic/memory responses over a throwaway
SQLite file, with no model, no Qdrant, and no retriever. A Playwright spec drives
the genuine Angular SPA against this over real HTTP, so the frontend<->backend
request/response contract is exercised deterministically in CI without a model.

Retrieval is intentionally disabled (``actor_context_retriever=None``): the turn
pipeline is fail-open on retrieval, and the fake provider returns canned actor
text regardless, so the contract — create session, run a turn, render it — holds
without a vector store or ingested lore.

Run: ``uvicorn app.diagnostics.contract_app:app``
"""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from pathlib import Path

from app.agents import CriticAgent, MemoryCurator
from app.api.routes import get_read_services, get_turn_services
from app.composition import AppServices, build_file_loader, build_orchestrator_config
from app.config import Settings
from app.diagnostics.smoke_runner import SmokeTestProvider
from app.main import app
from app.memory import MemoryEpisodeStore, RecentDialogueStore
from app.orchestration.turn_orchestrator import TurnOrchestrator
from app.persistence import (
    SQLiteCanonRepository,
    SQLiteMemoryRepository,
    SQLiteSessionRepository,
    SQLiteTurnRepository,
    connect_sqlite,
    initialize_database,
)

# Throwaway per-process DB so the contract run never touches data/rolerag.db.
_DB_PATH = Path(tempfile.mkdtemp(prefix="rolerag-contract-")) / "contract.db"
_SETTINGS = Settings(database_path=str(_DB_PATH))


def _build_fake_services() -> AppServices:
    connection = connect_sqlite(_SETTINGS.database_path)
    initialize_database(connection)
    session_repository = SQLiteSessionRepository(connection)
    turn_repository = SQLiteTurnRepository(connection)
    memory_repository = SQLiteMemoryRepository(connection)
    canon_repository = SQLiteCanonRepository(connection)
    memory_store = MemoryEpisodeStore(memory_repository=memory_repository)
    recent_dialogue_store = RecentDialogueStore(
        turn_repository=turn_repository,
        recent_turns=_SETTINGS.recent_dialogue_turns,
    )
    orchestrator = TurnOrchestrator(
        loader=build_file_loader(_SETTINGS.content_root),
        loader_factory=build_file_loader,
        provider=SmokeTestProvider(),
        cloud_provider=None,
        critic_agent=CriticAgent(),
        session_repository=session_repository,
        turn_repository=turn_repository,
        recent_dialogue_store=recent_dialogue_store,
        memory_store=memory_store,
        memory_curator=MemoryCurator(),
        memory_indexer=None,
        actor_context_retriever=None,
        canon_repository=canon_repository,
        config=build_orchestrator_config(_SETTINGS, content_root=str(_SETTINGS.content_root)),
    )
    return AppServices(
        connection=connection,
        session_repository=session_repository,
        orchestrator=orchestrator,
        recent_dialogue_store=recent_dialogue_store,
        turn_repository=turn_repository,
        memory_repository=memory_repository,
        canon_repository=canon_repository,
        memory_indexer=None,
    )


def _fake_services() -> Generator[AppServices, None, None]:
    # Fresh connection per request (check_same_thread=False + WAL), mirroring the
    # real dependency's lifecycle; state persists across requests via the DB file.
    services = _build_fake_services()
    try:
        yield services
    finally:
        services.close()


# Both the read and turn service dependencies get the same fake builder -- the
# fake stack has no retriever, so the read/turn (enable_retrieval) split is moot.
app.dependency_overrides[get_read_services] = _fake_services
app.dependency_overrides[get_turn_services] = _fake_services

__all__ = ["app"]
