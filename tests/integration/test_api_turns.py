from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Iterator
from pathlib import Path
from typing import cast

import pytest
from fastapi.testclient import TestClient

from app.api.routes import get_read_services, get_turn_services, stream_turn
from app.api.schemas import CreateTurnRequest
from app.composition import AppServices
from app.config import Settings, get_settings
from app.domain import (
    CriticResult,
    PersonaCard,
    RetrievedChunk,
    SceneState,
    SessionState,
    Visibility,
)
from app.evals.fixtures import DeterministicKeywordEmbeddingProvider
from app.llm.provider import LlmMessage, LlmProvider, LlmRequest, LlmResponse
from app.main import app
from app.memory import MemoryIndexer, RecentDialogueStore
from app.orchestration.turn_orchestrator import TurnOrchestrator, TurnOrchestratorConfig
from app.persistence import (
    DemoWorldRecord,
    SQLiteMemoryRepository,
    SQLiteSessionRepository,
    SQLiteTurnRepository,
)
from app.persistence.sqlite import connect_sqlite, initialize_database, serialize_datetime
from app.rag import ActorContextRetriever, InMemoryVectorStore, RagChunk, RagCollection, Retriever


@pytest.fixture(autouse=True)
def _isolate_deferred_memory_settings(tmp_path: Path) -> Iterator[None]:
    # Every test in this module exercises POST /turns or /turns/stream, which
    # schedules a deferred memory job (app.api.routes._schedule_deferred_memory).
    # That job builds its OWN AppServices from Settings resolved via the
    # get_settings dependency -- it does NOT reuse the fake services injected via
    # get_turn_services/get_read_services overrides below. Without overriding
    # get_settings too, the deferred job falls back to a real Settings() and
    # touches data/rolerag.db in the repo's CWD during `make check`. Point it at
    # a per-test tmp_path database instead so no test suite run has real DB
    # side effects.
    app.dependency_overrides[get_settings] = lambda: Settings(
        database_path=str(tmp_path / "deferred-memory-settings.db")
    )
    yield
    app.dependency_overrides.pop(get_settings, None)


class SequencedFakeProvider(LlmProvider):
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.requests: list[LlmRequest] = []

    async def generate(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        return LlmResponse(
            text=self.responses[len(self.requests) - 1],
            provider="fake",
            model=request.model,
            usage={"total_tokens": 15},
            finish_reason="stop",
        )


class FakeLoader:
    def load_world(self, world_id: str) -> DemoWorldRecord:
        return DemoWorldRecord(
            id=world_id,
            name="Winter Palace Intrigue",
            default_scene_id="rose-gallery",
            persona_ids=["archivist", "warden"],
            scene_ids=["rose-gallery", "east-wing"],
        )

    def load_persona(self, persona_id: str) -> PersonaCard:
        return PersonaCard(
            id=persona_id,
            name="Iria Vale",
            role="npc",
            public_description="A composed palace archivist.",
            private_description="She is quietly aiding the coup.",
            speaking_style="Precise and dry.",
            secrets=["She hides a cipher key in the gallery clock."],
            forbidden_knowledge=["The regent ordered the poisoning."],
        )

    def load_scene(self, scene_id: str) -> SceneState:
        return SceneState(
            id=scene_id,
            title="Rose Gallery",
            location="Winter Palace",
            player_visible_summary="Courtiers drift between mirrors and roses.",
            gm_private_summary="A spy waits behind the mirrored column.",
            recent_events=["The regent's envoy left in haste."],
        )


class FakeCritic:
    async def evaluate(self, **_: object) -> CriticResult:
        return CriticResult(accepted=True)

    def build_local_repair_messages(
        self,
        *,
        actor_messages: list[LlmMessage],
        rejected_draft: str,
        issues: list[str],
        repair_instruction: str | None,
    ) -> list[LlmMessage]:
        raise AssertionError("repair should not be used in this test")


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def retrieve_for_actor(self, **kwargs: object) -> list[RetrievedChunk]:
        self.calls.append(kwargs)
        return [
            RetrievedChunk(
                id="public-lore",
                source="demo_lore.md",
                source_type="lore",
                text="The west door has stayed locked for years.",
                score=0.9,
                visibility=Visibility.PLAYER,
            ),
            RetrievedChunk(
                id="gm-lore",
                source="demo_lore.md",
                source_type="lore",
                text="A spy waits behind the mirrored column.",
                score=0.99,
                visibility=Visibility.GM,
            ),
        ]


class FailingRetriever:
    def retrieve_for_actor(self, **_: object) -> list[RetrievedChunk]:
        raise RuntimeError("qdrant offline")


class RecordingFakeMemoryIndexer:
    """Records unindex() calls instead of touching a real vector store."""

    def __init__(self) -> None:
        self.unindexed_calls: list[list[str]] = []

    def unindex(self, memory_ids: list[str]) -> None:
        self.unindexed_calls.append(list(memory_ids))


class RejectingCritic:
    async def evaluate(self, **_: object) -> CriticResult:
        return CriticResult(
            accepted=False,
            issues=["unsafe output"],
            repair_instruction="Remove hidden context.",
        )

    def build_local_repair_messages(
        self,
        *,
        actor_messages: list[LlmMessage],
        rejected_draft: str,
        issues: list[str],
        repair_instruction: str | None,
    ) -> list[LlmMessage]:
        return actor_messages


def _parse_sse(response_text: str) -> list[tuple[str, dict[str, object]]]:
    import json

    events = []
    for frame in response_text.strip().split("\n\n"):
        event_line, data_line = frame.splitlines()
        events.append(
            (
                event_line.removeprefix("event: "),
                json.loads(data_line.removeprefix("data: ")),
            )
        )
    return events


def _non_stage_frames(response_text: str) -> list[tuple[str, dict[str, object]]]:
    # Progress frames (event: stage) now precede the buffered content frames; strip them
    # so assertions about the terminal payload shape don't need to know the stage sequence.
    return [event for event in _parse_sse(response_text) if event[0] != "stage"]


def _build_services(tmp_path: Path) -> tuple[AppServices, SequencedFakeProvider, FakeRetriever]:
    connection = connect_sqlite(tmp_path / "api-turns.db")
    initialize_database(connection)
    session_repository = SQLiteSessionRepository(connection)
    turn_repository = SQLiteTurnRepository(connection)
    session_repository.create_session(
        SessionState(
            id="session-1",
            world_id="demo_world",
            active_scene_id="rose-gallery",
            active_persona_id="archivist",
            player_name="Avery",
        )
    )
    provider = SequencedFakeProvider(["Only archivists and locksmiths speak of that door."])
    retriever = FakeRetriever()
    orchestrator = TurnOrchestrator(
        loader=FakeLoader(),
        provider=provider,
        critic_agent=FakeCritic(),
        session_repository=session_repository,
        turn_repository=turn_repository,
        recent_dialogue_store=RecentDialogueStore(turn_repository=turn_repository, recent_turns=8),
        memory_store=None,
        memory_curator=None,
        actor_context_retriever=retriever,
        config=TurnOrchestratorConfig(
            retrieval_top_k=5,
            max_retrieved_chunk_chars=800,
            local_model="local-model",
            cloud_model="cloud-model",
            local_max_tokens=700,
            cloud_max_tokens=1000,
            local_temperature=0.75,
            cloud_temperature=0.65,
        ),
    )
    return (
        AppServices(
            connection=connection,
            session_repository=session_repository,
            orchestrator=orchestrator,
            recent_dialogue_store=RecentDialogueStore(
                turn_repository=turn_repository,
                recent_turns=8,
            ),
            turn_repository=turn_repository,
        ),
        provider,
        retriever,
    )


def _build_services_with_memory(
    tmp_path: Path,
) -> tuple[AppServices, SequencedFakeProvider, SQLiteMemoryRepository, RecordingFakeMemoryIndexer]:
    # Mirrors _build_services, but wires a real SQLiteMemoryRepository and a
    # recording fake memory_indexer onto AppServices so tests can exercise the
    # memory-deletion + unindex half of DELETE /turns/last (the plain
    # _build_services leaves memory_repository=None, so that path is never hit).
    connection = connect_sqlite(tmp_path / "api-turns-memory.db")
    initialize_database(connection)
    session_repository = SQLiteSessionRepository(connection)
    turn_repository = SQLiteTurnRepository(connection)
    memory_repository = SQLiteMemoryRepository(connection)
    session_repository.create_session(
        SessionState(
            id="session-1",
            world_id="demo_world",
            active_scene_id="rose-gallery",
            active_persona_id="archivist",
            player_name="Avery",
        )
    )
    provider = SequencedFakeProvider(["Only archivists and locksmiths speak of that door."])
    retriever = FakeRetriever()
    memory_indexer = RecordingFakeMemoryIndexer()
    orchestrator = TurnOrchestrator(
        loader=FakeLoader(),
        provider=provider,
        critic_agent=FakeCritic(),
        session_repository=session_repository,
        turn_repository=turn_repository,
        recent_dialogue_store=RecentDialogueStore(turn_repository=turn_repository, recent_turns=8),
        memory_store=None,
        memory_curator=None,
        actor_context_retriever=retriever,
        config=TurnOrchestratorConfig(
            retrieval_top_k=5,
            max_retrieved_chunk_chars=800,
            local_model="local-model",
            cloud_model="cloud-model",
            local_max_tokens=700,
            cloud_max_tokens=1000,
            local_temperature=0.75,
            cloud_temperature=0.65,
        ),
    )
    return (
        AppServices(
            connection=connection,
            session_repository=session_repository,
            orchestrator=orchestrator,
            recent_dialogue_store=RecentDialogueStore(
                turn_repository=turn_repository,
                recent_turns=8,
            ),
            turn_repository=turn_repository,
            memory_repository=memory_repository,
            memory_indexer=cast(MemoryIndexer, memory_indexer),
        ),
        provider,
        memory_repository,
        memory_indexer,
    )


def _build_in_memory_retrieval_services(
    tmp_path: Path,
) -> tuple[AppServices, SequencedFakeProvider]:
    connection = connect_sqlite(tmp_path / "api-in-memory-turns.db")
    initialize_database(connection)
    session_repository = SQLiteSessionRepository(connection)
    turn_repository = SQLiteTurnRepository(connection)
    session_repository.create_session(
        SessionState(
            id="session-1",
            world_id="demo_world",
            active_scene_id="rose-gallery",
            active_persona_id="archivist",
            player_name="Avery",
        )
    )
    embedding_provider = DeterministicKeywordEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    chunks = [
        RagChunk(
            id="public-lore",
            source="demo_lore.md",
            source_type="lore",
            text="The gallery archive mirror marks the locked west door.",
            visibility=Visibility.PLAYER,
            world_id="demo_world",
        ),
        RagChunk(
            id="wrong-world",
            source="other_lore.md",
            source_type="lore",
            text="Another gallery archive mirror marks a west door.",
            visibility=Visibility.PLAYER,
            world_id="other_world",
        ),
        RagChunk(
            id="gm-lore",
            source="gm_lore.md",
            source_type="lore",
            text="A spy waits behind the gallery archive mirror.",
            visibility=Visibility.GM,
            world_id="demo_world",
        ),
    ]
    vector_store.ensure_collection(RagCollection.CANON_LORE, embedding_provider.dimension)
    vector_store.upsert_chunks(
        RagCollection.CANON_LORE,
        chunks,
        embedding_provider.embed_batch([chunk.text for chunk in chunks]),
    )
    provider = SequencedFakeProvider(
        ["Only archivists speak of the gallery mirror and what it marks."]
    )
    recent_dialogue_store = RecentDialogueStore(turn_repository=turn_repository, recent_turns=8)
    orchestrator = TurnOrchestrator(
        loader=FakeLoader(),
        provider=provider,
        critic_agent=FakeCritic(),
        session_repository=session_repository,
        turn_repository=turn_repository,
        recent_dialogue_store=recent_dialogue_store,
        actor_context_retriever=ActorContextRetriever(
            retriever=Retriever(
                embedding_provider=embedding_provider,
                vector_store=vector_store,
            )
        ),
        config=TurnOrchestratorConfig(
            local_model="local-model",
            cloud_model="cloud-model",
            local_max_tokens=700,
            cloud_max_tokens=1000,
            local_temperature=0.75,
            cloud_temperature=0.65,
        ),
    )
    return (
        AppServices(
            connection=connection,
            session_repository=session_repository,
            orchestrator=orchestrator,
            recent_dialogue_store=recent_dialogue_store,
            turn_repository=turn_repository,
        ),
        provider,
    )


def test_post_turn_runs_orchestrator_and_returns_safe_response(tmp_path: Path) -> None:
    services, provider, retriever = _build_services(tmp_path)
    app.dependency_overrides[get_turn_services] = lambda: services
    client = TestClient(app)

    response = client.post(
        "/sessions/session-1/turns",
        json={
            "message": "I ask what the locked door hides.",
            "active_persona_id": "archivist",
        },
    )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    payload = response.json()
    assert payload.pop("stage_timings")
    warnings = payload.pop("warnings")
    errors = payload.pop("errors")
    assert payload == {
        "status": "completed",
        "outcome": "success",
        "text": "Only archivists and locksmiths speak of that door.",
        "route": {
            "provider": "local",
            "model": "local-model",
            "reason": "session provider: local",
        },
        "finish_reason": "stop",
        "memory_written": False,
        "critic_status": "accepted",
        "retrieval": None,
        "token_usage": {"total_tokens": 15},
    }
    # Memory curation runs after this response on API turns (Task 8); the
    # warning is additive and memory_written stays False until the deferred
    # job completes.
    assert warnings == ["memory curation deferred: runs after this response"]
    assert errors == [
        {
            "category": "warning",
            "stage": "general",
            "message": "memory curation deferred: runs after this response",
            "suggestion": None,
        }
    ]
    assert len(provider.requests) == 1
    assert len(retriever.calls) == 1
    prompt = provider.requests[0].messages[0].content
    assert "The west door has stayed locked for years." in prompt
    assert "spy waits behind the mirrored column" not in prompt
    assert "cipher key" not in prompt
    assert "The regent ordered the poisoning." not in prompt
    assert "route_max_tokens" not in response.text


@pytest.mark.asyncio
async def test_api_turn_defers_memory_curation(tmp_path: Path) -> None:
    # API turns (Task 8) return immediately with memory_written=False and a
    # deferred-curation warning; the curator LLM call runs after the response
    # via a fire-and-forget task on app.api.routes._DEFERRED_MEMORY_TASKS. The
    # background job builds its own real AppServices from Settings rather than
    # reusing this test's injected fakes, so the completion guarantee (that
    # run_deferred_memory actually writes memory + updates diagnostics) is
    # covered at the unit level by
    # test_run_deferred_memory_writes_and_updates_diagnostics; here we only
    # assert the additive response contract and that scheduling didn't raise.
    from app.api import routes

    services, _, _ = _build_services(tmp_path)
    app.dependency_overrides[get_turn_services] = lambda: services
    client = TestClient(app)

    response = client.post(
        "/sessions/session-1/turns",
        json={"message": "I ask what the locked door hides."},
    )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    body = response.json()
    assert body["memory_written"] is False
    assert any("memory curation deferred" in w for w in body["warnings"])

    pending = [task for task in routes._DEFERRED_MEMORY_TASKS if not task.done()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


def test_post_turn_response_includes_stage_timings(tmp_path: Path) -> None:
    services, provider, retriever = _build_services(tmp_path)
    app.dependency_overrides[get_turn_services] = lambda: services
    client = TestClient(app)

    response = client.post(
        "/sessions/session-1/turns",
        json={
            "message": "I ask what the locked door hides.",
            "active_persona_id": "archivist",
        },
    )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    timings = response.json()["stage_timings"]
    assert {"retrieval", "routing", "generation", "critique"} <= set(timings)
    assert all(value >= 0.0 for value in timings.values())


def test_post_turn_uses_in_memory_retrieval_without_live_qdrant(tmp_path: Path) -> None:
    services, provider = _build_in_memory_retrieval_services(tmp_path)
    app.dependency_overrides[get_turn_services] = lambda: services
    client = TestClient(app)

    response = client.post(
        "/sessions/session-1/turns",
        json={"message": "What does the gallery archive mirror mark?"},
    )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    prompt = provider.requests[0].messages[0].content
    assert "The gallery archive mirror marks the locked west door." in prompt
    assert "Another gallery archive mirror marks a west door." not in prompt
    assert "A spy waits behind the gallery archive mirror." not in prompt
    retrieval = response.json()["retrieval"]
    assert retrieval is not None
    assert [entry["id"] for entry in retrieval["selected"]] == ["public-lore"]
    assert retrieval["selected"][0]["collection"] == "canon_lore"
    assert retrieval["selected"][0]["selected_rank"] == 1
    assert "text" not in retrieval["selected"][0]
    assert retrieval["rejected"] == []


def test_post_turn_returns_404_for_missing_session(tmp_path: Path) -> None:
    services, _, _ = _build_services(tmp_path)
    app.dependency_overrides[get_turn_services] = lambda: services
    client = TestClient(app)

    response = client.post(
        "/sessions/missing-session/turns",
        json={"message": "Hello there."},
    )

    app.dependency_overrides.clear()
    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "session_not_found",
            "message": "Unknown session id: missing-session",
            "details": [],
        }
    }


def test_post_turn_rejects_persona_override_for_unknown_persona_with_400_envelope(
    tmp_path: Path,
) -> None:
    services, _, _ = _build_services(tmp_path)
    app.dependency_overrides[get_turn_services] = lambda: services
    client = TestClient(app)

    response = client.post(
        "/sessions/session-1/turns",
        json={"message": "Hello there.", "active_persona_id": "someone-else"},
    )

    app.dependency_overrides.clear()
    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "invalid_turn_request",
            "message": "Unknown persona for world demo_world: someone-else",
            "details": [],
        }
    }


def test_post_turn_with_known_persona_override_switches_the_session_persona(
    tmp_path: Path,
) -> None:
    services, _, _ = _build_services(tmp_path)
    app.dependency_overrides[get_turn_services] = lambda: services
    client = TestClient(app)

    response = client.post(
        "/sessions/session-1/turns",
        json={"message": "Hello there.", "active_persona_id": "warden"},
    )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    reloaded = services.session_repository.get_session("session-1")
    assert reloaded is not None
    assert reloaded.active_persona_id == "warden"


def test_post_turn_returns_retrieval_failure_warning_in_success_response(tmp_path: Path) -> None:
    services, _, _ = _build_services(tmp_path)
    services.orchestrator.actor_context_retriever = FailingRetriever()
    app.dependency_overrides[get_turn_services] = lambda: services
    client = TestClient(app)

    response = client.post(
        "/sessions/session-1/turns",
        json={"message": "Hello there."},
    )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["warnings"] == [
        "retrieval skipped: qdrant offline",
        "memory curation deferred: runs after this response",
    ]


def test_post_turn_includes_structured_errors_derived_from_warnings(tmp_path: Path) -> None:
    services, _, _ = _build_services(tmp_path)
    services.orchestrator.actor_context_retriever = FailingRetriever()
    app.dependency_overrides[get_turn_services] = lambda: services
    client = TestClient(app)

    response = client.post(
        "/sessions/session-1/turns",
        json={"message": "Hello there."},
    )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    payload = response.json()
    assert payload["warnings"] == [
        "retrieval skipped: qdrant offline",
        "memory curation deferred: runs after this response",
    ]
    # A second, unrelated error is derived from the deferred-memory warning
    # (Task 8); this test only cares about the retrieval-derived one.
    assert len(payload["errors"]) == 2
    error = payload["errors"][0]
    assert error["category"] == "degraded"
    assert error["stage"] == "retrieval"
    assert error["message"] == "retrieval skipped: qdrant offline"
    assert "vector store (Qdrant)" in error["suggestion"]


def test_post_turn_rejects_the_removed_cloud_request_flags(tmp_path: Path) -> None:
    # The per-turn confirm flow is gone: request_cloud/cloud_confirmed/force_local are
    # no longer part of the contract, and CreateTurnRequest is extra="forbid".
    services, _, _ = _build_services(tmp_path)
    app.dependency_overrides[get_turn_services] = lambda: services
    client = TestClient(app)

    response = client.post(
        "/sessions/session-1/turns",
        json={"message": "Hello there.", "request_cloud": True},
    )

    app.dependency_overrides.clear()
    assert response.status_code == 422


def test_post_turn_response_does_not_expose_hidden_context_or_prompt_text(tmp_path: Path) -> None:
    services, _, _ = _build_services(tmp_path)
    app.dependency_overrides[get_turn_services] = lambda: services
    client = TestClient(app)

    response = client.post(
        "/sessions/session-1/turns",
        json={"message": "I ask what the locked door hides."},
    )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    serialized = response.text
    assert "The west door has stayed locked for years." not in serialized
    assert "spy waits behind the mirrored column" not in serialized
    assert "cipher key" not in serialized
    assert "The regent ordered the poisoning." not in serialized
    assert "route_max_tokens" not in serialized


def test_post_turn_rejects_invalid_request_with_422(tmp_path: Path) -> None:
    services, _, _ = _build_services(tmp_path)
    app.dependency_overrides[get_turn_services] = lambda: services
    client = TestClient(app)

    response = client.post(
        "/sessions/session-1/turns",
        json={"message": ""},
    )

    app.dependency_overrides.clear()
    assert response.status_code == 422
    payload = response.json()
    assert payload["error"]["code"] == "validation_error"
    assert payload["error"]["message"] == "Request validation failed"
    assert payload["error"]["details"]


def test_post_turn_rejects_message_exceeding_max_length(tmp_path: Path) -> None:
    services, _, _ = _build_services(tmp_path)
    app.dependency_overrides[get_turn_services] = lambda: services
    client = TestClient(app)

    response = client.post(
        "/sessions/session-1/turns",
        json={"message": "x" * 4001},
    )

    app.dependency_overrides.clear()
    assert response.status_code == 422
    payload = response.json()
    assert payload["error"]["code"] == "validation_error"
    assert payload["error"]["message"] == "Request validation failed"
    assert payload["error"]["details"]


def test_post_turn_rejects_active_persona_id_exceeding_max_length(tmp_path: Path) -> None:
    services, _, _ = _build_services(tmp_path)
    app.dependency_overrides[get_turn_services] = lambda: services
    client = TestClient(app)

    response = client.post(
        "/sessions/session-1/turns",
        json={"message": "Hello.", "active_persona_id": "x" * 201},
    )

    app.dependency_overrides.clear()
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_post_turn_stream_returns_buffered_text_then_final_metadata(tmp_path: Path) -> None:
    services, _, _ = _build_services(tmp_path)
    app.dependency_overrides[get_turn_services] = lambda: services
    client = TestClient(app)

    response = client.post(
        "/sessions/session-1/turns/stream",
        json={"message": "I ask what the locked door hides."},
    )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = _non_stage_frames(response.text)
    assert frames[1][1].pop("stage_timings")
    assert frames == [
        ("text", {"text": "Only archivists and locksmiths speak of that door."}),
        (
            "final",
            {
                "route": {
                    "provider": "local",
                    "model": "local-model",
                    "reason": "session provider: local",
                },
                "finish_reason": "stop",
                "memory_written": False,
                "critic_status": "accepted",
                "warnings": ["memory curation deferred: runs after this response"],
                "errors": [
                    {
                        "category": "warning",
                        "stage": "general",
                        "message": "memory curation deferred: runs after this response",
                        "suggestion": None,
                    }
                ],
                "retrieval": None,
                "token_usage": {"total_tokens": 15},
            },
        ),
    ]


def test_stream_turn_emits_stage_frames_before_final(tmp_path: Path) -> None:
    services, _, _ = _build_services(tmp_path)
    app.dependency_overrides[get_turn_services] = lambda: services
    client = TestClient(app)

    response = client.post(
        "/sessions/session-1/turns/stream",
        json={"message": "I ask what the locked door hides."},
    )

    app.dependency_overrides.clear()
    body = response.text
    assert "event: stage" in body
    assert body.index("event: stage") < body.index("event: final")
    stages = [event[1]["stage"] for event in _parse_sse(body) if event[0] == "stage"]
    assert stages[:4] == ["session", "retrieval", "routing", "generation"]
    # API turns defer memory curation until after the response (Task 8), so the
    # orchestrator no longer emits a "memory" stage frame on this path.
    assert stages[-1] == "persistence"
    assert "memory" not in stages


def test_post_turn_stream_emits_error_frame_for_missing_session(tmp_path: Path) -> None:
    # Once streaming has started the HTTP status is committed to 200; a session lookup
    # failure now arrives as a terminal `event: error` frame carrying the original code
    # and status instead of a 404 JSON envelope.
    services, _, _ = _build_services(tmp_path)
    app.dependency_overrides[get_turn_services] = lambda: services
    client = TestClient(app)

    response = client.post(
        "/sessions/missing-session/turns/stream",
        json={"message": "Hello there."},
    )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _non_stage_frames(response.text)
    assert events[-1][0] == "error"
    assert events[-1][1]["code"] == "session_not_found"
    assert events[-1][1]["status"] == 404


def test_post_turn_stream_emits_error_frame_for_invalid_turn_request(tmp_path: Path) -> None:
    services, _, _ = _build_services(tmp_path)
    app.dependency_overrides[get_turn_services] = lambda: services
    client = TestClient(app)

    response = client.post(
        "/sessions/session-1/turns/stream",
        json={"message": "Hello there.", "active_persona_id": "someone-else"},
    )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _non_stage_frames(response.text)
    assert events[-1][0] == "error"
    assert events[-1][1]["code"] == "invalid_turn_request"
    assert events[-1][1]["status"] == 400


def test_post_turn_stream_returns_sanitized_json_422_before_streaming(tmp_path: Path) -> None:
    services, _, _ = _build_services(tmp_path)
    app.dependency_overrides[get_turn_services] = lambda: services
    client = TestClient(app)

    response = client.post(
        "/sessions/session-1/turns/stream",
        json={"message": ["/Users/private/secret.txt"]},
    )

    app.dependency_overrides.clear()
    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["details"] == [
        {
            "loc": ["body", "message"],
            "type": "string_type",
            "message": "Request field validation failed",
        }
    ]
    assert "/Users/private/secret.txt" not in response.text


def test_post_turn_stream_sanitizes_path_like_validation_location(tmp_path: Path) -> None:
    services, _, _ = _build_services(tmp_path)
    app.dependency_overrides[get_turn_services] = lambda: services
    client = TestClient(app)

    response = client.post(
        "/sessions/session-1/turns/stream",
        json={"message": "Hello there.", "/Users/private/secret.txt": "reflected-key"},
    )

    app.dependency_overrides.clear()
    assert response.status_code == 422
    assert response.json()["error"]["details"] == [
        {
            "loc": ["body", "<field>"],
            "type": "extra_forbidden",
            "message": "Request field validation failed",
        }
    ]
    assert "/Users/private/secret.txt" not in response.text


def test_post_turn_stream_terminal_event_includes_fail_open_warnings(tmp_path: Path) -> None:
    services, _, _ = _build_services(tmp_path)
    services.orchestrator.actor_context_retriever = FailingRetriever()
    app.dependency_overrides[get_turn_services] = lambda: services
    client = TestClient(app)

    response = client.post(
        "/sessions/session-1/turns/stream",
        json={"message": "Hello there."},
    )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    events = _parse_sse(response.text)
    assert events[-1][0] == "final"
    warnings = cast("list[str]", events[-1][1]["warnings"])
    assert "retrieval skipped: qdrant offline" in warnings


def test_post_turn_stream_does_not_expose_input_or_hidden_context(tmp_path: Path) -> None:
    services, _, _ = _build_services(tmp_path)
    app.dependency_overrides[get_turn_services] = lambda: services
    client = TestClient(app)

    response = client.post(
        "/sessions/session-1/turns/stream",
        json={"message": "RAW PLAYER PROMPT"},
    )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    assert "RAW PLAYER PROMPT" not in response.text
    assert "The west door has stayed locked for years." not in response.text
    assert "spy waits behind the mirrored column" not in response.text
    assert "cipher key" not in response.text
    assert "The regent ordered the poisoning." not in response.text


def test_post_turn_stream_reconstructs_non_streaming_response(tmp_path: Path) -> None:
    json_services, _, _ = _build_services(tmp_path / "json")
    app.dependency_overrides[get_turn_services] = lambda: json_services
    client = TestClient(app)
    json_response = client.post(
        "/sessions/session-1/turns",
        json={"message": "Hello there."},
    )
    json_services.close()

    stream_services, _, _ = _build_services(tmp_path / "stream")
    app.dependency_overrides[get_turn_services] = lambda: stream_services
    stream_response = client.post(
        "/sessions/session-1/turns/stream",
        json={"message": "Hello there."},
    )

    app.dependency_overrides.clear()
    text_event, final_event = _non_stage_frames(stream_response.text)
    reconstructed = {"text": text_event[1]["text"], **final_event[1]}
    json_payload = json_response.json()
    assert reconstructed.pop("stage_timings")
    assert json_payload.pop("stage_timings")
    assert json_payload.pop("status") == "completed"
    # The final SSE frame doesn't carry outcome (failure frames do); the JSON body does.
    assert json_payload.pop("outcome") == "success"
    assert reconstructed == json_payload


class _StallingOrchestrator:
    """Stands in for TurnOrchestrator.run_turn: blocks until cancelled and records
    whether cancellation actually propagated into the running turn, so the test can
    assert the disconnect path leaves nothing running."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = False

    async def run_turn(
        self, *, turn_input: object, on_stage: object = None, defer_memory: bool = False
    ) -> object:
        self.started.set()
        try:
            await asyncio.Event().wait()  # blocks forever unless cancelled
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        raise AssertionError("unreachable: run_turn should only end via cancellation")


@pytest.mark.asyncio
async def test_event_stream_cancels_in_flight_turn_on_client_disconnect(
    tmp_path: Path,
) -> None:
    # Regression test: Starlette closes the `event_stream` async generator
    # (GeneratorExit) when a client disconnects mid-stream. Before the fix, nothing
    # cancelled the orchestrator task, so it kept running orphaned and held the
    # request-scoped SQLite connection open past the request's lifetime.
    services, _, _ = _build_services(tmp_path)
    stalling_orchestrator = _StallingOrchestrator()
    services.orchestrator = stalling_orchestrator  # type: ignore[assignment]

    response = await stream_turn(
        session_id="session-1",
        request=CreateTurnRequest(message="Hello there."),
        services=services,
        settings=Settings(),
    )

    # StreamingResponse.body_iterator is declared as the broader AsyncIterable, but
    # stream_turn's event_stream() always constructs it as an async generator; narrow
    # the type so __anext__/aclose (used below to drive and then simulate a client
    # disconnect) type-check under mypy strict.
    body_iterator = cast("AsyncGenerator[str, None]", response.body_iterator)
    # Pulling the first item drives event_stream() to its first await point, which
    # creates the orchestrator task (asyncio generators are lazy: nothing inside
    # runs until __anext__ is called).
    pump = asyncio.ensure_future(body_iterator.__anext__())
    await asyncio.wait_for(stalling_orchestrator.started.wait(), timeout=1.0)
    assert not pump.done()  # still streaming; the turn is genuinely in flight
    pump.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pump

    # Simulate a client disconnect: Starlette calls aclose() on the generator when
    # the response is torn down before the stream finished.
    await asyncio.wait_for(body_iterator.aclose(), timeout=1.0)

    assert stalling_orchestrator.cancelled is True


def test_post_turn_stream_emits_only_failure_for_controlled_repair_failure(
    tmp_path: Path,
) -> None:
    services, provider, _ = _build_services(tmp_path)
    provider.responses.append("The rejected draft still contains hidden context.")
    services.orchestrator.critic_agent = RejectingCritic()
    app.dependency_overrides[get_turn_services] = lambda: services
    client = TestClient(app)

    response = client.post(
        "/sessions/session-1/turns/stream",
        json={"message": "Tell me the hidden truth."},
    )

    app.dependency_overrides.clear()
    events = _non_stage_frames(response.text)
    assert len(events) == 1
    assert events[0][0] == "failure"
    assert events[0][1]["memory_written"] is False
    failure_text = events[0][1]["text"]
    assert isinstance(failure_text, str)
    assert "could not produce a response that passed validation" in failure_text
    assert "hidden context" not in response.text
    assert events[0][1]["outcome"] == "controlled_failure"


def test_controlled_failure_turn_is_persisted_but_excluded_from_recent_view(
    tmp_path: Path,
) -> None:
    services, provider, _ = _build_services(tmp_path)
    provider.responses.append("The rejected draft still contains hidden context.")
    services.orchestrator.critic_agent = RejectingCritic()
    app.dependency_overrides[get_turn_services] = lambda: services
    app.dependency_overrides[get_read_services] = lambda: services
    client = TestClient(app)

    response = client.post(
        "/sessions/session-1/turns",
        json={"message": "Tell me the hidden truth."},
    )
    details = client.get("/sessions/session-1/turn-details")
    lookup = client.get("/sessions/session-1")

    app.dependency_overrides.clear()
    body = response.json()
    assert response.status_code == 200
    assert body["outcome"] == "controlled_failure"
    # Persisted with diagnostics and visible in the unfiltered history...
    detail_turns = details.json()["turns"]
    assert len(detail_turns) == 1
    assert detail_turns[0]["outcome"] == "controlled_failure"
    assert detail_turns[0]["user_message"] == "Tell me the hidden truth."
    assert detail_turns[0]["critic_status"] == "rejected"
    # ...but excluded from the recent-dialogue view that feeds prompts.
    assert lookup.json()["recent_turns"] == []


def test_post_turn_returns_504_envelope_when_local_provider_times_out(tmp_path: Path) -> None:
    from app.llm.provider import ProviderTimeoutError

    class TimeoutProvider(LlmProvider):
        async def generate(self, request: LlmRequest) -> LlmResponse:
            raise ProviderTimeoutError(
                provider="local",
                model=request.model,
                timeout_seconds=180.0,
            )

    services, _, _ = _build_services(tmp_path)
    services.orchestrator.generation_stage.provider = TimeoutProvider()
    app.dependency_overrides[get_turn_services] = lambda: services
    client = TestClient(app)

    response = client.post(
        "/sessions/session-1/turns",
        json={"message": "I ask what the locked door hides."},
    )

    app.dependency_overrides.clear()
    assert response.status_code == 504
    payload = response.json()
    assert payload["error"]["code"] == "provider_timeout"
    assert "timed out" in payload["error"]["message"]
    assert payload["error"]["details"] == []


def test_post_turn_returns_503_envelope_when_local_provider_unavailable(tmp_path: Path) -> None:
    from app.llm.provider import ProviderUnavailableError

    class UnavailableProvider(LlmProvider):
        async def generate(self, request: LlmRequest) -> LlmResponse:
            raise ProviderUnavailableError(
                provider="local",
                model=request.model,
                base_url="http://127.0.0.1:8080/v1",
            )

    services, _, _ = _build_services(tmp_path)
    services.orchestrator.generation_stage.provider = UnavailableProvider()
    app.dependency_overrides[get_turn_services] = lambda: services
    client = TestClient(app)

    response = client.post(
        "/sessions/session-1/turns",
        json={"message": "I ask what the locked door hides."},
    )

    app.dependency_overrides.clear()
    assert response.status_code == 503
    payload = response.json()
    assert payload["error"]["code"] == "provider_unavailable"
    assert payload["error"]["details"] == []


def test_api_get_turn_detail_returns_stored_fields_and_diagnostics(tmp_path: Path) -> None:
    services, _ = _build_in_memory_retrieval_services(tmp_path)
    app.dependency_overrides[get_turn_services] = lambda: services
    app.dependency_overrides[get_read_services] = lambda: services
    client = TestClient(app)

    create = client.post(
        "/sessions/session-1/turns",
        json={"message": "What does the gallery archive mirror mark?"},
    )
    assert create.status_code == 200

    response = client.get("/sessions/session-1/turns/1")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["turn_index"] == 1
    assert payload["scene_id"] == "rose-gallery"
    assert payload["persona_id"] == "archivist"
    assert payload["user_message"] == "What does the gallery archive mirror mark?"
    assert payload["assistant_message"]
    assert payload["route"] == {
        "provider": "local",
        "model": "local-model",
        "reason": "session provider: local",
    }
    assert "T" in payload["created_at"]
    assert payload["finish_reason"] == "stop"
    assert payload["memory_written"] is False
    assert payload["critic_status"] == "accepted"
    assert payload["warnings"] == ["memory curation deferred: runs after this response"]
    assert {"retrieval", "routing", "generation", "critique"} <= set(payload["stage_timings"])

    retrieval = payload["retrieval"]
    assert retrieval is not None
    assert retrieval["query"]
    candidates = retrieval["selected"] + retrieval["rejected"]
    assert candidates
    for candidate in candidates:
        # Retrieval diagnostics are metadata-only and never carry chunk text.
        assert "text" not in candidate
        assert "chunk_text" not in candidate
        assert set(candidate) <= {
            "id",
            "source",
            "source_type",
            "collection",
            "visibility",
            "tags",
            "original_score",
            "adjusted_score",
            "applied_boosts",
            "selected_rank",
        }


def test_api_get_turn_detail_returns_404_for_unknown_session(tmp_path: Path) -> None:
    services, _, _ = _build_services(tmp_path)
    app.dependency_overrides[get_read_services] = lambda: services
    client = TestClient(app)

    response = client.get("/sessions/nonexistent/turns/1")
    app.dependency_overrides.clear()

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "session_not_found"


def test_delete_last_turn_reroll_flow(tmp_path: Path) -> None:
    services, _, _ = _build_services(tmp_path)
    app.dependency_overrides[get_turn_services] = lambda: services
    client = TestClient(app)

    create = client.post(
        "/sessions/session-1/turns",
        json={"message": "I ask what the locked door hides."},
    )
    assert create.status_code == 200

    response = client.delete("/sessions/session-1/turns/last")

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == "session-1"
    assert body["deleted_turn_index"] == 1
    assert body["user_message"] == "I ask what the locked door hides."
    assert body["deleted_memory_count"] == 0

    second = client.delete("/sessions/session-1/turns/last")
    app.dependency_overrides.clear()
    assert second.status_code == 404


def test_delete_last_turn_reroll_flow_deletes_memories_and_unindexes(
    tmp_path: Path,
) -> None:
    # test_delete_last_turn_reroll_flow above only exercises deleted_memory_count == 0
    # because _build_services wires no memory_repository, so DELETE /turns/last never
    # touches the memory-deletion branch. This sibling test wires a real
    # SQLiteMemoryRepository and a recording fake memory_indexer so both the
    # SQLite deletion and the vector-store unindex() call are actually verified.
    services, _, memory_repository, memory_indexer = _build_services_with_memory(tmp_path)
    app.dependency_overrides[get_turn_services] = lambda: services
    client = TestClient(app)

    create = client.post(
        "/sessions/session-1/turns",
        json={"message": "I ask what the locked door hides."},
    )
    assert create.status_code == 200

    assert services.turn_repository is not None
    stored_turn = services.turn_repository.list_all_turns("session-1")[0]
    assert stored_turn.created_at is not None

    # Insert a memory episode timestamped at/after the turn's created_at, mirroring
    # how the orchestrator would have written it after generating the turn.
    memory_created_at = stored_turn.created_at
    memory_id = "mem-from-turn-1"
    services.connection.execute(
        """
        INSERT INTO memory_episodes (
            id, session_id, scene_id, actor_id, summary, importance,
            visibility, tags_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            memory_id,
            "session-1",
            "rose-gallery",
            "archivist",
            "The archivist mentioned the locked door.",
            3,
            Visibility.PLAYER.value,
            "[]",
            serialize_datetime(memory_created_at),
        ),
    )
    services.connection.commit()
    assert [m.id for m in memory_repository.list_memories_for_session("session-1")] == [
        memory_id
    ]

    response = client.delete("/sessions/session-1/turns/last")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["deleted_memory_count"] == 1
    assert memory_repository.list_memories_for_session("session-1") == []
    assert memory_indexer.unindexed_calls == [[memory_id]]


@pytest.mark.asyncio
async def test_delete_last_turn_drains_in_flight_deferred_memory_job_before_sweeping(
    tmp_path: Path,
) -> None:
    # Regression test for the reroll-vs-deferred-curation race: if a deferred memory
    # job for the turn being deleted is still in flight when DELETE /turns/last runs,
    # the job can write that turn's memories AFTER the sweep, resurrecting them.
    # delete_last_turn must drain (await) every pending _DEFERRED_MEMORY_TASKS entry
    # FIRST, so the job's write lands before delete_memories_since runs and gets
    # deleted along with the rest of the turn's memories.
    #
    # Calls routes.delete_last_turn directly (like
    # test_event_stream_cancels_in_flight_turn_on_client_disconnect does for
    # stream_turn) rather than through TestClient, so the in-flight task and the
    # endpoint call share the same event loop -- TestClient drives requests on its
    # own portal loop, which would make a task inserted from this test's loop
    # unawaitable from inside the endpoint.
    from app.api import routes

    services, _, memory_repository, memory_indexer = _build_services_with_memory(tmp_path)
    app.dependency_overrides[get_turn_services] = lambda: services
    client = TestClient(app)

    create = client.post(
        "/sessions/session-1/turns",
        json={"message": "I ask what the locked door hides."},
    )
    assert create.status_code == 200
    app.dependency_overrides.clear()

    assert services.turn_repository is not None
    stored_turn = services.turn_repository.list_all_turns("session-1")[0]
    assert stored_turn.created_at is not None

    # A controllable stand-in for a still-running deferred memory job: it blocks on
    # an Event and, once released, writes a memory row timestamped at the deleted
    # turn -- exactly what the real deferred job would do. Inserted directly into
    # _DEFERRED_MEMORY_TASKS to simulate "job scheduled, not yet finished" without
    # depending on real curation timing.
    release = asyncio.Event()
    memory_id = "mem-from-in-flight-job"

    async def slow_deferred_job() -> None:
        await release.wait()
        services.connection.execute(
            """
            INSERT INTO memory_episodes (
                id, session_id, scene_id, actor_id, summary, importance,
                visibility, tags_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory_id,
                "session-1",
                "rose-gallery",
                "archivist",
                "Written by the in-flight deferred job.",
                3,
                Visibility.PLAYER.value,
                "[]",
                serialize_datetime(stored_turn.created_at),
            ),
        )
        services.connection.commit()

    task = asyncio.ensure_future(slow_deferred_job())
    routes._DEFERRED_MEMORY_TASKS.add(task)
    task.add_done_callback(routes._DEFERRED_MEMORY_TASKS.discard)

    # Release the job shortly after delete_last_turn starts awaiting the drain, so
    # the ordering is genuinely exercised rather than the job finishing beforehand.
    async def release_soon() -> None:
        await asyncio.sleep(0.01)
        release.set()

    releaser = asyncio.ensure_future(release_soon())
    body = await routes.delete_last_turn(session_id="session-1", services=services)
    await releaser

    assert task.done()  # the drain awaited the job to completion

    # If delete_last_turn had swept BEFORE draining, this row (written by the job
    # after release) would still be present. Draining first means the sweep's
    # delete_memories_since call happens after the write, so it is included.
    assert body.deleted_memory_count == 1
    assert memory_repository.list_memories_for_session("session-1") == []
    assert memory_indexer.unindexed_calls == [[memory_id]]


def test_delete_last_turn_returns_404_for_unknown_session(tmp_path: Path) -> None:
    services, _, _ = _build_services(tmp_path)
    app.dependency_overrides[get_turn_services] = lambda: services
    client = TestClient(app)

    response = client.delete("/sessions/missing-session/turns/last")

    app.dependency_overrides.clear()
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "session_not_found"


def test_delete_last_turn_restores_pre_override_persona(tmp_path: Path) -> None:
    # Regression test for #66: TurnOrchestrator.run_turn commits a per-turn persona
    # override to sessions.active_persona_id only AFTER the turn persists
    # successfully. Rerolling (deleting) that turn must undo the commit, or the
    # session is stranded on a persona the surviving history never switched to.
    #
    # A first, non-switching turn establishes "archivist" as a value recorded on a
    # surviving turn -- the switch commit made by the second (rerolled) turn must
    # restore exactly that value. (A switch on the very first turn of a session is
    # a separate, documented edge case: see
    # test_restore_persona_after_turn_delete_leaves_unrecoverable_creation_persona_alone
    # in tests/unit/test_repositories.py.)
    services, provider, _ = _build_services(tmp_path)
    provider.responses.append("I am the warden of this wing.")
    app.dependency_overrides[get_turn_services] = lambda: services
    client = TestClient(app)

    before_switch = services.session_repository.get_session("session-1")
    assert before_switch is not None
    assert before_switch.active_persona_id == "archivist"

    first = client.post("/sessions/session-1/turns", json={"message": "Hello there."})
    assert first.status_code == 200

    switch = client.post(
        "/sessions/session-1/turns",
        json={"message": "Who are you really?", "active_persona_id": "warden"},
    )
    assert switch.status_code == 200
    after_switch = services.session_repository.get_session("session-1")
    assert after_switch is not None
    assert after_switch.active_persona_id == "warden"

    response = client.delete("/sessions/session-1/turns/last")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    reloaded = services.session_repository.get_session("session-1")
    assert reloaded is not None
    assert reloaded.active_persona_id == "archivist"


def test_delete_last_turn_without_switch_leaves_active_persona_untouched(
    tmp_path: Path,
) -> None:
    # Sibling to test_delete_last_turn_restores_pre_override_persona: a reroll of a
    # turn that never requested a persona override must be a no-op for
    # active_persona_id, even when a remaining prior turn shares that same persona.
    services, provider, _ = _build_services(tmp_path)
    provider.responses.append("Only archivists and locksmiths speak of that door.")
    app.dependency_overrides[get_turn_services] = lambda: services
    client = TestClient(app)

    first = client.post("/sessions/session-1/turns", json={"message": "Hello there."})
    assert first.status_code == 200
    second = client.post("/sessions/session-1/turns", json={"message": "And then?"})
    assert second.status_code == 200
    before_delete = services.session_repository.get_session("session-1")
    assert before_delete is not None
    assert before_delete.active_persona_id == "archivist"

    response = client.delete("/sessions/session-1/turns/last")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    reloaded = services.session_repository.get_session("session-1")
    assert reloaded is not None
    assert reloaded.active_persona_id == "archivist"


def test_delete_last_turn_does_not_revert_an_explicit_scene_switch(tmp_path: Path) -> None:
    # Scoping guard for #66: unlike persona, a scene never changes as a per-turn
    # side effect -- TurnInput has no scene field, and active_scene_id only ever
    # changes via the explicit POST /sessions/{id}/scene endpoint. A scene switch
    # made after the last turn is a deliberate, independent action and must
    # SURVIVE a reroll of that turn, not be undone by one.
    services, _, _ = _build_services(tmp_path)
    app.dependency_overrides[get_turn_services] = lambda: services
    app.dependency_overrides[get_read_services] = lambda: services
    client = TestClient(app)

    create = client.post("/sessions/session-1/turns", json={"message": "Hello there."})
    assert create.status_code == 200

    scene_switch = client.post("/sessions/session-1/scene", json={"scene_id": "east-wing"})
    assert scene_switch.status_code == 200

    response = client.delete("/sessions/session-1/turns/last")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    reloaded = services.session_repository.get_session("session-1")
    assert reloaded is not None
    assert reloaded.active_scene_id == "east-wing"


def test_api_get_turn_detail_returns_404_for_unknown_turn_index(tmp_path: Path) -> None:
    services, _ = _build_in_memory_retrieval_services(tmp_path)
    app.dependency_overrides[get_turn_services] = lambda: services
    app.dependency_overrides[get_read_services] = lambda: services
    client = TestClient(app)

    create = client.post(
        "/sessions/session-1/turns",
        json={"message": "What does the gallery archive mirror mark?"},
    )
    assert create.status_code == 200

    response = client.get("/sessions/session-1/turns/999")
    app.dependency_overrides.clear()

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "turn_not_found"
