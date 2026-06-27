from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.api.routes import get_read_services, get_turn_services
from app.composition import AppServices
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
from app.llm.router import CloudMode
from app.main import app
from app.memory import RecentDialogueStore
from app.orchestration.turn_orchestrator import TurnOrchestrator
from app.persistence import DemoWorldRecord, SQLiteSessionRepository, SQLiteTurnRepository
from app.persistence.sqlite import connect_sqlite, initialize_database
from app.rag import ActorContextRetriever, InMemoryVectorStore, RagChunk, RagCollection, Retriever


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
            persona_ids=["archivist"],
            scene_ids=["rose-gallery"],
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

    def build_cloud_repair_messages(
        self,
        *,
        actor_messages: list[LlmMessage],
        issues: list[str],
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

    def build_cloud_repair_messages(
        self,
        *,
        actor_messages: list[LlmMessage],
        issues: list[str],
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
        retrieval_top_k=5,
        max_retrieved_chunk_chars=800,
        local_model="local-model",
        cloud_model="cloud-model",
        local_max_tokens=700,
        cloud_max_tokens=1000,
        local_temperature=0.75,
        cloud_temperature=0.65,
        cloud_mode="ask",
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
        local_model="local-model",
        cloud_model="cloud-model",
        local_max_tokens=700,
        cloud_max_tokens=1000,
        local_temperature=0.75,
        cloud_temperature=0.65,
        cloud_mode="ask",
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
            "request_cloud": False,
        },
    )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    payload = response.json()
    assert payload.pop("stage_timings")
    assert payload == {
        "status": "completed",
        "text": "Only archivists and locksmiths speak of that door.",
        "route": {
            "provider": "local",
            "model": "local-model",
            "reason": "default local route",
        },
        "finish_reason": "stop",
        "memory_written": False,
        "critic_status": "accepted",
        "warnings": [],
        "retrieval": None,
    }
    assert len(provider.requests) == 1
    assert len(retriever.calls) == 1
    prompt = provider.requests[0].messages[0].content
    assert "The west door has stayed locked for years." in prompt
    assert "spy waits behind the mirrored column" not in prompt
    assert "cipher key" not in prompt
    assert "The regent ordered the poisoning." not in prompt
    assert "route_max_tokens" not in response.text


def test_post_turn_response_includes_stage_timings(tmp_path: Path) -> None:
    services, provider, retriever = _build_services(tmp_path)
    app.dependency_overrides[get_turn_services] = lambda: services
    client = TestClient(app)

    response = client.post(
        "/sessions/session-1/turns",
        json={
            "message": "I ask what the locked door hides.",
            "active_persona_id": "archivist",
            "request_cloud": False,
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


def test_post_turn_rejects_persona_override_mismatch_with_400_envelope(tmp_path: Path) -> None:
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
            "message": "Turn persona override does not match the stored session persona",
            "details": [],
        }
    }


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
    assert response.json()["warnings"] == ["retrieval skipped: qdrant offline"]


def test_post_turn_cloud_request_in_ask_mode_returns_confirmation_required(
    tmp_path: Path,
) -> None:
    services, provider, _ = _build_services(tmp_path)
    app.dependency_overrides[get_turn_services] = lambda: services
    client = TestClient(app)

    response = client.post(
        "/sessions/session-1/turns",
        json={"message": "Hello there.", "request_cloud": True},
    )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "confirmation_required"
    assert payload["text"] == ""
    assert payload["route"]["provider"] == "cloud"
    assert payload["route"]["reason"] == "user requested cloud"
    assert provider.requests == []


def test_post_turn_force_local_declines_cloud_route(tmp_path: Path) -> None:
    services, provider, _ = _build_services(tmp_path)
    app.dependency_overrides[get_turn_services] = lambda: services
    client = TestClient(app)

    response = client.post(
        "/sessions/session-1/turns",
        json={"message": "Hello there.", "request_cloud": True, "force_local": True},
    )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["route"]["provider"] == "local"
    assert payload["route"]["reason"] == "user declined cloud"
    assert len(provider.requests) == 1


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
    frames = _parse_sse(response.text)
    assert frames[1][1].pop("stage_timings")
    assert frames == [
        ("text", {"text": "Only archivists and locksmiths speak of that door."}),
        (
            "final",
            {
                "route": {
                    "provider": "local",
                    "model": "local-model",
                    "reason": "default local route",
                },
                "finish_reason": "stop",
                "memory_written": False,
                "critic_status": "accepted",
                "warnings": [],
                "retrieval": None,
            },
        ),
    ]


def test_post_turn_stream_returns_json_404_before_streaming(tmp_path: Path) -> None:
    services, _, _ = _build_services(tmp_path)
    app.dependency_overrides[get_turn_services] = lambda: services
    client = TestClient(app)

    response = client.post(
        "/sessions/missing-session/turns/stream",
        json={"message": "Hello there."},
    )

    app.dependency_overrides.clear()
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["code"] == "session_not_found"


def test_post_turn_stream_returns_json_400_before_streaming(tmp_path: Path) -> None:
    services, _, _ = _build_services(tmp_path)
    app.dependency_overrides[get_turn_services] = lambda: services
    client = TestClient(app)

    response = client.post(
        "/sessions/session-1/turns/stream",
        json={"message": "Hello there.", "active_persona_id": "someone-else"},
    )

    app.dependency_overrides.clear()
    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["code"] == "invalid_turn_request"


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
        json={"message": "Hello there.", "request_cloud": True},
    )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    events = _parse_sse(response.text)
    assert events[-1][0] == "confirmation_required"
    assert events[-1][1]["status"] == "confirmation_required"
    assert events[-1][1]["warnings"] == ["retrieval skipped: qdrant offline"]


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
    text_event, final_event = _parse_sse(stream_response.text)
    reconstructed = {"text": text_event[1]["text"], **final_event[1]}
    json_payload = json_response.json()
    assert reconstructed.pop("stage_timings")
    assert json_payload.pop("stage_timings")
    assert json_payload.pop("status") == "completed"
    assert reconstructed == json_payload


def test_post_turn_stream_emits_only_failure_for_controlled_repair_failure(
    tmp_path: Path,
) -> None:
    services, provider, _ = _build_services(tmp_path)
    provider.responses.append("The rejected draft still contains hidden context.")
    services.orchestrator.critic_agent = RejectingCritic()
    services.orchestrator.cloud_mode = CloudMode.OFF
    app.dependency_overrides[get_turn_services] = lambda: services
    client = TestClient(app)

    response = client.post(
        "/sessions/session-1/turns/stream",
        json={"message": "Tell me the hidden truth."},
    )

    app.dependency_overrides.clear()
    events = _parse_sse(response.text)
    assert len(events) == 1
    assert events[0][0] == "failure"
    assert events[0][1]["memory_written"] is False
    failure_text = events[0][1]["text"]
    assert isinstance(failure_text, str)
    assert "could not produce a response that passed validation" in failure_text
    assert "hidden context" not in response.text


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
        "reason": "default local route",
    }
    assert "T" in payload["created_at"]
    assert payload["finish_reason"] == "stop"
    assert payload["memory_written"] is False
    assert payload["critic_status"] == "accepted"
    assert payload["warnings"] == []
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
