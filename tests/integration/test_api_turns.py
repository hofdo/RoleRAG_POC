from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.api.routes import get_turn_services
from app.composition import AppServices
from app.domain import (
    CriticResult,
    PersonaCard,
    RetrievedChunk,
    SceneState,
    SessionState,
    Visibility,
)
from app.llm.provider import LlmMessage, LlmProvider, LlmRequest, LlmResponse
from app.main import app
from app.memory import RecentDialogueStore
from app.orchestration.turn_orchestrator import TurnOrchestrator
from app.persistence import DemoWorldRecord, SQLiteSessionRepository, SQLiteTurnRepository
from app.persistence.sqlite import connect_sqlite, initialize_database


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
            orchestrator=orchestrator,
            recent_dialogue_store=RecentDialogueStore(
                turn_repository=turn_repository,
                recent_turns=8,
            ),
        ),
        provider,
        retriever,
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
    assert payload == {
        "text": "Only archivists and locksmiths speak of that door.",
        "route": {
            "provider": "local",
            "model": "local-model",
            "reason": "default local route",
        },
        "memory_written": False,
        "warnings": [],
    }
    assert len(provider.requests) == 1
    assert len(retriever.calls) == 1
    prompt = provider.requests[0].messages[0].content
    assert "The west door has stayed locked for years." in prompt
    assert "spy waits behind the mirrored column" not in prompt
    assert "cipher key" not in prompt
    assert "The regent ordered the poisoning." not in prompt
    assert "route_max_tokens" not in response.text


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
    assert response.json()["detail"] == "Unknown session id: missing-session"


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
