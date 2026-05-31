from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.agents.critic_agent import CriticAgent
from app.api.routes import get_read_services
from app.composition import AppServices
from app.domain import PersonaCard, SceneState
from app.llm.provider import LlmProvider
from app.main import app
from app.memory import RecentDialogueStore
from app.orchestration.turn_orchestrator import TurnOrchestrator
from app.persistence import DemoWorldRecord, SQLiteSessionRepository, SQLiteTurnRepository
from app.persistence.sqlite import connect_sqlite, initialize_database


class UnusedProvider(LlmProvider):
    async def generate(self, request: Any) -> Any:
        raise AssertionError("provider should not be called in session API tests")


class FakeLoader:
    def load_world(self, world_id: str) -> DemoWorldRecord:
        if world_id != "demo_world":
            raise ValueError(f"Unknown world: {world_id}")
        return DemoWorldRecord(
            id="demo_world",
            name="Winter Palace Intrigue",
            default_scene_id="rose-gallery",
            persona_ids=["archivist"],
            scene_ids=["rose-gallery"],
        )

    def load_persona(self, persona_id: str) -> PersonaCard:
        if persona_id != "archivist":
            raise ValueError(f"Unknown persona: {persona_id}")
        return PersonaCard(
            id="archivist",
            name="Iria Vale",
            role="npc",
            public_description="A composed palace archivist.",
            private_description="She is quietly aiding the coup.",
            speaking_style="Precise and dry.",
            secrets=["The regent's spy keeps a second ledger."],
            forbidden_knowledge=["The rose seal hides the vault key sequence."],
        )

    def load_scene(self, scene_id: str) -> SceneState:
        if scene_id != "rose-gallery":
            raise ValueError(f"Unknown scene: {scene_id}")
        return SceneState(
            id="rose-gallery",
            title="Rose Gallery",
            location="Winter Palace",
            player_visible_summary="Courtiers drift between mirrors and roses.",
            gm_private_summary="A loyalist listener waits in the south alcove.",
        )


def _build_services(tmp_path: Path) -> AppServices:
    connection = connect_sqlite(tmp_path / "api-sessions.db")
    initialize_database(connection)
    turn_repository = SQLiteTurnRepository(connection)
    session_repository = SQLiteSessionRepository(connection)
    orchestrator = TurnOrchestrator(
        loader=FakeLoader(),
        provider=UnusedProvider(),
        critic_agent=CriticAgent(),
        session_repository=session_repository,
        turn_repository=turn_repository,
        recent_dialogue_store=RecentDialogueStore(
            turn_repository=turn_repository,
            recent_turns=8,
        ),
        local_model="local-model",
        cloud_model="cloud-model",
        local_max_tokens=700,
        cloud_max_tokens=1000,
        local_temperature=0.75,
        cloud_temperature=0.65,
        cloud_mode="ask",
    )
    return AppServices(
        connection=connection,
        orchestrator=orchestrator,
        recent_dialogue_store=RecentDialogueStore(
            turn_repository=turn_repository,
            recent_turns=8,
        ),
    )


def test_fastapi_app_imports_successfully() -> None:
    assert app.title == "rolerag-poc"


def test_post_sessions_creates_session_and_returns_safe_fields(tmp_path: Path) -> None:
    app.dependency_overrides[get_read_services] = lambda: _build_services(tmp_path)
    client = TestClient(app)

    response = client.post(
        "/sessions",
        json={
            "world_id": "demo_world",
            "scene_id": "rose-gallery",
            "player_name": "Player",
            "active_persona_id": "archivist",
        },
    )

    app.dependency_overrides.clear()
    assert response.status_code == 201
    payload = response.json()
    assert payload["world_id"] == "demo_world"
    assert payload["active_scene_id"] == "rose-gallery"
    assert payload["active_persona_id"] == "archivist"
    assert "player_name" not in payload
    assert "created_at" not in payload
    assert "updated_at" not in payload


def test_get_session_returns_recent_turns_without_private_state(tmp_path: Path) -> None:
    services = _build_services(tmp_path)
    session = services.orchestrator.create_session(
        world_id="demo_world",
        scene_id="rose-gallery",
        active_persona_id="archivist",
        player_name="Player",
        session_id="session-1",
    )
    services.orchestrator.turn_repository.append_turn(
        session_id=session.id,
        scene_id=session.active_scene_id,
        persona_id=session.active_persona_id,
        user_message="What do you know about the locked door?",
        assistant_message="Only that it has not opened in years.",
        route=services.orchestrator._build_local_route(reason="default local route"),
    )
    services.close()
    app.dependency_overrides[get_read_services] = lambda: _build_services(tmp_path)
    client = TestClient(app)

    response = client.get("/sessions/session-1")

    app.dependency_overrides.clear()
    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "session-1"
    assert payload["recent_turns"] == [
        {
            "turn_index": 1,
            "user_message": "What do you know about the locked door?",
            "assistant_message": "Only that it has not opened in years.",
            "created_at": payload["recent_turns"][0]["created_at"],
        }
    ]
    serialized = response.text
    assert "south alcove" not in serialized
    assert "second ledger" not in serialized
    assert "vault key sequence" not in serialized


def test_get_session_returns_404_for_missing_session(tmp_path: Path) -> None:
    app.dependency_overrides[get_read_services] = lambda: _build_services(tmp_path)
    client = TestClient(app)

    response = client.get("/sessions/missing-session")

    app.dependency_overrides.clear()
    assert response.status_code == 404
    assert response.json()["detail"] == "Unknown session id: missing-session"


def test_post_sessions_rejects_invalid_request_with_422(tmp_path: Path) -> None:
    app.dependency_overrides[get_read_services] = lambda: _build_services(tmp_path)
    client = TestClient(app)

    response = client.post(
        "/sessions",
        json={
            "world_id": "demo_world",
            "scene_id": "rose-gallery",
            "player_name": "",
            "active_persona_id": "archivist",
        },
    )

    app.dependency_overrides.clear()
    assert response.status_code == 422
