from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app import __version__
from app.agents.critic_agent import CriticAgent
from app.api.routes import get_read_services
from app.composition import AppServices
from app.config import Settings, get_settings
from app.domain import PersonaCard, SceneState, SessionState
from app.llm.provider import LlmProvider
from app.llm.router import CloudMode, ModelProviderName, ModelRoute
from app.main import app
from app.memory import RecentDialogueStore
from app.orchestration.turn_orchestrator import TurnOrchestrator, TurnOrchestratorConfig
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
            scene_ids=["rose-gallery", "east-wing"],
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
        if scene_id == "rose-gallery":
            return SceneState(
                id="rose-gallery",
                title="Rose Gallery",
                location="Winter Palace",
                player_visible_summary="Courtiers drift between mirrors and roses.",
                gm_private_summary="A loyalist listener waits in the south alcove.",
            )
        if scene_id == "east-wing":
            return SceneState(
                id="east-wing",
                title="East Wing",
                location="Winter Palace",
                player_visible_summary="A cold corridor lined with portraits.",
                gm_private_summary="The regent's ledger is hidden behind a portrait.",
            )
        raise ValueError(f"Unknown scene: {scene_id}")


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
        config=TurnOrchestratorConfig(
            local_model="local-model",
            cloud_model="cloud-model",
            local_max_tokens=700,
            cloud_max_tokens=1000,
            local_temperature=0.75,
            cloud_temperature=0.65,
        ),
    )
    return AppServices(
        connection=connection,
        session_repository=session_repository,
        orchestrator=orchestrator,
        recent_dialogue_store=RecentDialogueStore(
            turn_repository=turn_repository,
            recent_turns=8,
        ),
    )


def _write_scenario_pack(root: Path) -> None:
    (root / "worlds").mkdir(parents=True)
    (root / "personas").mkdir()
    (root / "scenes").mkdir()
    (root / "worlds" / "custom_world.json").write_text(
        """
        {
          "id": "custom_world",
          "name": "Custom World",
          "default_scene_id": "custom-opening",
          "persona_ids": ["custom-narrator"],
          "scene_ids": ["custom-opening"]
        }
        """,
        encoding="utf-8",
    )
    (root / "personas" / "custom-narrator.json").write_text(
        """
        {
          "id": "custom-narrator",
          "name": "Custom Narrator",
          "role": "narrator",
          "public_description": "A custom narrator.",
          "private_description": "Private narrator notes.",
          "speaking_style": "Clear.",
          "secrets": ["A hidden alliance."],
          "forbidden_knowledge": ["The ending."]
        }
        """,
        encoding="utf-8",
    )
    (root / "scenes" / "custom_opening.json").write_text(
        """
        {
          "id": "custom-opening",
          "title": "Custom Opening",
          "location": "Custom Hall",
          "player_visible_summary": "A custom hall waits.",
          "gm_private_summary": "A trap is armed."
        }
        """,
        encoding="utf-8",
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


def test_post_sessions_defaults_to_local_provider(tmp_path: Path) -> None:
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
    assert response.json()["provider"] == "local"


def test_create_cloud_session_rejected_when_cloud_mode_off(tmp_path: Path) -> None:
    database_path = tmp_path / "api-sessions.db"
    app.dependency_overrides[get_read_services] = lambda: _build_services(tmp_path)
    app.dependency_overrides[get_settings] = lambda: Settings(
        database_path=str(database_path),
        cloud_mode=CloudMode.OFF,
        cloud_llm_api_key="real-cloud-key",
    )
    client = TestClient(app)

    response = client.post(
        "/sessions",
        json={
            "world_id": "demo_world",
            "scene_id": "rose-gallery",
            "player_name": "Player",
            "active_persona_id": "archivist",
            "provider": "cloud",
        },
    )

    app.dependency_overrides.clear()
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "cloud_unavailable"


def test_create_cloud_session_rejected_when_no_usable_key(tmp_path: Path) -> None:
    database_path = tmp_path / "api-sessions.db"
    app.dependency_overrides[get_read_services] = lambda: _build_services(tmp_path)
    app.dependency_overrides[get_settings] = lambda: Settings(
        database_path=str(database_path),
        cloud_mode=CloudMode.AUTO,
        cloud_llm_api_key="replace_me",
    )
    client = TestClient(app)

    response = client.post(
        "/sessions",
        json={
            "world_id": "demo_world",
            "scene_id": "rose-gallery",
            "player_name": "Player",
            "active_persona_id": "archivist",
            "provider": "cloud",
        },
    )

    app.dependency_overrides.clear()
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "cloud_unavailable"


def test_create_cloud_session_succeeds_when_configured(tmp_path: Path) -> None:
    database_path = tmp_path / "api-sessions.db"
    app.dependency_overrides[get_read_services] = lambda: _build_services(tmp_path)
    app.dependency_overrides[get_settings] = lambda: Settings(
        database_path=str(database_path),
        cloud_mode=CloudMode.AUTO,
        cloud_llm_api_key="real-cloud-key",
    )
    client = TestClient(app)

    response = client.post(
        "/sessions",
        json={
            "world_id": "demo_world",
            "scene_id": "rose-gallery",
            "player_name": "Player",
            "active_persona_id": "archivist",
            "provider": "cloud",
        },
    )

    app.dependency_overrides.clear()
    assert response.status_code == 201
    assert response.json()["provider"] == "cloud"


def test_create_cloud_session_allowed_when_cloud_mode_ask(tmp_path: Path) -> None:
    # ask vs auto is a client-side confirmation concern (Task 5's SPA); the API
    # itself allows both once a usable key is configured.
    database_path = tmp_path / "api-sessions.db"
    app.dependency_overrides[get_read_services] = lambda: _build_services(tmp_path)
    app.dependency_overrides[get_settings] = lambda: Settings(
        database_path=str(database_path),
        cloud_mode=CloudMode.ASK,
        cloud_llm_api_key="real-cloud-key",
    )
    client = TestClient(app)

    response = client.post(
        "/sessions",
        json={
            "world_id": "demo_world",
            "scene_id": "rose-gallery",
            "player_name": "Player",
            "active_persona_id": "archivist",
            "provider": "cloud",
        },
    )

    app.dependency_overrides.clear()
    assert response.status_code == 201
    assert response.json()["provider"] == "cloud"


def test_post_sessions_rejects_invalid_provider_value(tmp_path: Path) -> None:
    app.dependency_overrides[get_read_services] = lambda: _build_services(tmp_path)
    client = TestClient(app)

    response = client.post(
        "/sessions",
        json={
            "world_id": "demo_world",
            "scene_id": "rose-gallery",
            "player_name": "Player",
            "active_persona_id": "archivist",
            "provider": "quantum",
        },
    )

    app.dependency_overrides.clear()
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_get_sessions_returns_recent_safe_metadata_only(tmp_path: Path) -> None:
    services = _build_services(tmp_path)
    created_at = datetime(2026, 6, 3, 10, 0, tzinfo=UTC)
    services.session_repository.create_session(
        SessionState(
            id="older-session",
            world_id="demo_world",
            active_scene_id="rose-gallery",
            active_persona_id="archivist",
            player_name="Avery",
            created_at=created_at,
            updated_at=created_at + timedelta(minutes=1),
        )
    )
    services.session_repository.create_session(
        SessionState(
            id="newer-session",
            world_id="demo_world",
            active_scene_id="rose-gallery",
            active_persona_id="archivist",
            player_name="Blair",
            content_root=str(tmp_path / "private-pack"),
            created_at=created_at + timedelta(minutes=2),
            updated_at=created_at + timedelta(minutes=3),
        )
    )
    services.close()
    app.dependency_overrides[get_read_services] = lambda: _build_services(tmp_path)
    client = TestClient(app)

    response = client.get("/sessions")

    app.dependency_overrides.clear()
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"sessions"}
    assert [session["session_id"] for session in payload["sessions"]] == [
        "newer-session",
        "older-session",
    ]
    assert set(payload["sessions"][0]) == {
        "session_id",
        "world_id",
        "active_scene_id",
        "active_persona_id",
        "player_name",
        "created_at",
        "updated_at",
    }
    serialized = response.text.lower()
    for forbidden in [
        "recent_turns",
        "messages",
        "prompt",
        "content_root",
        str(tmp_path).lower(),
        "retrieved",
        "memory",
        "sqlite",
        "qdrant",
        "provider",
        "private-pack",
        "hidden",
    ]:
        assert forbidden not in serialized


def test_get_sessions_orders_by_updated_at_then_created_at_and_limits_to_10(
    tmp_path: Path,
) -> None:
    services = _build_services(tmp_path)
    base = datetime(2026, 6, 3, 10, 0, tzinfo=UTC)
    for index in range(12):
        services.session_repository.create_session(
            SessionState(
                id=f"session-{index:02d}",
                world_id="demo_world",
                active_scene_id="rose-gallery",
                active_persona_id="archivist",
                player_name=f"Player {index}",
                created_at=base + timedelta(minutes=index),
                updated_at=base + timedelta(hours=index),
            )
        )
    services.session_repository.create_session(
        SessionState(
            id="tie-breaker",
            world_id="demo_world",
            active_scene_id="rose-gallery",
            active_persona_id="archivist",
            player_name="Tie",
            created_at=base + timedelta(days=1),
            updated_at=base + timedelta(hours=11),
        )
    )
    services.close()
    app.dependency_overrides[get_read_services] = lambda: _build_services(tmp_path)
    client = TestClient(app)

    response = client.get("/sessions")

    app.dependency_overrides.clear()
    assert response.status_code == 200
    session_ids = [session["session_id"] for session in response.json()["sessions"]]
    assert len(session_ids) == 10
    assert session_ids[:3] == ["tie-breaker", "session-11", "session-10"]
    assert "session-00" not in session_ids


def test_get_content_catalog_returns_default_public_catalog() -> None:
    client = TestClient(app)

    response = client.get("/content/catalog")

    assert response.status_code == 200
    payload = response.json()
    assert payload["worlds"] == [
        {
            "id": "demo_world",
            "name": "Winter Palace Intrigue",
            "default_scene_id": "rose-gallery",
            "scene_ids": ["rose-gallery"],
            "persona_ids": ["archivist"],
        }
    ]
    assert payload["scenes"] == [
        {
            "id": "rose-gallery",
            "title": "Rose Gallery",
            "location": "Winter Palace",
            "player_visible_summary": "Courtiers drift between mirrors and roses while musicians "
            "play softly.",
            "active_personas": ["archivist"],
        }
    ]
    assert payload["personas"] == [
        {
            "id": "archivist",
            "name": "Iria Vale",
            "role": "npc",
            "public_description": "A composed palace archivist with an excellent memory.",
            "speaking_style": "Precise, dry, and difficult to rattle.",
        }
    ]
    forbidden_terms = [
        "gm_private_summary",
        "private_description",
        "secrets",
        "forbidden_knowledge",
        "content_root",
        "data/",
        "lore",
        "prompt",
        "retrieval",
        "sqlite",
        "qdrant",
        "provider",
    ]
    response_text = response.text.lower()
    for term in forbidden_terms:
        assert term not in response_text


def test_get_content_catalog_uses_custom_process_content_root_without_path_leak(
    tmp_path: Path,
) -> None:
    pack_root = tmp_path / "private-pack"
    database_path = tmp_path / "api-custom-catalog.db"
    _write_scenario_pack(pack_root)
    app.dependency_overrides[get_settings] = lambda: Settings(
        database_path=str(database_path),
        content_root=pack_root,
    )
    client = TestClient(app)

    response = client.get("/content/catalog")

    app.dependency_overrides.clear()
    assert response.status_code == 200
    payload = response.json()
    assert payload["worlds"][0]["id"] == "custom_world"
    assert payload["scenes"][0]["id"] == "custom-opening"
    assert payload["personas"][0]["id"] == "custom-narrator"
    assert str(pack_root) not in response.text
    assert "Private narrator notes" not in response.text
    assert "A trap is armed" not in response.text


def test_get_content_catalog_returns_safe_structured_error_for_invalid_root(
    tmp_path: Path,
) -> None:
    pack_root = tmp_path / "missing-private-pack"
    app.dependency_overrides[get_settings] = lambda: Settings(content_root=pack_root)
    client = TestClient(app)

    response = client.get("/content/catalog")

    app.dependency_overrides.clear()
    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "invalid_content_catalog",
            "message": "Configured content catalog is missing the worlds directory.",
            "details": [],
        }
    }
    assert str(pack_root) not in response.text


def test_get_runtime_status_returns_exact_safe_shape() -> None:
    client = TestClient(app)

    response = client.get("/runtime/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "app_name": "rolerag-poc",
        "app_version": __version__,
        "environment": "local",
        "cloud_mode": "ask",
        "retrieval_configured": True,
        "content_catalog_available": True,
        "local_provider_configured": True,
        "cloud_provider_configured": False,
    }
    assert set(payload) == {
        "app_name",
        "app_version",
        "environment",
        "cloud_mode",
        "retrieval_configured",
        "content_catalog_available",
        "local_provider_configured",
        "cloud_provider_configured",
    }


def test_get_runtime_status_booleans_come_from_settings_and_catalog(
    tmp_path: Path,
) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(
        app_env="test",
        content_root=tmp_path / "missing-content",
        local_llm_base_url=" ",
        local_llm_api_key="local-key",
        local_llm_model="local-model",
        cloud_mode=CloudMode.AUTO,
        cloud_llm_base_url="https://cloud.example/v1",
        cloud_llm_api_key="real-cloud-key",
        cloud_llm_model="cloud-model",
        qdrant_url=" ",
        embedding_model="embedding-model",
    )
    client = TestClient(app)

    response = client.get("/runtime/status")

    app.dependency_overrides.clear()
    assert response.status_code == 200
    payload = response.json()
    assert payload["environment"] == "test"
    assert payload["cloud_mode"] == "auto"
    assert payload["retrieval_configured"] is False
    assert payload["content_catalog_available"] is False
    assert payload["local_provider_configured"] is False
    assert payload["cloud_provider_configured"] is True


def test_get_runtime_status_does_not_expose_diagnostic_or_private_values(
    tmp_path: Path,
) -> None:
    private_root = tmp_path / "private-pack"
    app.dependency_overrides[get_settings] = lambda: Settings(
        app_env="prod",
        database_path=str(tmp_path / "private.db"),
        content_root=private_root,
        local_llm_base_url="http://local.example/v1",
        local_llm_api_key="super-secret-local",
        local_llm_model="secret-local-model",
        cloud_llm_base_url="https://cloud.example/v1",
        cloud_llm_api_key="super-secret-cloud",
        cloud_llm_model="secret-cloud-model",
        qdrant_url="http://qdrant.example:6333",
        embedding_model="secret-embedding-model",
    )
    client = TestClient(app)

    response = client.get("/runtime/status")

    app.dependency_overrides.clear()
    assert response.status_code == 200
    serialized = response.text.lower()
    for forbidden in [
        "super-secret-local",
        "super-secret-cloud",
        "http://local.example",
        "https://cloud.example",
        "secret-local-model",
        "secret-cloud-model",
        "secret-embedding-model",
        str(private_root).lower(),
        "private.db",
        "content_root",
        "sqlite",
        "qdrant.example",
        "prompt",
        "retrieved_chunks",
        "gm_private_summary",
        "private_description",
        "hidden_context",
    ]:
        assert forbidden not in serialized


def test_post_sessions_uses_process_content_root_without_request_path(
    tmp_path: Path,
) -> None:
    pack_root = tmp_path / "pack"
    database_path = tmp_path / "api-custom.db"
    _write_scenario_pack(pack_root)
    app.dependency_overrides[get_settings] = lambda: Settings(
        database_path=str(database_path),
        content_root=pack_root,
    )
    client = TestClient(app)

    response = client.post(
        "/sessions",
        json={
            "world_id": "custom_world",
            "scene_id": "custom-opening",
            "player_name": "Player",
            "active_persona_id": "custom-narrator",
        },
    )

    app.dependency_overrides.clear()
    assert response.status_code == 201
    payload = response.json()
    assert payload["world_id"] == "custom_world"
    assert "content_root" not in payload

    connection = connect_sqlite(database_path)
    initialize_database(connection)
    rows = connection.execute("SELECT content_root FROM sessions").fetchall()
    connection.close()
    assert [row["content_root"] for row in rows] == [str(pack_root)]


def test_post_sessions_uses_default_process_content_root(tmp_path: Path) -> None:
    database_path = tmp_path / "api-default.db"
    app.dependency_overrides[get_settings] = lambda: Settings(database_path=str(database_path))
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

    connection = connect_sqlite(database_path)
    initialize_database(connection)
    rows = connection.execute("SELECT content_root FROM sessions").fetchall()
    connection.close()
    assert [row["content_root"] for row in rows] == ["data"]


def test_post_sessions_rejects_per_request_content_root(tmp_path: Path) -> None:
    app.dependency_overrides[get_read_services] = lambda: _build_services(tmp_path)
    client = TestClient(app)

    response = client.post(
        "/sessions",
        json={
            "world_id": "demo_world",
            "scene_id": "rose-gallery",
            "player_name": "Player",
            "active_persona_id": "archivist",
            "content_root": "/tmp/other-pack",
        },
    )

    app.dependency_overrides.clear()
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_post_sessions_does_not_expose_process_content_root_on_missing_world(
    tmp_path: Path,
) -> None:
    pack_root = tmp_path / "private-pack"
    database_path = tmp_path / "api-private-pack.db"
    _write_scenario_pack(pack_root)
    app.dependency_overrides[get_settings] = lambda: Settings(
        database_path=str(database_path),
        content_root=pack_root,
    )
    client = TestClient(app)

    response = client.post(
        "/sessions",
        json={
            "world_id": "missing-world",
            "scene_id": "custom-opening",
            "player_name": "Player",
            "active_persona_id": "custom-narrator",
        },
    )

    app.dependency_overrides.clear()
    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "invalid_session_request",
            "message": "Unknown world id: missing-world",
            "details": [],
        }
    }
    assert str(pack_root) not in response.text


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
        route=ModelRoute(
            provider=ModelProviderName.LOCAL,
            model=services.orchestrator.config.local_model,
            max_tokens=services.orchestrator.config.local_max_tokens,
            temperature=services.orchestrator.config.local_temperature,
            reason="default local route",
        ),
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
    assert response.json() == {
        "error": {
            "code": "session_not_found",
            "message": "Unknown session id: missing-session",
            "details": [],
        }
    }


def test_post_sessions_rejects_invalid_world_with_400_envelope(tmp_path: Path) -> None:
    app.dependency_overrides[get_read_services] = lambda: _build_services(tmp_path)
    client = TestClient(app)

    response = client.post(
        "/sessions",
        json={
            "world_id": "missing-world",
            "scene_id": "rose-gallery",
            "player_name": "Player",
            "active_persona_id": "archivist",
        },
    )

    app.dependency_overrides.clear()
    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "invalid_session_request",
            "message": "Unknown world: missing-world",
            "details": [],
        }
    }


def test_post_sessions_rejects_invalid_scene_with_400_envelope(tmp_path: Path) -> None:
    app.dependency_overrides[get_read_services] = lambda: _build_services(tmp_path)
    client = TestClient(app)

    response = client.post(
        "/sessions",
        json={
            "world_id": "demo_world",
            "scene_id": "missing-scene",
            "player_name": "Player",
            "active_persona_id": "archivist",
        },
    )

    app.dependency_overrides.clear()
    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "invalid_session_request",
            "message": "Unknown scene for world demo_world: missing-scene",
            "details": [],
        }
    }


def test_post_sessions_rejects_invalid_persona_with_400_envelope(tmp_path: Path) -> None:
    app.dependency_overrides[get_read_services] = lambda: _build_services(tmp_path)
    client = TestClient(app)

    response = client.post(
        "/sessions",
        json={
            "world_id": "demo_world",
            "scene_id": "rose-gallery",
            "player_name": "Player",
            "active_persona_id": "missing-persona",
        },
    )

    app.dependency_overrides.clear()
    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "invalid_session_request",
            "message": "Unknown persona for world demo_world: missing-persona",
            "details": [],
        }
    }


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
    payload = response.json()
    assert payload["error"]["code"] == "validation_error"
    assert payload["error"]["message"] == "Request validation failed"
    assert payload["error"]["details"]


def test_post_sessions_rejects_player_name_exceeding_max_length(tmp_path: Path) -> None:
    app.dependency_overrides[get_read_services] = lambda: _build_services(tmp_path)
    client = TestClient(app)

    response = client.post(
        "/sessions",
        json={
            "world_id": "demo_world",
            "scene_id": "rose-gallery",
            "player_name": "x" * 201,
            "active_persona_id": "archivist",
        },
    )

    app.dependency_overrides.clear()
    assert response.status_code == 422
    payload = response.json()
    assert payload["error"]["code"] == "validation_error"
    assert payload["error"]["message"] == "Request validation failed"
    assert payload["error"]["details"]


def test_post_sessions_rejects_over_long_id_fields(tmp_path: Path) -> None:
    base = {
        "world_id": "demo_world",
        "scene_id": "rose-gallery",
        "player_name": "Avery",
        "active_persona_id": "archivist",
    }
    for field in ("world_id", "scene_id", "active_persona_id"):
        app.dependency_overrides[get_read_services] = lambda: _build_services(tmp_path)
        client = TestClient(app)
        response = client.post("/sessions", json={**base, field: "x" * 201})
        app.dependency_overrides.clear()
        assert response.status_code == 422, field
        assert response.json()["error"]["code"] == "validation_error"


def test_get_session_memories_returns_episode_metadata(tmp_path: Path) -> None:
    from app.api.routes import get_read_services
    from app.domain import MemoryCandidate, Visibility
    from app.persistence import SQLiteMemoryRepository

    services = _build_services(tmp_path)
    services.session_repository.create_session(
        SessionState(
            id="session-1",
            world_id="demo_world",
            active_scene_id="rose-gallery",
            active_persona_id="archivist",
            player_name="Avery",
        )
    )
    memory_repository = SQLiteMemoryRepository(services.connection)
    memory_repository.append_memories(
        session_id="session-1",
        memories=[
            MemoryCandidate(
                summary="The player promised to return before dawn.",
                visibility=Visibility.PLAYER,
                importance=4,
                tags=["promise"],
                scene_id="rose-gallery",
                actor_id="archivist",
            )
        ],
    )
    services.memory_repository = memory_repository
    app.dependency_overrides[get_read_services] = lambda: services
    client = TestClient(app)

    response = client.get("/sessions/session-1/memories")

    app.dependency_overrides.clear()
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["memories"]) == 1
    memory = payload["memories"][0]
    assert memory["summary"] == "The player promised to return before dawn."
    assert memory["visibility"] == "player"
    assert memory["importance"] == 4
    assert memory["tags"] == ["promise"]


def test_get_session_memories_returns_all_visibilities(tmp_path: Path) -> None:
    # The memory viewer is an author/admin surface: by design it returns GM and
    # CHARACTER_PRIVATE memories too. Actor-prompt leakage is blocked elsewhere
    # (retrieval/prompt filtering), so this endpoint deliberately does not filter.
    from app.api.routes import get_read_services
    from app.domain import MemoryCandidate, Visibility
    from app.persistence import SQLiteMemoryRepository

    services = _build_services(tmp_path)
    services.session_repository.create_session(
        SessionState(
            id="session-1",
            world_id="demo_world",
            active_scene_id="rose-gallery",
            active_persona_id="archivist",
            player_name="Avery",
        )
    )
    memory_repository = SQLiteMemoryRepository(services.connection)
    memory_repository.append_memories(
        session_id="session-1",
        memories=[
            MemoryCandidate(
                summary="Public commitment.",
                visibility=Visibility.PLAYER,
                importance=4,
                tags=["promise"],
                scene_id="rose-gallery",
                actor_id="archivist",
            ),
            MemoryCandidate(
                summary="GM-only plot note.",
                visibility=Visibility.GM,
                importance=3,
                tags=["gm"],
                scene_id="rose-gallery",
                actor_id="archivist",
            ),
            MemoryCandidate(
                summary="Character-private fear.",
                visibility=Visibility.CHARACTER_PRIVATE,
                importance=2,
                tags=["private"],
                scene_id="rose-gallery",
                actor_id="archivist",
            ),
        ],
    )
    services.memory_repository = memory_repository
    app.dependency_overrides[get_read_services] = lambda: services
    client = TestClient(app)

    response = client.get("/sessions/session-1/memories")

    app.dependency_overrides.clear()
    assert response.status_code == 200
    returned = {m["visibility"] for m in response.json()["memories"]}
    assert returned == {"player", "gm", "character_private"}


def test_get_session_memories_returns_404_for_unknown_session(tmp_path: Path) -> None:
    from app.api.routes import get_read_services

    services = _build_services(tmp_path)
    app.dependency_overrides[get_read_services] = lambda: services
    client = TestClient(app)

    response = client.get("/sessions/missing/memories")

    app.dependency_overrides.clear()
    assert response.status_code == 404


def test_update_session_scene(tmp_path: Path) -> None:
    from app.api.routes import get_read_services

    services = _build_services(tmp_path)
    services.session_repository.create_session(
        SessionState(
            id="session-1",
            world_id="demo_world",
            active_scene_id="rose-gallery",
            active_persona_id="archivist",
            player_name="Avery",
        )
    )
    app.dependency_overrides[get_read_services] = lambda: services
    client = TestClient(app)

    response = client.post("/sessions/session-1/scene", json={"scene_id": "east-wing"})
    invalid = client.post("/sessions/session-1/scene", json={"scene_id": "nope"})

    app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["active_scene_id"] == "east-wing"
    assert invalid.status_code == 400
    reloaded = services.session_repository.get_session("session-1")
    assert reloaded is not None
    assert reloaded.active_scene_id == "east-wing"


def test_update_session_scene_returns_404_for_missing_session(tmp_path: Path) -> None:
    from app.api.routes import get_read_services

    services = _build_services(tmp_path)
    app.dependency_overrides[get_read_services] = lambda: services
    client = TestClient(app)

    response = client.post("/sessions/missing-session/scene", json={"scene_id": "east-wing"})

    app.dependency_overrides.clear()
    assert response.status_code == 404


def test_session_canon_add_list_and_delete(tmp_path: Path) -> None:
    from app.api.routes import get_read_services
    from app.persistence import SQLiteCanonRepository

    services = _build_services(tmp_path)
    services.session_repository.create_session(
        SessionState(
            id="session-1",
            world_id="demo_world",
            active_scene_id="rose-gallery",
            active_persona_id="archivist",
            player_name="Avery",
        )
    )
    services.canon_repository = SQLiteCanonRepository(services.connection)
    app.dependency_overrides[get_read_services] = lambda: services
    client = TestClient(app)

    added = client.post("/sessions/session-1/canon", json={"text": "The east gate stays shut."})
    listed = client.get("/sessions/session-1/canon")
    fact_id = added.json()["id"]
    deleted = client.delete(f"/sessions/session-1/canon/{fact_id}")
    after = client.get("/sessions/session-1/canon")
    missing_session = client.post("/sessions/missing/canon", json={"text": "x"})

    app.dependency_overrides.clear()
    assert added.status_code == 201
    assert listed.status_code == 200
    assert [fact["text"] for fact in listed.json()["facts"]] == ["The east gate stays shut."]
    assert deleted.status_code == 204
    assert after.json()["facts"] == []
    assert missing_session.status_code == 404
