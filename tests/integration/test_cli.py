from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from app.cli import app
from app.domain import PersonaCard, SceneState, Visibility
from app.llm.provider import LlmProvider, LlmRequest, LlmResponse
from app.rag.models import RagChunk, RagCollection

runner = CliRunner()


class FakeProvider(LlmProvider):
    async def generate(self, request: LlmRequest) -> LlmResponse:
        return LlmResponse(
            text="I have heard enough to know the regent fears open daylight.",
            provider="fake",
            model=request.model,
            usage={"total_tokens": 15},
            finish_reason="stop",
        )


class FakeLoader:
    def load_world(self, world_id: str) -> object:
        if world_id != "demo_world":
            raise ValueError(f"Unknown world: {world_id}")
        return type(
            "WorldRecord",
            (),
            {
                "id": world_id,
                "default_scene_id": "rose-gallery",
                "persona_ids": ["archivist"],
                "scene_ids": ["rose-gallery"],
            },
        )()

    def load_persona(self, persona_id: str) -> PersonaCard:
        if persona_id != "archivist":
            raise ValueError(f"Unknown persona: {persona_id}")
        return PersonaCard(
            id="archivist",
            name="Iria Vale",
            role="npc",
            public_description="A composed palace archivist.",
            speaking_style="Precise and dry.",
        )

    def load_scene(self, scene_id: str) -> SceneState:
        if scene_id != "rose-gallery":
            raise ValueError(f"Unknown scene: {scene_id}")
        return SceneState(
            id="rose-gallery",
            title="Rose Gallery",
            location="Winter Palace",
            player_visible_summary="Courtiers drift between mirrors and roses.",
        )


class FakeEmbeddingProvider:
    dimension = 2

    def embed_text(self, text: str) -> list[float]:
        return [1.0, float(len(text))]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, float(len(text))] for text in texts]


class RecordingVectorStore:
    def __init__(self) -> None:
        self.ensure_calls: list[tuple[RagCollection, int]] = []
        self.replace_calls: list[tuple[RagCollection, str, list[RagChunk], list[list[float]]]] = []

    def ensure_collection(self, collection: RagCollection, vector_size: int) -> None:
        self.ensure_calls.append((collection, vector_size))

    def replace_source(
        self,
        collection: RagCollection,
        source: str,
        chunks: list[RagChunk],
        vectors: list[list[float]],
    ) -> None:
        self.replace_calls.append((collection, source, chunks, vectors))


def test_cli_help_exits_successfully() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Usage" in result.stdout


def test_cli_config_redacts_api_keys() -> None:
    result = runner.invoke(
        app,
        ["config"],
        env={
            "LOCAL_LLM_API_KEY": "super-secret-local",
            "CLOUD_LLM_API_KEY": "super-secret-cloud",
        },
    )

    assert result.exit_code == 0
    assert "super-secret-local" not in result.stdout
    assert "super-secret-cloud" not in result.stdout
    assert '"local_llm_api_key": "***"' in result.stdout
    assert '"cloud_llm_api_key": "***"' in result.stdout
    assert "cloud_llm_enabled" not in result.stdout


def test_cli_route_shows_local_route_by_default() -> None:
    result = runner.invoke(app, ["route", "--task", "actor_response"])

    assert result.exit_code == 0
    assert '"provider": "local"' in result.stdout


def test_cli_route_requires_confirmation_in_ask_mode() -> None:
    result = runner.invoke(
        app,
        ["route", "--task", "repair", "--failed-local-attempts", "2"],
        env={"CLOUD_MODE": "ask"},
    )

    assert result.exit_code == 0
    assert '"provider": "cloud"' in result.stdout
    assert '"requires_user_confirmation": true' in result.stdout


def test_cli_route_forces_local_in_off_mode() -> None:
    result = runner.invoke(
        app,
        [
            "route",
            "--task",
            "repair",
            "--failed-local-attempts",
            "2",
        ],
        env={"CLOUD_MODE": "off"},
    )

    assert result.exit_code == 0
    assert '"provider": "local"' in result.stdout
    assert '"reason": "cloud mode is off"' in result.stdout


def test_cli_start_session_and_turn_run_with_mocked_provider(tmp_path: Path) -> None:
    with (
        patch("app.cli._build_local_provider", return_value=FakeProvider()),
        patch("app.cli._build_file_loader", return_value=FakeLoader()),
    ):
        start_result = runner.invoke(
            app,
            [
                "start-session",
                "--session-id",
                "demo-session",
                "--player-name",
                "Avery",
            ],
            env={"DATABASE_PATH": str(tmp_path / "sessions.db")},
        )
        turn_result = runner.invoke(
            app,
            [
                "turn",
                "--message",
                "What have you heard about the regent?",
                "--session-id",
                "demo-session",
            ],
            env={"DATABASE_PATH": str(tmp_path / "sessions.db")},
        )

    assert start_result.exit_code == 0
    assert json.loads(start_result.stdout)["id"] == "demo-session"
    assert turn_result.exit_code == 0
    assert "I have heard enough to know the regent fears open daylight." in turn_result.stdout


def test_cli_resume_prints_session_metadata(tmp_path: Path) -> None:
    with (
        patch("app.cli._build_local_provider", return_value=FakeProvider()),
        patch("app.cli._build_file_loader", return_value=FakeLoader()),
    ):
        runner.invoke(
            app,
            [
                "start-session",
                "--session-id",
                "demo-session",
                "--player-name",
                "Avery",
            ],
            env={"DATABASE_PATH": str(tmp_path / "sessions.db")},
        )
        result = runner.invoke(
            app,
            ["resume", "--session-id", "demo-session"],
            env={"DATABASE_PATH": str(tmp_path / "sessions.db")},
        )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["id"] == "demo-session"
    assert payload["active_scene_id"] == "rose-gallery"


def test_cli_turn_fails_clearly_for_missing_session(tmp_path: Path) -> None:
    with patch("app.cli._build_local_provider", return_value=FakeProvider()):
        result = runner.invoke(
            app,
            [
                "turn",
                "--message",
                "What have you heard about the regent?",
                "--session-id",
                "missing-session",
            ],
            env={"DATABASE_PATH": str(tmp_path / "sessions.db")},
        )

    assert result.exit_code == 1
    assert "missing-session" in result.stdout


def test_cli_ingest_uses_fake_embedding_provider_and_vector_store(tmp_path: Path) -> None:
    document = tmp_path / "demo_lore.md"
    document.write_text(
        "# Rose Gallery\n\nCourtiers drift between mirrors and roses.",
        encoding="utf-8",
    )
    vector_store = RecordingVectorStore()

    with (
        patch("app.cli._build_embedding_provider", return_value=FakeEmbeddingProvider()),
        patch("app.cli._build_vector_store", return_value=vector_store),
    ):
        result = runner.invoke(
            app,
            [
                "ingest",
                str(document),
                "--visibility",
                Visibility.PLAYER.value,
                "--source-type",
                "lore",
                "--world-id",
                "demo_world",
                "--tag",
                "palace",
            ],
        )

    assert result.exit_code == 0
    assert '"collection": "canon_lore"' in result.stdout
    assert '"chunk_count": 1' in result.stdout
    assert vector_store.ensure_calls == [(RagCollection.CANON_LORE, 2)]
    assert len(vector_store.replace_calls) == 1
    _, source, chunks, vectors = vector_store.replace_calls[0]
    assert source == str(document)
    assert len(chunks) == 1
    assert len(vectors) == 1
    assert chunks[0].visibility == Visibility.PLAYER
    assert chunks[0].tags == ["palace"]
    assert chunks[0].world_id == "demo_world"
