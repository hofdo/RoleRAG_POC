from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from app.cli import app
from app.domain import PersonaCard, SceneState
from app.llm.provider import LlmProvider, LlmRequest, LlmResponse

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


def test_cli_turn_runs_with_mocked_provider() -> None:
    with (
        patch("app.cli._build_local_provider", return_value=FakeProvider()),
        patch("app.cli._build_file_loader", return_value=FakeLoader()),
    ):
        result = runner.invoke(
            app,
            [
                "turn",
                "--message",
                "What have you heard about the regent?",
                "--world-id",
                "demo_world",
                "--scene-id",
                "rose-gallery",
            ],
        )

    assert result.exit_code == 0
    assert "I have heard enough to know the regent fears open daylight." in result.stdout


def test_cli_turn_fails_clearly_for_missing_scene() -> None:
    with (
        patch("app.cli._build_local_provider", return_value=FakeProvider()),
        patch("app.cli._build_file_loader", return_value=FakeLoader()),
    ):
        result = runner.invoke(
            app,
            [
                "turn",
                "--message",
                "What have you heard about the regent?",
                "--scene-id",
                "missing-scene",
            ],
        )

    assert result.exit_code == 1
    assert "missing-scene" in result.stdout
