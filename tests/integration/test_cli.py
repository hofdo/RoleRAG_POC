from typer.testing import CliRunner

from app.cli import app

runner = CliRunner()


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
