from __future__ import annotations

import json
from unittest.mock import patch

from typer.testing import CliRunner

from app import __version__
from app.cli import app

runner = CliRunner()


def test_cli_health_reports_redacted_settings_without_building_services() -> None:
    with patch("app.cli._build_services") as build_services:
        result = runner.invoke(
            app,
            ["health"],
            env={
                "LOCAL_LLM_API_KEY": "super-secret-local",
                "CLOUD_LLM_API_KEY": "super-secret-cloud",
            },
        )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["name"] == "rolerag-poc"
    assert payload["status"] == "ok"
    assert payload["version"] == __version__
    assert payload["settings"]["local_llm_api_key"] == "***"
    assert payload["settings"]["cloud_llm_api_key"] == "***"
    build_services.assert_not_called()
