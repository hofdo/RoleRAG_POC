from __future__ import annotations

import json
from unittest.mock import patch

from typer.testing import CliRunner

from app import __version__
from app.cli import app
from app.diagnostics.runtime_checks import (
    DiagnosticCheck,
    DiagnosticStatus,
    RuntimeDiagnosticsReport,
)
from app.diagnostics.smoke_runner import RetrievalSelectionSummary, SmokeRunSummary

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


def test_cli_doctor_prints_structured_report() -> None:
    with patch(
        "app.cli.run_doctor",
        return_value=RuntimeDiagnosticsReport(
            status=DiagnosticStatus.WARN,
            checks=[
                DiagnosticCheck(
                    name="qdrant",
                    status=DiagnosticStatus.SKIPPED,
                    message="Qdrant check skipped. Use --check-qdrant to verify.",
                )
            ],
        ),
    ) as run_doctor:
        result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "warn"
    assert payload["checks"][0]["name"] == "qdrant"
    run_doctor.assert_called_once()


def test_cli_smoke_run_prints_structured_summary() -> None:
    with patch(
        "app.cli.run_smoke",
        return_value=SmokeRunSummary(
            status=DiagnosticStatus.PASS,
            mode="safe",
            session_id="smoke-session",
            temp_database_path="/tmp/rolerag-smoke.db",
            ingested_chunk_count=1,
            persisted_turns=1,
            memory_written=True,
            persisted_memories=1,
            indexed_memories=1,
            retrieval_selected_count=1,
            actor_response="Archive answer.",
            warnings=[],
            steps=[
                DiagnosticCheck(
                    name="turn",
                    status=DiagnosticStatus.PASS,
                    message="Turn completed.",
                )
            ],
            retrieval_selected=[
                RetrievalSelectionSummary(
                    id="memory-1",
                    source="memory_episode:memory-1",
                    source_type="session_memory",
                    collection="session_memory",
                    visibility="player",
                    selected_rank=1,
                    original_score=0.61,
                    adjusted_score=0.8,
                    tags=["archive"],
                )
            ],
        ),
    ) as run_smoke:
        result = runner.invoke(app, ["smoke-run"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "pass"
    assert payload["persisted_turns"] == 1
    assert payload["retrieval_selected"][0]["source"] == "memory_episode:memory-1"
    assert "text" not in result.stdout
    run_smoke.assert_called_once()
