"""Guards against the two composition roots drifting apart (#48).

The CLI (`app.cli._build_services`) and the API (`app.composition.build_services`)
each assemble a `TurnOrchestrator`. Both must derive its `TurnOrchestratorConfig`
from the same `Settings` via `build_orchestrator_config`, or `.env`-backed knobs
silently apply on one surface and not the other (the original bug: the CLI ran the
local critic/memory extractor at the 350-token dataclass default while the API used
the configured 640).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from app.cli import _build_services
from app.composition import build_orchestrator_config
from app.config import Settings


def test_cli_build_services_uses_canonical_orchestrator_config(tmp_path: Path) -> None:
    settings = Settings(database_path=str(tmp_path / "sessions.db"))
    with (
        patch("app.cli._build_local_provider", return_value=MagicMock()),
        patch("app.cli._build_cloud_provider", return_value=MagicMock()),
        patch("app.cli._build_critic_agent", return_value=MagicMock()),
        patch("app.cli._build_memory_curator", return_value=MagicMock()),
        patch("app.cli._build_file_loader", return_value=MagicMock()),
    ):
        services = _build_services(settings, enable_retrieval=False, content_root=tmp_path)

    expected = build_orchestrator_config(settings, content_root=str(tmp_path))
    assert services.orchestrator.config == expected


def test_cli_orchestrator_config_honors_local_structured_max_tokens(tmp_path: Path) -> None:
    # The field the drift regressed: a non-default value must reach the CLI orchestrator.
    settings = Settings(
        database_path=str(tmp_path / "sessions.db"),
        local_structured_max_tokens=777,
    )
    with (
        patch("app.cli._build_local_provider", return_value=MagicMock()),
        patch("app.cli._build_cloud_provider", return_value=MagicMock()),
        patch("app.cli._build_critic_agent", return_value=MagicMock()),
        patch("app.cli._build_memory_curator", return_value=MagicMock()),
        patch("app.cli._build_file_loader", return_value=MagicMock()),
    ):
        services = _build_services(settings, enable_retrieval=False, content_root=tmp_path)

    assert services.orchestrator.config.local_structured_max_tokens == 777
