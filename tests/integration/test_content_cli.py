from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from app.cli import app

runner = CliRunner()


def test_validate_content_cli_returns_zero_for_demo_warnings() -> None:
    result = runner.invoke(app, ["validate-content"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "warn"
    assert payload["errors"] == []


def test_validate_content_cli_returns_non_zero_on_errors(tmp_path: Path) -> None:
    (tmp_path / "worlds").mkdir()
    (tmp_path / "personas").mkdir()
    (tmp_path / "scenes").mkdir()
    (tmp_path / "worlds" / "broken.json").write_text(
        json.dumps(
            {
                "id": "broken",
                "name": "Broken",
                "default_scene_id": "missing-scene",
                "persona_ids": ["missing-persona"],
                "scene_ids": ["missing-scene"],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["validate-content", "--content-root", str(tmp_path)])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "fail"
    assert payload["errors"]


def test_create_scenario_template_creates_valid_files(tmp_path: Path) -> None:
    output = tmp_path / "iron-archduke"

    result = runner.invoke(
        app,
        [
            "create-scenario-template",
            "--name",
            "Iron Archduke",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert (output / "README.md").exists()
    assert list((output / "worlds").glob("*.json"))
    assert list((output / "scenes").glob("*.json"))
    assert list((output / "personas").glob("*.json"))
    assert (output / "documents" / "lore.md").exists()
    assert (output / "documents" / "manifest.json").exists()

    validate_result = runner.invoke(app, ["validate-content", "--content-root", str(output)])
    assert validate_result.exit_code == 0
    assert json.loads(validate_result.stdout)["status"] == "pass"


def test_create_scenario_template_does_not_overwrite_by_default(tmp_path: Path) -> None:
    output = tmp_path / "iron-archduke"
    output.mkdir()
    (output / "README.md").write_text("existing", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "create-scenario-template",
            "--name",
            "Iron Archduke",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 1
    assert "already exists" in result.stdout.lower()
