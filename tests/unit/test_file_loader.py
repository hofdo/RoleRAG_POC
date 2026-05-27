from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.persistence.file_loader import (
    DataFileNotFoundError,
    DataValidationError,
    FileDataLoader,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_file_loader_loads_persona_scene_and_world_metadata(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "worlds" / "demo_world.json",
        {
            "id": "demo_world",
            "name": "Winter Palace Intrigue",
            "default_scene_id": "rose-gallery",
            "persona_ids": ["archivist"],
            "scene_ids": ["rose-gallery"],
        },
    )
    _write_json(
        tmp_path / "personas" / "archivist.json",
        {
            "id": "archivist",
            "name": "Iria Vale",
            "role": "npc",
            "public_description": "A composed palace archivist.",
            "speaking_style": "Precise and dry.",
        },
    )
    _write_json(
        tmp_path / "scenes" / "rose_gallery.json",
        {
            "id": "rose-gallery",
            "title": "Rose Gallery",
            "location": "Winter Palace",
            "player_visible_summary": "Courtiers drift between mirrors and roses.",
        },
    )
    loader = FileDataLoader(base_path=tmp_path)

    world = loader.load_world("demo_world")
    persona = loader.load_persona("archivist")
    scene = loader.load_scene("rose-gallery")

    assert world.id == "demo_world"
    assert world.default_scene_id == "rose-gallery"
    assert persona.name == "Iria Vale"
    assert scene.title == "Rose Gallery"


def test_file_loader_raises_clear_error_for_missing_persona(tmp_path: Path) -> None:
    loader = FileDataLoader(base_path=tmp_path)

    with pytest.raises(DataFileNotFoundError) as exc_info:
        loader.load_persona("missing-persona")

    assert "missing-persona" in str(exc_info.value)
    assert "persona" in str(exc_info.value)


def test_file_loader_raises_validation_error_for_invalid_scene(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "scenes" / "rose_gallery.json",
        {
            "id": "rose-gallery",
            "title": "Rose Gallery",
            "location": "Winter Palace",
        },
    )
    loader = FileDataLoader(base_path=tmp_path)

    with pytest.raises(DataValidationError) as exc_info:
        loader.load_scene("rose-gallery")

    assert "rose-gallery" in str(exc_info.value)
    assert "scene" in str(exc_info.value)
