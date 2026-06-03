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


def test_file_loader_loads_public_catalog_sorted_by_id(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "worlds" / "z_world.json",
        {
            "id": "z_world",
            "name": "Zed",
            "default_scene_id": "z-scene",
            "persona_ids": ["z-persona"],
            "scene_ids": ["z-scene"],
        },
    )
    _write_json(
        tmp_path / "worlds" / "a_world.json",
        {
            "id": "a_world",
            "name": "Alpha",
            "default_scene_id": "a-scene",
            "persona_ids": ["a-persona"],
            "scene_ids": ["a-scene"],
        },
    )
    _write_json(
        tmp_path / "personas" / "z-persona.json",
        {
            "id": "z-persona",
            "name": "Zed Persona",
            "role": "npc",
            "public_description": "Public.",
            "private_description": "Private.",
            "speaking_style": "Dry.",
            "secrets": ["Hidden."],
            "forbidden_knowledge": ["Forbidden."],
        },
    )
    _write_json(
        tmp_path / "personas" / "a-persona.json",
        {
            "id": "a-persona",
            "name": "Alpha Persona",
            "role": "narrator",
            "public_description": "Public.",
            "speaking_style": "Clear.",
        },
    )
    _write_json(
        tmp_path / "scenes" / "z_scene.json",
        {
            "id": "z-scene",
            "title": "Zed Scene",
            "location": "Zed",
            "player_visible_summary": "Visible.",
            "gm_private_summary": "Private.",
        },
    )
    _write_json(
        tmp_path / "scenes" / "a_scene.json",
        {
            "id": "a-scene",
            "title": "Alpha Scene",
            "location": "Alpha",
            "player_visible_summary": "Visible.",
        },
    )

    catalog = FileDataLoader(base_path=tmp_path).load_catalog()

    assert [world.id for world in catalog.worlds] == ["a_world", "z_world"]
    assert [scene.id for scene in catalog.scenes] == ["a-scene", "z-scene"]
    assert [persona.id for persona in catalog.personas] == ["a-persona", "z-persona"]


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
