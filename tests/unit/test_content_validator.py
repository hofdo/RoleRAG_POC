from __future__ import annotations

import json
from pathlib import Path

from app.content.validator import ContentValidationStatus, validate_content


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_valid_content_root(root: Path) -> Path:
    _write_json(
        root / "worlds" / "court-intrigue.json",
        {
            "id": "court-intrigue",
            "name": "Court Intrigue",
            "default_scene_id": "opening-hall",
            "persona_ids": ["court-narrator"],
            "scene_ids": ["opening-hall"],
        },
    )
    _write_json(
        root / "personas" / "court-narrator.json",
        {
            "id": "court-narrator",
            "name": "Court Narrator",
            "role": "narrator",
            "public_description": "A poised voice that frames the court's tensions.",
            "speaking_style": "Measured and observant.",
        },
    )
    _write_json(
        root / "scenes" / "opening_hall.json",
        {
            "id": "opening-hall",
            "title": "Opening Hall",
            "location": "North Wing",
            "active_personas": ["court-narrator"],
            "player_visible_summary": "Guests trade careful glances beneath painted ceilings.",
            "gm_private_summary": "The usher is secretly tracking every sealed note.",
        },
    )
    documents_dir = root / "documents"
    documents_dir.mkdir(parents=True, exist_ok=True)
    (documents_dir / "lore.md").write_text(
        "# Hall Notes\n\nGuests gather before the procession begins.\n",
        encoding="utf-8",
    )
    _write_json(
        documents_dir / "manifest.json",
        {
            "documents": [
                {
                    "path": "lore.md",
                    "visibility": "player",
                    "source_type": "lore",
                    "tags": ["court"],
                    "world_id": "court-intrigue",
                }
            ]
        },
    )
    return root


def test_demo_content_validates_without_errors() -> None:
    report = validate_content(content_root=Path("data"))

    assert report.errors == []
    assert report.warnings == []
    assert report.status == ContentValidationStatus.PASS


def test_missing_persona_reference_is_reported_as_error(tmp_path: Path) -> None:
    root = _build_valid_content_root(tmp_path)
    _write_json(
        root / "worlds" / "court-intrigue.json",
        {
            "id": "court-intrigue",
            "name": "Court Intrigue",
            "default_scene_id": "opening-hall",
            "persona_ids": ["missing-persona"],
            "scene_ids": ["opening-hall"],
        },
    )

    report = validate_content(content_root=root)

    assert report.status == ContentValidationStatus.FAIL
    assert any(issue.code == "world_persona_missing" for issue in report.errors)


def test_missing_scene_reference_is_reported_as_error(tmp_path: Path) -> None:
    root = _build_valid_content_root(tmp_path)
    _write_json(
        root / "worlds" / "court-intrigue.json",
        {
            "id": "court-intrigue",
            "name": "Court Intrigue",
            "default_scene_id": "missing-scene",
            "persona_ids": ["court-narrator"],
            "scene_ids": ["missing-scene"],
        },
    )

    report = validate_content(content_root=root)

    assert report.status == ContentValidationStatus.FAIL
    assert any(issue.code == "world_scene_missing" for issue in report.errors)


def test_invalid_manifest_visibility_is_reported_as_error(tmp_path: Path) -> None:
    root = _build_valid_content_root(tmp_path)
    _write_json(
        root / "documents" / "manifest.json",
        {
            "documents": [
                {
                    "path": "lore.md",
                    "visibility": "public",
                    "source_type": "lore",
                }
            ]
        },
    )

    report = validate_content(content_root=root)

    assert report.status == ContentValidationStatus.FAIL
    assert any(issue.code == "documents_manifest_invalid" for issue in report.errors)


def test_persona_secret_in_public_description_is_reported_as_warning(tmp_path: Path) -> None:
    root = _build_valid_content_root(tmp_path)
    _write_json(
        root / "personas" / "court-narrator.json",
        {
            "id": "court-narrator",
            "name": "Court Narrator",
            "role": "narrator",
            "public_description": "A poised voice who knows the usher hides every sealed note.",
            "speaking_style": "Measured and observant.",
            "secrets": ["The usher hides every sealed note."],
        },
    )

    report = validate_content(content_root=root)

    assert any(issue.code == "persona_secret_in_public_description" for issue in report.warnings)


def test_persona_secret_in_goals_is_reported_as_warning(tmp_path: Path) -> None:
    root = _build_valid_content_root(tmp_path)
    _write_json(
        root / "personas" / "court-narrator.json",
        {
            "id": "court-narrator",
            "name": "Court Narrator",
            "role": "narrator",
            "public_description": "A poised voice.",
            "speaking_style": "Measured and observant.",
            "secrets": ["The usher hides every sealed note."],
            "goals": ["The usher hides every sealed note."],
        },
    )

    report = validate_content(content_root=root)

    assert any(
        issue.code == "persona_secret_in_goals_or_values" for issue in report.warnings
    )


def test_persona_forbidden_knowledge_in_values_is_reported_as_warning(tmp_path: Path) -> None:
    root = _build_valid_content_root(tmp_path)
    _write_json(
        root / "personas" / "court-narrator.json",
        {
            "id": "court-narrator",
            "name": "Court Narrator",
            "role": "narrator",
            "public_description": "A poised voice.",
            "speaking_style": "Measured and observant.",
            "private_description": "Keeps the regent's confidence.",
            "forbidden_knowledge": ["The regent poisoned the late duke."],
            "values": ["The regent poisoned the late duke."],
        },
    )

    report = validate_content(content_root=root)

    assert any(
        issue.code == "persona_forbidden_knowledge_in_goals_or_values"
        for issue in report.warnings
    )


def test_gm_private_summary_duplication_is_reported_as_warning(tmp_path: Path) -> None:
    root = _build_valid_content_root(tmp_path)
    _write_json(
        root / "scenes" / "opening_hall.json",
        {
            "id": "opening-hall",
            "title": "Opening Hall",
            "location": "North Wing",
            "active_personas": ["court-narrator"],
            "player_visible_summary": "The usher is secretly tracking every sealed note.",
            "gm_private_summary": "The usher is secretly tracking every sealed note.",
        },
    )

    report = validate_content(content_root=root)

    assert any(
        issue.code == "gm_private_summary_in_player_visible_summary"
        for issue in report.warnings
    )
