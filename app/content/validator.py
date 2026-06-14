from __future__ import annotations

import json
import re
from enum import Enum
from pathlib import Path
from typing import TypeVar, overload

from pydantic import BaseModel, Field, ValidationError

from app.domain import PersonaCard, SceneState, Visibility
from app.persistence import DemoWorldRecord


class ContentValidationStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class ContentIssue(BaseModel):
    code: str
    file: str
    message: str
    suggested_fix: str | None = None


class ContentValidationReport(BaseModel):
    status: ContentValidationStatus
    errors: list[ContentIssue] = Field(default_factory=list)
    warnings: list[ContentIssue] = Field(default_factory=list)
    checked_files: list[str] = Field(default_factory=list)


class LoreDocumentMetadata(BaseModel):
    path: str
    visibility: Visibility
    source_type: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    world_id: str | None = None
    scene_id: str | None = None
    persona_id: str | None = None


class LoreManifest(BaseModel):
    documents: list[LoreDocumentMetadata] = Field(default_factory=list)


ContentModel = DemoWorldRecord | PersonaCard | SceneState
TContentModel = TypeVar("TContentModel", DemoWorldRecord, PersonaCard, SceneState)


def validate_content(
    *,
    content_root: Path = Path("data"),
    world_id: str | None = None,
) -> ContentValidationReport:
    report = ContentValidationReport(status=ContentValidationStatus.PASS)
    worlds_dir = content_root / "worlds"
    personas_dir = content_root / "personas"
    scenes_dir = content_root / "scenes"
    documents_dir = content_root / "documents"

    for required_dir in (worlds_dir, personas_dir, scenes_dir):
        if not required_dir.exists() or not required_dir.is_dir():
            report.errors.append(
                _issue(
                    code="required_directory_missing",
                    file=_display_path(required_dir),
                    message=f"Required directory is missing: {required_dir.name}",
                    suggested_fix=f"Create {required_dir.name}/ under {content_root}.",
                )
            )

    worlds = _load_models(
        directory=worlds_dir,
        model_type=DemoWorldRecord,
        entity_name="world",
        report=report,
    )
    personas = _load_models(
        directory=personas_dir,
        model_type=PersonaCard,
        entity_name="persona",
        report=report,
    )
    scenes = _load_models(
        directory=scenes_dir,
        model_type=SceneState,
        entity_name="scene",
        report=report,
    )

    selected_worlds = worlds
    if world_id is not None:
        if world_id not in worlds:
            report.errors.append(
                _issue(
                    code="world_not_found",
                    file=_display_path(worlds_dir / f"{world_id}.json"),
                    message=f"World '{world_id}' was not found in this content root.",
                    suggested_fix="Use an existing world id or remove --world-id.",
                )
            )
            selected_worlds = {}
        else:
            selected_worlds = {world_id: worlds[world_id]}

    _validate_world_references(
        worlds=selected_worlds,
        personas=personas,
        scenes=scenes,
        report=report,
        content_root=content_root,
    )
    _validate_scene_persona_references(
        worlds=selected_worlds,
        personas=personas,
        scenes=scenes,
        report=report,
        content_root=content_root,
    )
    _validate_secrecy_rules(
        personas=personas,
        scenes=scenes,
        report=report,
        content_root=content_root,
    )
    _validate_documents(
        documents_dir=documents_dir,
        worlds=worlds,
        personas=personas,
        scenes=scenes,
        report=report,
    )

    report.status = _status_for(report)
    report.checked_files = sorted(set(report.checked_files))
    return report


@overload
def _load_models(
    *,
    directory: Path,
    model_type: type[DemoWorldRecord],
    entity_name: str,
    report: ContentValidationReport,
) -> dict[str, DemoWorldRecord]: ...


@overload
def _load_models(
    *,
    directory: Path,
    model_type: type[PersonaCard],
    entity_name: str,
    report: ContentValidationReport,
) -> dict[str, PersonaCard]: ...


@overload
def _load_models(
    *,
    directory: Path,
    model_type: type[SceneState],
    entity_name: str,
    report: ContentValidationReport,
) -> dict[str, SceneState]: ...


def _load_models(
    *,
    directory: Path,
    model_type: type[TContentModel],
    entity_name: str,
    report: ContentValidationReport,
) -> dict[str, TContentModel]:
    loaded: dict[str, TContentModel] = {}
    if not directory.exists() or not directory.is_dir():
        return loaded
    for path in sorted(directory.glob("*.json")):
        report.checked_files.append(_display_path(path))
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            report.errors.append(
                _issue(
                    code=f"{entity_name}_json_invalid",
                    file=_display_path(path),
                    message=f"Invalid JSON for {entity_name}: {exc}",
                    suggested_fix="Fix the JSON syntax and rerun validation.",
                )
            )
            continue
        try:
            model = model_type.model_validate(payload)
        except ValidationError as exc:
            report.errors.append(
                _issue(
                    code=f"{entity_name}_model_invalid",
                    file=_display_path(path),
                    message=(
                        f"{entity_name.capitalize()} failed model validation: "
                        f"{exc.errors()[0]['msg']}"
                    ),
                    suggested_fix="Align the JSON payload with the existing domain model.",
                )
            )
            continue

        model_id = str(model.id)
        if model_id in loaded:
            report.errors.append(
                _issue(
                    code=f"{entity_name}_duplicate_id",
                    file=_display_path(path),
                    message=f"Duplicate {entity_name} id '{model_id}' detected.",
                    suggested_fix=f"Rename one of the {entity_name} ids to keep them unique.",
                )
            )
            continue
        loaded[model_id] = model
    return loaded


def _validate_world_references(
    *,
    worlds: dict[str, DemoWorldRecord],
    personas: dict[str, PersonaCard],
    scenes: dict[str, SceneState],
    report: ContentValidationReport,
    content_root: Path,
) -> None:
    for world in worlds.values():
        world_file = content_root / "worlds" / f"{world.id}.json"
        if world.default_scene_id not in world.scene_ids:
            report.errors.append(
                _issue(
                    code="world_default_scene_not_listed",
                    file=_display_path(world_file),
                    message=(
                        f"World '{world.id}' default_scene_id '{world.default_scene_id}' "
                        "is not listed in scene_ids."
                    ),
                    suggested_fix=(
                        "Add the default scene id to scene_ids or change default_scene_id."
                    ),
                )
            )
        for scene_id in world.scene_ids:
            if scene_id not in scenes:
                report.errors.append(
                    _issue(
                        code="world_scene_missing",
                        file=_display_path(world_file),
                        message=f"World '{world.id}' references missing scene '{scene_id}'.",
                        suggested_fix=(
                            f"Add scenes/{scene_id.replace('-', '_')}.json "
                            "or remove the reference."
                        ),
                    )
                )
        for persona_id in world.persona_ids:
            if persona_id not in personas:
                report.errors.append(
                    _issue(
                        code="world_persona_missing",
                        file=_display_path(world_file),
                        message=f"World '{world.id}' references missing persona '{persona_id}'.",
                        suggested_fix=f"Add personas/{persona_id}.json or remove the reference.",
                    )
                )


def _validate_scene_persona_references(
    *,
    worlds: dict[str, DemoWorldRecord],
    personas: dict[str, PersonaCard],
    scenes: dict[str, SceneState],
    report: ContentValidationReport,
    content_root: Path,
) -> None:
    scene_to_world_ids: dict[str, list[str]] = {}
    for world in worlds.values():
        for scene_id in world.scene_ids:
            scene_to_world_ids.setdefault(scene_id, []).append(world.id)

    for scene in scenes.values():
        scene_file = content_root / "scenes" / f"{scene.id.replace('-', '_')}.json"
        for persona_id in scene.active_personas:
            if persona_id not in personas:
                report.errors.append(
                    _issue(
                        code="scene_active_persona_missing",
                        file=_display_path(scene_file),
                        message=(
                            f"Scene '{scene.id}' references missing active persona "
                            f"'{persona_id}'."
                        ),
                        suggested_fix=(
                            f"Add personas/{persona_id}.json or remove the active persona "
                            "reference."
                        ),
                    )
                )
                continue
            for world_id in scene_to_world_ids.get(scene.id, []):
                owner_world = worlds.get(world_id)
                if owner_world is not None and persona_id not in owner_world.persona_ids:
                    report.errors.append(
                        _issue(
                            code="scene_persona_not_listed_by_world",
                            file=_display_path(scene_file),
                            message=(
                                f"Scene '{scene.id}' uses active persona '{persona_id}', "
                                f"but world '{world_id}' does not list it."
                            ),
                            suggested_fix=(
                                "Add the persona id to the world or remove it from "
                                "active_personas."
                            ),
                        )
                    )


def _validate_secrecy_rules(
    *,
    personas: dict[str, PersonaCard],
    scenes: dict[str, SceneState],
    report: ContentValidationReport,
    content_root: Path,
) -> None:
    for persona in personas.values():
        persona_file = content_root / "personas" / f"{persona.id}.json"
        public_description = _normalize_text(persona.public_description)
        for secret in persona.secrets:
            normalized_secret = _normalize_text(secret)
            if normalized_secret and normalized_secret in public_description:
                report.warnings.append(
                    _issue(
                        code="persona_secret_in_public_description",
                        file=_display_path(persona_file),
                        message=(
                            f"Persona '{persona.id}' public_description contains "
                            "a listed secret."
                        ),
                        suggested_fix=(
                            "Move the secret into a private field or rewrite the "
                            "public description."
                        ),
                    )
                )
        for forbidden_knowledge in persona.forbidden_knowledge:
            normalized_knowledge = _normalize_text(forbidden_knowledge)
            if normalized_knowledge and normalized_knowledge in public_description:
                report.warnings.append(
                    _issue(
                        code="persona_forbidden_knowledge_in_public_description",
                        file=_display_path(persona_file),
                        message=(
                            f"Persona '{persona.id}' public_description contains "
                            "forbidden_knowledge."
                        ),
                        suggested_fix="Remove hidden knowledge from the public description.",
                    )
                )
        # goals and values are injected verbatim into the actor prompt
        # (context_builder), so a hidden fact here would reach the player.
        goals_and_values = _normalize_text(" ".join([*persona.goals, *persona.values]))
        for secret in persona.secrets:
            normalized_secret = _normalize_text(secret)
            if normalized_secret and normalized_secret in goals_and_values:
                report.warnings.append(
                    _issue(
                        code="persona_secret_in_goals_or_values",
                        file=_display_path(persona_file),
                        message=(
                            f"Persona '{persona.id}' goals or values contain a listed secret."
                        ),
                        suggested_fix=(
                            "Goals and values feed the actor prompt; move the secret to a "
                            "private field."
                        ),
                    )
                )
        for forbidden_knowledge in persona.forbidden_knowledge:
            normalized_knowledge = _normalize_text(forbidden_knowledge)
            if normalized_knowledge and normalized_knowledge in goals_and_values:
                report.warnings.append(
                    _issue(
                        code="persona_forbidden_knowledge_in_goals_or_values",
                        file=_display_path(persona_file),
                        message=(
                            f"Persona '{persona.id}' goals or values contain forbidden_knowledge."
                        ),
                        suggested_fix=(
                            "Goals and values are player-visible context; remove hidden knowledge."
                        ),
                    )
                )

        if persona.forbidden_knowledge and not persona.private_description and not persona.secrets:
            report.warnings.append(
                _issue(
                    code="persona_forbidden_knowledge_without_private_context",
                    file=_display_path(persona_file),
                    message=(
                        f"Persona '{persona.id}' defines forbidden_knowledge without secrets or "
                        "private_description."
                    ),
                    suggested_fix="Confirm that forbidden_knowledge belongs on this persona.",
                )
            )

    for scene in scenes.values():
        if not scene.gm_private_summary:
            continue
        scene_file = content_root / "scenes" / f"{scene.id.replace('-', '_')}.json"
        public_summary = _normalize_text(scene.player_visible_summary)
        gm_summary = _normalize_text(scene.gm_private_summary)
        if gm_summary and (gm_summary in public_summary or public_summary in gm_summary):
            report.warnings.append(
                _issue(
                    code="gm_private_summary_in_player_visible_summary",
                    file=_display_path(scene_file),
                    message=f"Scene '{scene.id}' public and GM summaries overlap too closely.",
                    suggested_fix=(
                        "Rewrite the player summary so it does not expose GM-only "
                        "information."
                    ),
                )
            )


def _validate_documents(
    *,
    documents_dir: Path,
    worlds: dict[str, DemoWorldRecord],
    personas: dict[str, PersonaCard],
    scenes: dict[str, SceneState],
    report: ContentValidationReport,
) -> None:
    if not documents_dir.exists() or not documents_dir.is_dir():
        return

    document_paths = sorted(
        path for path in documents_dir.iterdir() if path.suffix.lower() in {".md", ".txt"}
    )
    for path in document_paths:
        report.checked_files.append(_display_path(path))

    manifest_path = documents_dir / "manifest.json"
    if not manifest_path.exists():
        if document_paths:
            report.warnings.append(
                _issue(
                    code="lore_manifest_missing",
                    file=_display_path(documents_dir),
                    message="Lore documents exist, but documents/manifest.json is missing.",
                    suggested_fix="Add documents/manifest.json to make lore metadata explicit.",
                )
            )
        return

    report.checked_files.append(_display_path(manifest_path))
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        report.errors.append(
            _issue(
                code="documents_manifest_invalid_json",
                file=_display_path(manifest_path),
                message=f"Manifest JSON is invalid: {exc}",
                suggested_fix="Fix documents/manifest.json syntax.",
            )
        )
        return

    try:
        manifest = LoreManifest.model_validate(payload)
    except ValidationError as exc:
        report.errors.append(
            _issue(
                code="documents_manifest_invalid",
                file=_display_path(manifest_path),
                message=f"Manifest validation failed: {exc.errors()[0]['msg']}",
                suggested_fix=(
                    "Use explicit valid visibility and metadata fields in "
                    "documents/manifest.json."
                ),
            )
        )
        return

    listed_paths: set[str] = set()
    for document in manifest.documents:
        listed_paths.add(document.path)
        document_path = (documents_dir / document.path).resolve()
        try:
            relative_path = document_path.relative_to(documents_dir.resolve())
        except ValueError:
            report.errors.append(
                _issue(
                    code="documents_manifest_path_outside_root",
                    file=_display_path(manifest_path),
                    message=f"Manifest path '{document.path}' escapes the documents directory.",
                    suggested_fix="Use a relative path that stays inside documents/.",
                )
            )
            continue
        if relative_path.parts and relative_path.parts[0] == "..":
            report.errors.append(
                _issue(
                    code="documents_manifest_path_outside_root",
                    file=_display_path(manifest_path),
                    message=f"Manifest path '{document.path}' escapes the documents directory.",
                    suggested_fix="Use a relative path that stays inside documents/.",
                )
            )
            continue
        if document_path.suffix.lower() not in {".md", ".txt"}:
            report.errors.append(
                _issue(
                    code="documents_manifest_unsupported_suffix",
                    file=_display_path(manifest_path),
                    message=f"Manifest path '{document.path}' must point to a .md or .txt file.",
                    suggested_fix="Use a markdown or text document path.",
                )
            )
        if not document_path.exists():
            report.errors.append(
                _issue(
                    code="documents_manifest_missing_file",
                    file=_display_path(manifest_path),
                    message=f"Manifest references missing document '{document.path}'.",
                    suggested_fix="Create the document or remove it from the manifest.",
                )
            )
        if document.world_id is not None and document.world_id not in worlds:
            report.errors.append(
                _issue(
                    code="documents_manifest_world_missing",
                    file=_display_path(manifest_path),
                    message=(
                        f"Manifest document '{document.path}' references missing world "
                        f"'{document.world_id}'."
                    ),
                    suggested_fix="Use an existing world_id or remove the world scope.",
                )
            )
        if document.scene_id is not None and document.scene_id not in scenes:
            report.errors.append(
                _issue(
                    code="documents_manifest_scene_missing",
                    file=_display_path(manifest_path),
                    message=(
                        f"Manifest document '{document.path}' references missing scene "
                        f"'{document.scene_id}'."
                    ),
                    suggested_fix="Use an existing scene_id or remove the scene scope.",
                )
            )
        if document.persona_id is not None and document.persona_id not in personas:
            report.errors.append(
                _issue(
                    code="documents_manifest_persona_missing",
                    file=_display_path(manifest_path),
                    message=(
                        f"Manifest document '{document.path}' references missing persona "
                        f"'{document.persona_id}'."
                    ),
                    suggested_fix="Use an existing persona_id or remove the persona scope.",
                )
            )

    for path in document_paths:
        if path.name not in listed_paths:
            report.warnings.append(
                _issue(
                    code="lore_document_not_listed_in_manifest",
                    file=_display_path(path),
                    message=(
                        f"Lore document '{path.name}' is not listed in "
                        "documents/manifest.json."
                    ),
                    suggested_fix="Add the document to the manifest or remove the unused file.",
                )
            )


def _issue(*, code: str, file: str, message: str, suggested_fix: str | None = None) -> ContentIssue:
    return ContentIssue(code=code, file=file, message=message, suggested_fix=suggested_fix)


def _status_for(report: ContentValidationReport) -> ContentValidationStatus:
    if report.errors:
        return ContentValidationStatus.FAIL
    if report.warnings:
        return ContentValidationStatus.WARN
    return ContentValidationStatus.PASS


def _normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip().lower()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)
