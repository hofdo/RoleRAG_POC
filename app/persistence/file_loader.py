from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.domain import PersonaCard, SceneState

ModelT = TypeVar("ModelT", bound=BaseModel)


class DemoWorldRecord(BaseModel):
    id: str
    name: str
    default_scene_id: str
    persona_ids: list[str]
    scene_ids: list[str]


class ContentCatalogRecord(BaseModel):
    worlds: list[DemoWorldRecord]
    scenes: list[SceneState]
    personas: list[PersonaCard]


class ContentCatalogError(ValueError):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class DataFileNotFoundError(FileNotFoundError):
    def __init__(self, *, entity_type: str, entity_id: str, path: Path) -> None:
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.path = path
        super().__init__(f"Missing {entity_type} '{entity_id}' at {path}")


class DataValidationError(ValueError):
    def __init__(self, *, entity_type: str, entity_id: str, path: Path, cause: Exception) -> None:
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.path = path
        self.cause = cause
        super().__init__(f"Invalid {entity_type} '{entity_id}' at {path}: {cause}")


def _validate_content_id(entity_type: str, entity_id: str) -> None:
    """Reject ids that could escape the data directory or read arbitrary files.

    Allows ASCII alphanumerics plus '-' and '_' only; rejects path separators,
    '..', absolute paths, empty/whitespace ids, and control characters.
    """

    def _reject(reason: str) -> None:
        raise DataValidationError(
            entity_type=entity_type,
            entity_id=entity_id,
            path=Path(entity_id),
            cause=ValueError(reason),
        )

    if not entity_id.strip():
        _reject("identifier is empty")
    if "/" in entity_id or "\\" in entity_id:
        _reject("identifier contains a path separator")
    if ".." in entity_id:
        _reject("identifier contains a parent-directory reference")
    if Path(entity_id).is_absolute():
        _reject("identifier is an absolute path")
    if any(ord(char) < 32 or ord(char) == 127 for char in entity_id):
        _reject("identifier contains control characters")
    if not all(char.isascii() and (char.isalnum() or char in "-_") for char in entity_id):
        _reject("identifier contains disallowed characters")


class FileDataLoader:
    def __init__(self, *, base_path: Path | str = "data") -> None:
        self.base_path = Path(base_path)

    def load_world(self, world_id: str) -> DemoWorldRecord:
        _validate_content_id("world", world_id)
        return self._load_model(
            entity_type="world",
            entity_id=world_id,
            path=self.base_path / "worlds" / f"{world_id}.json",
            model_type=DemoWorldRecord,
        )

    def load_persona(self, persona_id: str) -> PersonaCard:
        _validate_content_id("persona", persona_id)
        return self._load_model(
            entity_type="persona",
            entity_id=persona_id,
            path=self.base_path / "personas" / f"{persona_id}.json",
            model_type=PersonaCard,
        )

    def load_scene(self, scene_id: str) -> SceneState:
        _validate_content_id("scene", scene_id)
        filename = f"{scene_id.replace('-', '_')}.json"
        return self._load_model(
            entity_type="scene",
            entity_id=scene_id,
            path=self.base_path / "scenes" / filename,
            model_type=SceneState,
        )

    def load_catalog(self) -> ContentCatalogRecord:
        for directory in ("worlds", "scenes", "personas"):
            if not (self.base_path / directory).is_dir():
                raise ContentCatalogError(
                    f"Configured content catalog is missing the {directory} directory."
                )
        return ContentCatalogRecord(
            worlds=sorted(
                self._load_catalog_directory(
                    entity_type="world",
                    directory="worlds",
                    model_type=DemoWorldRecord,
                ),
                key=lambda item: item.id,
            ),
            scenes=sorted(
                self._load_catalog_directory(
                    entity_type="scene",
                    directory="scenes",
                    model_type=SceneState,
                ),
                key=lambda item: item.id,
            ),
            personas=sorted(
                self._load_catalog_directory(
                    entity_type="persona",
                    directory="personas",
                    model_type=PersonaCard,
                ),
                key=lambda item: item.id,
            ),
        )

    def _load_catalog_directory(
        self,
        *,
        entity_type: str,
        directory: str,
        model_type: type[ModelT],
    ) -> list[ModelT]:
        records: list[ModelT] = []
        for path in sorted((self.base_path / directory).glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ContentCatalogError(
                    f"Configured content catalog contains an invalid {entity_type} file."
                ) from exc
            try:
                records.append(model_type.model_validate(payload))
            except ValidationError as exc:
                raise ContentCatalogError(
                    f"Configured content catalog contains an invalid {entity_type} file."
                ) from exc
        return records

    def _load_model(
        self,
        *,
        entity_type: str,
        entity_id: str,
        path: Path,
        model_type: type[ModelT],
    ) -> ModelT:
        if not path.exists():
            raise DataFileNotFoundError(entity_type=entity_type, entity_id=entity_id, path=path)

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise DataValidationError(
                entity_type=entity_type,
                entity_id=entity_id,
                path=path,
                cause=exc,
            ) from exc

        try:
            return model_type.model_validate(payload)
        except ValidationError as exc:
            raise DataValidationError(
                entity_type=entity_type,
                entity_id=entity_id,
                path=path,
                cause=exc,
            ) from exc
