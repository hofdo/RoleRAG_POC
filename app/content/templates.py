from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, Field


class ScenarioTemplateResult(BaseModel):
    name: str
    slug: str
    output_root: str
    created_files: list[str] = Field(default_factory=list)


def create_scenario_template(
    *,
    output: Path,
    name: str | None = None,
    overwrite: bool = False,
) -> ScenarioTemplateResult:
    slug = _slugify(name or output.name)
    if not slug:
        raise ValueError("scenario template name must produce a non-empty slug")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output path already exists: {output}")

    scenario_name = name or _titleize_slug(slug)
    world_id = slug
    scene_id = f"{slug}-opening"
    persona_id = f"{slug}-narrator"

    files = {
        output / "README.md": _build_readme(scenario_name=scenario_name, slug=slug),
        output / "worlds" / f"{world_id}.json": json.dumps(
            {
                "id": world_id,
                "name": scenario_name,
                "default_scene_id": scene_id,
                "persona_ids": [persona_id],
                "scene_ids": [scene_id],
            },
            indent=2,
        )
        + "\n",
        output / "scenes" / f"{scene_id.replace('-', '_')}.json": json.dumps(
            {
                "id": scene_id,
                "title": f"{scenario_name} Opening",
                "location": "Main Stage",
                "active_personas": [persona_id],
                "player_visible_summary": (
                    "Tension hangs in the room as the first decision approaches."
                ),
                "gm_private_summary": "A private agenda is already in motion behind the welcome.",
            },
            indent=2,
        )
        + "\n",
        output / "personas" / f"{persona_id}.json": json.dumps(
            {
                "id": persona_id,
                "name": f"{scenario_name} Narrator",
                "role": "narrator",
                "public_description": "A steady narrative voice that frames the opening situation.",
                "speaking_style": "Clear, observant, and grounded in the immediate scene.",
            },
            indent=2,
        )
        + "\n",
        output / "documents" / "lore.md": (
            f"# {scenario_name}\n\n"
            "Add player-visible lore for this scenario here.\n"
        ),
        output / "documents" / "manifest.json": json.dumps(
            {
                "documents": [
                    {
                        "path": "lore.md",
                        "visibility": "player",
                        "source_type": "lore",
                        "tags": [slug],
                        "world_id": world_id,
                    }
                ]
            },
            indent=2,
        )
        + "\n",
    }

    created_files: list[str] = []
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        created_files.append(_display_path(path))

    return ScenarioTemplateResult(
        name=scenario_name,
        slug=slug,
        output_root=_display_path(output),
        created_files=created_files,
    )


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    return normalized.strip("-")


def _titleize_slug(slug: str) -> str:
    return " ".join(part.capitalize() for part in slug.split("-"))


def _build_readme(*, scenario_name: str, slug: str) -> str:
    return (
        f"# {scenario_name}\n\n"
        "This standalone scenario pack contains authored worlds, scenes, personas, and "
        "optional lore.\n\n"
        "Validate it with:\n\n"
        f"```bash\npython -m app.cli validate-content --content-root data/scenarios/{slug}\n```\n\n"
        "To make it part of the active runtime data tree in the current MVP, copy the "
        "validated files into `data/worlds`, `data/scenes`, `data/personas`, and "
        "`data/documents`, then ingest any lore you want available to retrieval.\n"
    )


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)
