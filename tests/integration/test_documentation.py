from __future__ import annotations

import re
from pathlib import Path

from app.config import Settings

REPOSITORY_ROOT = Path(__file__).parents[2]
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
LOCAL_PATH_MARKERS = ("/Users/", "/home/", "file://", "vscode://")


def _markdown_files() -> list[Path]:
    return [REPOSITORY_ROOT / "README.md", *sorted((REPOSITORY_ROOT / "docs").glob("**/*.md"))]


def test_markdown_links_do_not_use_local_machine_paths() -> None:
    local_links = [
        f"{path.relative_to(REPOSITORY_ROOT)}: {target}"
        for path in _markdown_files()
        for target in MARKDOWN_LINK.findall(path.read_text(encoding="utf-8"))
        if any(marker in target for marker in LOCAL_PATH_MARKERS)
    ]

    assert local_links == []


def test_repository_relative_markdown_links_resolve() -> None:
    broken_links: list[str] = []
    for path in _markdown_files():
        for target in MARKDOWN_LINK.findall(path.read_text(encoding="utf-8")):
            if target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            relative_target = target.split("#", maxsplit=1)[0]
            if not (path.parent / relative_target).resolve().exists():
                broken_links.append(f"{path.relative_to(REPOSITORY_ROOT)}: {target}")

    assert broken_links == []


def test_env_example_matches_settings_fields() -> None:
    env_keys = {
        line.split("=", maxsplit=1)[0].lower()
        for line in (REPOSITORY_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#") and "=" in line
    }

    assert env_keys == set(Settings.model_fields)
