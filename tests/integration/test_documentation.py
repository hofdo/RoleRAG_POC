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


def test_api_contract_document_is_linked_from_primary_docs() -> None:
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    implementation_guide = (REPOSITORY_ROOT / "docs" / "03_implementation_guide.md").read_text(
        encoding="utf-8"
    )

    assert "docs/12_api_contract.md" in readme
    assert "12_api_contract.md" in implementation_guide


def test_api_contract_document_covers_public_boundaries() -> None:
    contract = (REPOSITORY_ROOT / "docs" / "12_api_contract.md").read_text(encoding="utf-8")

    assert "GET /runtime/status" in contract
    assert "GET /content/catalog" in contract
    assert "POST /sessions" in contract
    assert "GET /sessions/{session_id}" in contract
    assert "POST /sessions/{session_id}/turns" in contract
    assert "POST /sessions/{session_id}/turns/stream" in contract
    assert "invalid_content_catalog" in contract
    assert "invalid_session_request" in contract
    assert "invalid_turn_request" in contract
    assert "session_not_found" in contract
    assert "validation_error" in contract
    assert "Content Catalog" in contract
    assert "worlds" in contract
    assert "scenes" in contract
    assert "personas" in contract
    assert "process-level" in contract
    assert "CONTENT_ROOT" in contract
    assert "per-request" in contract
    assert "frontend scenario-pack selection" in contract
    assert "warnings" in contract
    assert "text/event-stream" in contract
    assert "buffered" in contract
    assert "provider token streaming" in contract
    assert "concatenated" in contract
    assert "safe, shallow, non-diagnostic runtime metadata" in contract
    assert "does not call LLMs" in contract
    assert "probe Qdrant reachability" in contract
    assert "python -m app.cli doctor" in contract
    assert "python -m app.cli smoke-run --real-runtime" in contract
    for excluded in [
        "API keys",
        "provider URLs",
        "model secrets",
        "gm_private_summary",
        "private_description",
        "secrets",
        "forbidden_knowledge",
        "content_root",
        "file paths",
        "SQLite paths",
        "Qdrant URLs",
        "raw file paths",
        "hidden lore",
        "prompts",
        "raw prompts",
        "retrieved chunks",
        "GM/private fields",
        "hidden context",
        "internals",
        "SQLite internals",
        "Qdrant internals",
        "provider internals",
    ]:
        assert excluded in contract


def test_primary_docs_cover_local_play_ui_and_thin_client_limits() -> None:
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    architecture = (REPOSITORY_ROOT / "docs" / "02_architecture.md").read_text(encoding="utf-8")
    current_map = (REPOSITORY_ROOT / "docs" / "09_current_architecture_map.md").read_text(
        encoding="utf-8"
    )

    assert "uvicorn app.main:app --reload" in readme
    assert "http://127.0.0.1:8000/play" in readme
    assert "GET /content/catalog" in readme
    assert "catalog selectors" in readme
    assert "Create session" in readme
    assert "manual" in readme
    assert "fallback" in readme
    assert "Resume session" in readme
    assert "session_id" in readme
    assert "JSON" in readme
    assert "buffered SSE" in readme
    assert "scenario packs" in readme
    assert "process-level backend choice through `CONTENT_ROOT`" in readme
    assert "CONTENT_ROOT" in readme
    assert "thin client" in readme
    assert "no authentication or multi-user isolation" in readme
    assert "no provider token streaming" in readme
    assert "no production deployment hardening" in readme
    assert "no browser-local authoritative state" in readme
    assert "no frontend scenario-pack selection" in readme
    assert "no per-request content-root selection" in readme
    assert "no backend ownership moved into browser code" in readme
    assert "app/web/" in architecture
    assert "GET /play" in current_map
