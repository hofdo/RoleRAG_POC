from __future__ import annotations

from enum import Enum
from pathlib import Path

import httpx
from pydantic import BaseModel, Field

from app.config import Settings, is_usable_cloud_api_key
from app.llm.router import CloudMode
from app.persistence import FileDataLoader, connect_sqlite, initialize_database
from app.rag.vector_store import QdrantVectorStore


class DiagnosticStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIPPED = "skipped"


class DiagnosticCheck(BaseModel):
    name: str
    status: DiagnosticStatus
    message: str
    hint: str | None = None


class RuntimeDiagnosticsReport(BaseModel):
    status: DiagnosticStatus
    checks: list[DiagnosticCheck] = Field(default_factory=list)


def build_runtime_diagnostics(
    settings: Settings,
    *,
    check_qdrant: bool = False,
    check_local_provider: bool = False,
) -> RuntimeDiagnosticsReport:
    checks = [
        _check_settings(settings),
        _check_sqlite(settings),
        _check_demo_data(),
        _check_cloud_config(settings),
        _check_qdrant(settings) if check_qdrant else _skipped_qdrant_check(),
        _check_local_provider(settings) if check_local_provider else _skipped_provider_check(),
    ]
    return RuntimeDiagnosticsReport(status=_overall_status(checks), checks=checks)


def is_usable_cloud_key(value: str) -> bool:
    return is_usable_cloud_api_key(value)


def _check_settings(settings: Settings) -> DiagnosticCheck:
    return DiagnosticCheck(
        name="settings",
        status=DiagnosticStatus.PASS,
        message=f"Settings loaded successfully for APP_ENV={settings.app_env}.",
    )


def _check_sqlite(settings: Settings) -> DiagnosticCheck:
    database_path = Path(settings.database_path)
    temp_path = database_path.parent / ".doctor_runtime_check.db"
    connection = None
    try:
        connection = connect_sqlite(temp_path)
        initialize_database(connection)
        tables = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    except Exception as exc:
        return DiagnosticCheck(
            name="sqlite",
            status=DiagnosticStatus.FAIL,
            message=f"SQLite initialization failed: {exc}",
            hint="Check DATABASE_PATH permissions and parent directory accessibility.",
        )
    finally:
        if connection is not None:
            connection.close()
        if temp_path.exists():
            temp_path.unlink()

    required_tables = {"sessions", "turns", "memory_episodes"}
    if not required_tables.issubset(tables):
        return DiagnosticCheck(
            name="sqlite",
            status=DiagnosticStatus.FAIL,
            message="SQLite schema initialization did not create the required tables.",
            hint="Inspect app/persistence/sqlite.py and the local filesystem permissions.",
        )
    return DiagnosticCheck(
        name="sqlite",
        status=DiagnosticStatus.PASS,
        message="SQLite database initialized successfully in a temporary verification path.",
    )


def _check_demo_data() -> DiagnosticCheck:
    loader = FileDataLoader()
    try:
        loader.load_world("demo_world")
        loader.load_persona("archivist")
        loader.load_scene("rose-gallery")
    except Exception as exc:
        return DiagnosticCheck(
            name="demo_data",
            status=DiagnosticStatus.FAIL,
            message=f"Demo data failed to load: {exc}",
            hint="Verify the JSON files under data/worlds, data/personas, and data/scenes.",
        )
    return DiagnosticCheck(
        name="demo_data",
        status=DiagnosticStatus.PASS,
        message="Demo world, persona, and scene files loaded successfully.",
    )


def _check_cloud_config(settings: Settings) -> DiagnosticCheck:
    cloud_mode = CloudMode(settings.cloud_mode)
    usable_key = is_usable_cloud_key(settings.cloud_llm_api_key)
    placeholder_message = "Cloud API key is blank or still set to a placeholder."
    if cloud_mode == CloudMode.OFF:
        if usable_key:
            return DiagnosticCheck(
                name="cloud_config",
                status=DiagnosticStatus.WARN,
                message="Cloud mode is off, but a usable cloud API key is configured.",
                hint="Unset the cloud key if cloud access is not intended on this machine.",
            )
        return DiagnosticCheck(
            name="cloud_config",
            status=DiagnosticStatus.PASS,
            message="Cloud mode is off and no usable cloud API key is configured.",
        )
    if not settings.cloud_llm_base_url.strip() or not settings.cloud_llm_model.strip():
        return DiagnosticCheck(
            name="cloud_config",
            status=DiagnosticStatus.FAIL,
            message="Cloud mode is enabled, but cloud base URL or model is missing.",
            hint="Set CLOUD_LLM_BASE_URL and CLOUD_LLM_MODEL before using cloud routing.",
        )
    if not usable_key:
        status = DiagnosticStatus.FAIL if cloud_mode == CloudMode.AUTO else DiagnosticStatus.WARN
        hint = (
            "Set a real CLOUD_LLM_API_KEY or switch CLOUD_MODE=off if cloud usage is not intended."
        )
        return DiagnosticCheck(
            name="cloud_config",
            status=status,
            message=placeholder_message,
            hint=hint,
        )
    return DiagnosticCheck(
        name="cloud_config",
        status=DiagnosticStatus.PASS,
        message=f"Cloud configuration is present for CLOUD_MODE={cloud_mode.value}.",
    )


def _check_qdrant(settings: Settings) -> DiagnosticCheck:
    try:
        store = QdrantVectorStore(url=settings.qdrant_url)
        collections = store.client.get_collections()
        collection_count = len(getattr(collections, "collections", []))
    except Exception as exc:
        return DiagnosticCheck(
            name="qdrant",
            status=DiagnosticStatus.FAIL,
            message=f"Qdrant reachability check failed: {exc}",
            hint="Start Qdrant with `docker compose up -d qdrant` and verify QDRANT_URL.",
        )
    return DiagnosticCheck(
        name="qdrant",
        status=DiagnosticStatus.PASS,
        message=(
            f"Qdrant is reachable at {settings.qdrant_url} "
            f"with {collection_count} collections."
        ),
    )


def _check_local_provider(settings: Settings) -> DiagnosticCheck:
    url = settings.local_llm_base_url.rstrip("/") + "/models"
    try:
        response = httpx.get(url, timeout=5.0)
    except Exception as exc:
        return DiagnosticCheck(
            name="local_provider",
            status=DiagnosticStatus.FAIL,
            message=f"Local provider reachability check failed: {exc}",
            hint="Start the local OpenAI-compatible server and verify LOCAL_LLM_BASE_URL.",
        )
    if 200 <= response.status_code < 300:
        return DiagnosticCheck(
            name="local_provider",
            status=DiagnosticStatus.PASS,
            message=f"Local provider responded successfully from {url}.",
        )
    if response.status_code in {401, 403}:
        return DiagnosticCheck(
            name="local_provider",
            status=DiagnosticStatus.FAIL,
            message=f"Local provider rejected the request with HTTP {response.status_code}.",
            hint="Check LOCAL_LLM_API_KEY and the provider's auth expectations.",
        )
    if response.status_code in {404, 405}:
        return DiagnosticCheck(
            name="local_provider",
            status=DiagnosticStatus.WARN,
            message=f"Local provider is reachable, but {url} returned HTTP {response.status_code}.",
            hint=(
                "This endpoint may not expose `/models`; verify the server supports "
                "the expected OpenAI-compatible surface."
            ),
        )
    return DiagnosticCheck(
        name="local_provider",
        status=DiagnosticStatus.FAIL,
        message=f"Local provider returned HTTP {response.status_code}.",
        hint="Inspect the local provider logs and verify LOCAL_LLM_BASE_URL.",
    )


def _skipped_qdrant_check() -> DiagnosticCheck:
    return DiagnosticCheck(
        name="qdrant",
        status=DiagnosticStatus.SKIPPED,
        message="Qdrant check skipped. Use --check-qdrant to verify.",
    )


def _skipped_provider_check() -> DiagnosticCheck:
    return DiagnosticCheck(
        name="local_provider",
        status=DiagnosticStatus.SKIPPED,
        message="Local provider check skipped. Use --check-local-provider to verify.",
    )


def _overall_status(checks: list[DiagnosticCheck]) -> DiagnosticStatus:
    if any(check.status == DiagnosticStatus.FAIL for check in checks):
        return DiagnosticStatus.FAIL
    if any(check.status == DiagnosticStatus.WARN for check in checks):
        return DiagnosticStatus.WARN
    return DiagnosticStatus.PASS
