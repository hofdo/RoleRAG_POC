from __future__ import annotations

from pathlib import Path

from pytest import MonkeyPatch

from app.config import Settings
from app.diagnostics.runtime_checks import (
    DiagnosticStatus,
    build_runtime_diagnostics,
    is_usable_cloud_key,
)
from app.llm.router import CloudMode
from app.rag.models import RagCollection
from app.rag.vector_store import VectorStoreModelMismatch


def test_is_usable_cloud_api_key_rejects_placeholders() -> None:
    assert is_usable_cloud_key("replace_me") is False
    assert is_usable_cloud_key("  your_api_key_here  ") is False
    assert is_usable_cloud_key("") is False
    assert is_usable_cloud_key("real-secret") is True


def test_runtime_diagnostics_passes_config_and_skips_network_checks_by_default(
    tmp_path: Path,
) -> None:
    settings = Settings(  # type: ignore[call-arg]
        _env_file=tmp_path / ".missing",
        database_path=str(tmp_path / "configured.db"),
        cloud_mode=CloudMode.OFF,
    )

    report = build_runtime_diagnostics(settings)

    checks = {check.name: check for check in report.checks}
    assert report.status == DiagnosticStatus.PASS
    assert checks["settings"].status == DiagnosticStatus.PASS
    assert checks["sqlite"].status == DiagnosticStatus.PASS
    assert checks["demo_data"].status == DiagnosticStatus.PASS
    assert checks["qdrant"].status == DiagnosticStatus.SKIPPED
    assert checks["local_provider"].status == DiagnosticStatus.SKIPPED


def test_runtime_diagnostics_warns_when_cloud_mode_ask_has_placeholder_key(tmp_path: Path) -> None:
    settings = Settings(  # type: ignore[call-arg]
        _env_file=tmp_path / ".missing",
        database_path=str(tmp_path / "configured.db"),
        cloud_mode=CloudMode.ASK,
        cloud_llm_api_key="replace_me",
    )

    report = build_runtime_diagnostics(settings)

    cloud_check = next(check for check in report.checks if check.name == "cloud_config")
    assert cloud_check.status == DiagnosticStatus.WARN
    assert "placeholder" in cloud_check.message.lower()
    assert "replace_me" not in cloud_check.message


def test_runtime_diagnostics_fails_when_cloud_mode_auto_has_placeholder_key(tmp_path: Path) -> None:
    settings = Settings(  # type: ignore[call-arg]
        _env_file=tmp_path / ".missing",
        database_path=str(tmp_path / "configured.db"),
        cloud_mode=CloudMode.AUTO,
        cloud_llm_api_key="replace_me",
    )

    report = build_runtime_diagnostics(settings)

    cloud_check = next(check for check in report.checks if check.name == "cloud_config")
    assert report.status == DiagnosticStatus.FAIL
    assert cloud_check.status == DiagnosticStatus.FAIL


def test_runtime_diagnostics_never_leaks_api_keys(tmp_path: Path) -> None:
    settings = Settings(  # type: ignore[call-arg]
        _env_file=tmp_path / ".missing",
        database_path=str(tmp_path / "configured.db"),
        local_llm_api_key="super-secret-local",
        cloud_llm_api_key="super-secret-cloud",
    )

    report = build_runtime_diagnostics(settings)
    payload = report.model_dump(mode="json")
    serialized = str(payload)

    assert "super-secret-local" not in serialized
    assert "super-secret-cloud" not in serialized


def test_runtime_diagnostics_sends_local_provider_auth_header(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        status_code = 200

    def fake_get(url: str, *, headers: dict[str, str], timeout: float) -> FakeResponse:
        captured["url"] = url
        captured["headers"] = headers
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("app.diagnostics.runtime_checks.httpx.get", fake_get)
    settings = Settings(  # type: ignore[call-arg]
        _env_file=tmp_path / ".missing",
        database_path=str(tmp_path / "configured.db"),
        cloud_mode=CloudMode.OFF,
        local_llm_base_url="http://127.0.0.1:8080/v1",
        local_llm_api_key="super-secret-local",
    )

    report = build_runtime_diagnostics(settings, check_local_provider=True)

    checks = {check.name: check for check in report.checks}
    assert checks["local_provider"].status == DiagnosticStatus.PASS
    assert captured == {
        "url": "http://127.0.0.1:8080/v1/models",
        "headers": {"Authorization": "Bearer super-secret-local"},
        "timeout": 5.0,
    }
    assert "super-secret-local" not in str(report.model_dump(mode="json"))


class _FakeQdrantClientForFingerprintCheck:
    def get_collections(self) -> object:
        class _Collections:
            collections: list[object] = []

        return _Collections()


class _FakeQdrantVectorStore:
    """Stands in for QdrantVectorStore in the `doctor --check-qdrant` path: reachable,
    zero collections, with a canned fingerprint lookup (P1.4)."""

    def __init__(self, *, fingerprints: dict[RagCollection, str]) -> None:
        self._fingerprints = fingerprints
        self.client = _FakeQdrantClientForFingerprintCheck()

    def read_model_fingerprint(self, collection: RagCollection) -> str | None:
        return self._fingerprints.get(collection)


def test_doctor_check_qdrant_surfaces_embedding_model_fingerprint_mismatch(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    fake_store = _FakeQdrantVectorStore(
        fingerprints={RagCollection.CANON_LORE: "paraphrase-multilingual-MiniLM-L12-v2"}
    )
    monkeypatch.setattr(
        "app.diagnostics.runtime_checks.QdrantVectorStore", lambda url: fake_store
    )
    settings = Settings(  # type: ignore[call-arg]
        _env_file=tmp_path / ".missing",
        database_path=str(tmp_path / "configured.db"),
        cloud_mode=CloudMode.OFF,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )

    report = build_runtime_diagnostics(settings, check_qdrant=True)

    checks = {check.name: check for check in report.checks}
    qdrant_check = checks["qdrant"]
    assert qdrant_check.status == DiagnosticStatus.FAIL
    assert report.status == DiagnosticStatus.FAIL
    assert "canon_lore" in qdrant_check.message
    assert "paraphrase-multilingual-MiniLM-L12-v2" in qdrant_check.message
    assert "sentence-transformers/all-MiniLM-L6-v2" in qdrant_check.message
    assert qdrant_check.hint == VectorStoreModelMismatch.RUNBOOK_HINT


def test_doctor_check_qdrant_passes_when_fingerprint_matches_or_is_absent(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    fake_store = _FakeQdrantVectorStore(
        fingerprints={
            RagCollection.CANON_LORE: "sentence-transformers/all-MiniLM-L6-v2",
            # SESSION_MEMORY/PERSONA_MEMORY absent -> unfingerprinted (pre-P1.4), must not fail.
        }
    )
    monkeypatch.setattr(
        "app.diagnostics.runtime_checks.QdrantVectorStore", lambda url: fake_store
    )
    settings = Settings(  # type: ignore[call-arg]
        _env_file=tmp_path / ".missing",
        database_path=str(tmp_path / "configured.db"),
        cloud_mode=CloudMode.OFF,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )

    report = build_runtime_diagnostics(settings, check_qdrant=True)

    checks = {check.name: check for check in report.checks}
    assert checks["qdrant"].status == DiagnosticStatus.PASS
    assert report.status == DiagnosticStatus.PASS
