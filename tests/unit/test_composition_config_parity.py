"""Guards against the two composition roots drifting apart (#48, #67).

The CLI (`app.cli._build_services`) and the API (`app.composition.build_services`)
each assemble a `TurnOrchestrator`. #48 fixed `TurnOrchestratorConfig` parity: both
must derive it from the same `Settings` via `build_orchestrator_config`, or
`.env`-backed knobs silently apply on one surface and not the other (the original
bug: the CLI ran the local critic/memory extractor at the 350-token dataclass
default while the API used the configured 640).

#67 went further: `app.cli._build_services` now *delegates* to
`app.composition.build_services` outright instead of re-assembling the
orchestrator's collaborators locally. That re-assembly had drifted from the
canonical composition root -- the CLI silently omitted `canon_repository` (author
-pinned canon facts never reached CLI turns), `structured_failure_sink` (CLI
structured-output failures were never recorded), and `memory_embedding_provider`
(semantic write-dedup could never activate on CLI turns). Delegation makes that
drift structurally impossible: the tests below assert both surfaces produce
orchestrators wired with the same collaborators, not just the same config values.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from app.cli import _build_services, _build_vector_store
from app.composition import build_orchestrator_config, build_services, build_vector_store
from app.config import Settings


def test_cli_build_services_uses_canonical_orchestrator_config(tmp_path: Path) -> None:
    settings = Settings(database_path=str(tmp_path / "sessions.db"))
    with (
        patch("app.composition.build_local_provider", return_value=MagicMock()),
        patch("app.composition.build_cloud_provider", return_value=MagicMock()),
        patch("app.composition.build_critic_agent", return_value=MagicMock()),
        patch("app.composition.build_memory_curator", return_value=MagicMock()),
        patch("app.composition.build_file_loader", return_value=MagicMock()),
    ):
        services = _build_services(settings, enable_retrieval=False, content_root=tmp_path)

    expected = build_orchestrator_config(settings, content_root=str(tmp_path))
    assert services.orchestrator.config == expected


def test_cli_orchestrator_config_honors_local_structured_max_tokens(tmp_path: Path) -> None:
    # The field the drift regressed: a non-default value must reach the CLI orchestrator.
    settings = Settings(
        database_path=str(tmp_path / "sessions.db"),
        local_structured_max_tokens=777,
    )
    with (
        patch("app.composition.build_local_provider", return_value=MagicMock()),
        patch("app.composition.build_cloud_provider", return_value=MagicMock()),
        patch("app.composition.build_critic_agent", return_value=MagicMock()),
        patch("app.composition.build_memory_curator", return_value=MagicMock()),
        patch("app.composition.build_file_loader", return_value=MagicMock()),
    ):
        services = _build_services(settings, enable_retrieval=False, content_root=tmp_path)

    assert services.orchestrator.config.local_structured_max_tokens == 777


def test_cli_build_services_delegates_to_composition_build_services(tmp_path: Path) -> None:
    # The concrete #67 fix: the CLI no longer re-assembles the orchestrator locally -- it
    # calls the composition root's factory directly (imported by name into app.cli), so
    # patching that name is sufficient to prove delegation rather than local re-assembly.
    settings = Settings(database_path=str(tmp_path / "sessions.db"))
    with patch("app.cli.build_services") as mock_build_services:
        _build_services(settings, enable_retrieval=True)

    mock_build_services.assert_called_once_with(
        settings, enable_retrieval=True, content_root=None
    )


def test_cli_and_api_build_services_wire_the_same_collaborators(tmp_path: Path) -> None:
    # The concrete #67 bug: the CLI's hand-rolled orchestrator omitted canon_repository,
    # structured_failure_sink, and memory_embedding_provider. Assert both composition
    # roots now populate the same collaborators on the orchestrators they build.
    settings = Settings(
        database_path=str(tmp_path / "sessions.db"),
        structured_output_failure_log_dir=tmp_path / "structured-failures",
    )
    fake_embedding_provider = MagicMock(name="embedding_provider")
    fake_vector_store = MagicMock(name="vector_store")

    with (
        patch("app.composition.build_local_provider", return_value=MagicMock()),
        patch("app.composition.build_cloud_provider", return_value=None),
        patch("app.composition.build_critic_agent", return_value=MagicMock()),
        patch("app.composition.build_memory_curator", return_value=MagicMock()),
        patch("app.composition.build_embedding_provider", return_value=fake_embedding_provider),
        patch("app.composition.build_vector_store", return_value=fake_vector_store),
    ):
        cli_services = _build_services(settings, enable_retrieval=True, content_root=tmp_path)
        api_services = build_services(settings, enable_retrieval=True, content_root=tmp_path)

    for label, services in (("cli", cli_services), ("api", api_services)):
        assert services.canon_repository is not None, f"{label}: canon_repository missing"
        assert services.memory_indexer is not None, f"{label}: memory_indexer missing"
        assert services.turn_repository is not None, f"{label}: turn_repository missing"
        assert services.memory_repository is not None, f"{label}: memory_repository missing"

        orchestrator = services.orchestrator
        assert orchestrator.session_stage.canon_repository is services.canon_repository, (
            f"{label}: canon_repository not wired into session_stage -- author-pinned "
            "canon facts would not reach this surface's Standing facts"
        )
        assert orchestrator.critique_stage.failure_sink is not None, (
            f"{label}: structured_failure_sink not wired into critique_stage"
        )
        assert orchestrator.memory_stage.failure_sink is not None, (
            f"{label}: structured_failure_sink not wired into memory_stage"
        )
        dedup_embedding_provider = orchestrator.memory_stage._deduplicator.embedding_provider
        assert dedup_embedding_provider is fake_embedding_provider, (
            f"{label}: memory_embedding_provider not wired -- semantic write-dedup can "
            "never activate"
        )


def test_cli_and_api_wire_canon_tag_pinning_into_session_and_consolidator(
    tmp_path: Path,
) -> None:
    """docs/26 §3.3 (#78). canon_tag_pinning has two consumers -- TurnSessionLoader
    (build_standing_facts' widened eligibility + stale-fact safeguard) and
    MemoryConsolidator (select_consolidatable's German preserve-tags), the latter
    reached via TurnMemoryStage -- and both must receive the flag on both
    composition roots, the same #67-class concern
    test_cli_and_api_build_services_wire_the_same_collaborators pins for the other
    collaborators."""
    settings = Settings(
        database_path=str(tmp_path / "sessions.db"),
        canon_tag_pinning=True,
    )

    with (
        patch("app.composition.build_local_provider", return_value=MagicMock()),
        patch("app.composition.build_cloud_provider", return_value=None),
        patch("app.composition.build_critic_agent", return_value=MagicMock()),
        patch("app.composition.build_memory_curator", return_value=MagicMock()),
    ):
        cli_services = _build_services(settings, enable_retrieval=False, content_root=tmp_path)
        api_services = build_services(settings, enable_retrieval=False, content_root=tmp_path)

    for label, services in (("cli", cli_services), ("api", api_services)):
        orchestrator = services.orchestrator
        assert orchestrator.config.canon_tag_pinning is True, (
            f"{label}: canon_tag_pinning not set on TurnOrchestratorConfig"
        )
        assert orchestrator.session_stage.canon_tag_pinning is True, (
            f"{label}: canon_tag_pinning not wired into session_stage"
        )
        assert orchestrator.memory_stage._consolidator.canon_tag_pinning is True, (
            f"{label}: canon_tag_pinning not wired into memory_stage's consolidator"
        )


def test_cli_vector_store_builder_is_the_composition_root_function() -> None:
    """P2.1 (docs/22): settings.qdrant_scalar_quantization (and any future
    QdrantVectorStore-construction knob) must reach both composition roots identically.
    Unlike the collaborator-wiring drift #67 fixed, the CLI never re-assembles
    QdrantVectorStore at all -- ``app.cli._build_vector_store`` is a direct alias of
    ``app.composition.build_vector_store``, not an independent construction path. This pins
    that alias so a future refactor can't reintroduce the #48/#67 drift class here without
    failing a test."""
    assert _build_vector_store is build_vector_store
