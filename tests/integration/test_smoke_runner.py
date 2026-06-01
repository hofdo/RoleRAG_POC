from __future__ import annotations

from app.diagnostics.runtime_checks import DiagnosticStatus
from app.diagnostics.smoke_runner import run_smoke


def test_smoke_runner_completes_with_fake_provider_and_temp_sqlite() -> None:
    summary = run_smoke()

    assert summary.status == DiagnosticStatus.PASS
    assert summary.mode == "safe"
    assert summary.session_id is not None
    assert summary.persisted_turns == 1
    assert summary.memory_written is True
    assert summary.persisted_memories == 1
    assert summary.indexed_memories == 1
    assert summary.ingested_chunk_count >= 1
    assert summary.retrieval_selected_count >= 1
    assert summary.temp_database_path is not None


def test_smoke_runner_returns_metadata_only_retrieval_summary() -> None:
    summary = run_smoke()

    assert summary.status == DiagnosticStatus.PASS
    assert summary.retrieval_selected
    assert "archive" in summary.actor_response.lower()
    assert "Hidden" not in str(summary.model_dump(mode="json"))
