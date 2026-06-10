from __future__ import annotations

import json
from pathlib import Path

from app.diagnostics.structured_failures import StructuredOutputFailureSink


def test_structured_failure_sink_appends_jsonl_records(tmp_path: Path) -> None:
    sink = StructuredOutputFailureSink(directory=tmp_path / "failures")

    sink.record(
        task="critic",
        category="schema",
        raw_text='{"accepted": "maybe"}',
        model="local-model",
    )
    sink.record(
        task="memory_extraction",
        category="parse",
        raw_text="not json",
        model="local-model",
        session_id="session-1",
    )

    path = tmp_path / "failures" / "structured-output-failures.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert records[0]["task"] == "critic"
    assert records[0]["category"] == "schema"
    assert records[0]["raw_text"] == '{"accepted": "maybe"}'
    assert records[0]["model"] == "local-model"
    assert records[0]["session_id"] is None
    assert "recorded_at_utc" in records[0]
    assert records[1]["task"] == "memory_extraction"
    assert records[1]["session_id"] == "session-1"
