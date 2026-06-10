from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

FAILURE_LOG_FILENAME = "structured-output-failures.jsonl"


class StructuredOutputFailureSink:
    """Appends raw structured-output failures to a JSONL file for offline analysis."""

    def __init__(self, *, directory: Path | str) -> None:
        self.directory = Path(directory)

    def record(
        self,
        *,
        task: str,
        category: str,
        raw_text: str,
        model: str,
        session_id: str | None = None,
    ) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        entry = {
            "recorded_at_utc": datetime.now(UTC).isoformat(),
            "task": task,
            "category": category,
            "raw_text": raw_text,
            "model": model,
            "session_id": session_id,
        }
        path = self.directory / FAILURE_LOG_FILENAME
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
