from __future__ import annotations

from typing import Protocol


class StructuredFailureRecording(Protocol):
    def record(
        self,
        *,
        task: str,
        category: str,
        raw_text: str,
        model: str,
        session_id: str | None = None,
    ) -> None: ...
