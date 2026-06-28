from __future__ import annotations

from collections.abc import Sequence

from app.domain import TurnError

# Ordered (prefix, stage, category, suggestion); first matching prefix wins, so more specific
# prefixes are listed before their generic forms. The warning strings are produced by the stages
# with stable prefixes, so matching on them keeps the structured view in lockstep without forcing
# every warning site to construct a TurnError itself.
_RULES: tuple[tuple[str, str, str, str | None], ...] = (
    (
        "retrieval skipped",
        "retrieval",
        "degraded",
        "Retrieval was unavailable; verify the vector store (Qdrant) is reachable.",
    ),
    (
        "local actor failed",
        "generation",
        "provider_unavailable",
        "The local model failed; check the model server.",
    ),
    (
        "cloud actor skipped",
        "routing",
        "cloud_skipped",
        "Cloud would have been used; enable CLOUD_MODE or confirm the route to allow it.",
    ),
    ("cloud repair skipped", "repair", "cloud_skipped", None),
    ("draft validation repair failed", "repair", "validation_repair_failed", None),
    ("repair failed", "repair", "repair_failed", None),
    (
        "draft withheld",
        "critique",
        "fail_closed",
        "The critic could not validate the draft, so it was withheld for safety.",
    ),
    ("critic gated", "critique", "gated", None),
    ("critic output redacted", "critique", "security", None),
    ("critic skipped", "critique", "degraded", None),
    (
        "containment risk",
        "containment",
        "security",
        "The reply may paraphrase a hidden fact; review it before sharing.",
    ),
    ("containment:", "containment", "security", None),
    ("memory curation gated", "memory", "gated", None),
    ("memory curation skipped", "memory", "curation_failed", None),
    ("memory consolidation", "memory", "consolidation", None),
    ("semantic memory dedup skipped", "memory", "dedup_skipped", None),
    ("semantic memory dedup dropped", "memory", "dedup", None),
    ("memory dedup skipped", "memory", "dedup_skipped", None),
    ("memory dedup dropped", "memory", "dedup", None),
    ("memory indexing skipped", "memory", "index_failed", None),
    ("deterministic memory fallback skipped", "memory", "fallback_failed", None),
    ("deterministic memory fallback added", "memory", "fallback", None),
    ("structured failure log skipped", "diagnostics", "logging", None),
)


def classify_warning(warning: str) -> TurnError:
    """Map one free-form warning string to a structured TurnError (category/stage/suggestion)."""
    for prefix, stage, category, suggestion in _RULES:
        if warning.startswith(prefix):
            return TurnError(
                category=category, stage=stage, message=warning, suggestion=suggestion
            )
    return TurnError(category="warning", stage="general", message=warning, suggestion=None)


def classify_warnings(warnings: Sequence[str]) -> list[TurnError]:
    return [classify_warning(warning) for warning in warnings]
