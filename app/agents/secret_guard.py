"""Deterministic guard against the critic echoing hidden facts.

The critic is shown persona secrets / forbidden_knowledge / gm_private_summary
so it can detect leakage in a draft, and is instructed (prompt-only) not to
repeat them in its issues / repair_instruction. Those fields feed the next actor
generation, so an instruction-following lapse would propagate a secret. This
redacts any verbatim echo (whole fact or one of its sentences) before the
critic's output is used. It catches copied text, not paraphrase.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

_REDACTION = "[redacted]"
_MIN_FACT_CHARS = 8
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _matchable_units(facts: Sequence[str]) -> list[str]:
    """Expand hidden facts into whole-string and per-sentence units worth matching."""
    units: set[str] = set()
    for fact in facts:
        stripped = fact.strip()
        if not stripped:
            continue
        for candidate in (stripped, *_SENTENCE.split(stripped)):
            unit = candidate.strip().rstrip(".!?").strip()
            if len(unit) >= _MIN_FACT_CHARS:
                units.add(unit)
    # Longest first so the biggest verbatim match is redacted before its parts.
    return sorted(units, key=len, reverse=True)


def _redact_one(text: str, units: Sequence[str]) -> tuple[str, bool]:
    leaked = False
    for unit in units:
        lowered_unit = unit.lower()
        idx = text.lower().find(lowered_unit)
        while idx != -1:
            text = text[:idx] + _REDACTION + text[idx + len(unit) :]
            leaked = True
            idx = text.lower().find(lowered_unit)
    return text, leaked


def redact_hidden_facts(
    *,
    issues: Sequence[str],
    repair_instruction: str | None,
    hidden_facts: Sequence[str],
) -> tuple[list[str], str | None, bool]:
    """Redact verbatim hidden-fact echoes from critic output.

    Returns (redacted_issues, redacted_repair_instruction, leaked).
    """
    units = _matchable_units(hidden_facts)
    if not units:
        return list(issues), repair_instruction, False

    leaked = False
    redacted_issues: list[str] = []
    for issue in issues:
        redacted, hit = _redact_one(issue, units)
        redacted_issues.append(redacted)
        leaked = leaked or hit

    redacted_instruction = repair_instruction
    if repair_instruction is not None:
        redacted_instruction, hit = _redact_one(repair_instruction, units)
        leaked = leaked or hit

    return redacted_issues, redacted_instruction, leaked
