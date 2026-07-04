"""Deterministic guards against hidden facts leaking through generated text.

Two collaborators wrap the LLM around hidden persona/scene fields (secrets,
forbidden_knowledge, gm_private_summary). ``redact_hidden_facts`` runs in the
critique stage: the critic is shown those fields to detect leakage in a draft
but must not repeat them in its issues/repair_instruction (which feed the next
actor generation), so any verbatim echo is stripped before its output is reused.
``scan_reply`` runs in the orchestrator as an output-side backstop on the actor
reply itself: it redacts verbatim/per-sentence echoes and additionally flags
likely paraphrases via content-word overlap, since models can confabulate
secrets the prompt never contained. ``collect_hidden_facts`` gathers the fields
both guards operate over.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.domain import PersonaCard, SceneState

_REDACTION = "[redacted]"
_MIN_FACT_CHARS = 8
_SENTENCE = re.compile(r"(?<=[.!?])\s+")

_WORD = re.compile(r"[a-z0-9']+")
# Common words carry no secret-identifying signal; dropping them keeps the
# content-token overlap from firing on incidental shared filler.
_STOPWORDS = frozenset(
    "a an the and or but if then of to in on at by for with from into as is are was were be "
    "been being it its this that these those he she they them his her their you your i we our "
    "not no nor so do does did has have had will would can could may might must shall should "
    "who whom whose which what when where why how than because she's he's".split()
)
# Default fraction of a hidden fact's content words that must appear in a reply
# before it is flagged as a possible paraphrase/confabulation of that fact.
DEFAULT_PARAPHRASE_OVERLAP = 0.6


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


def collect_hidden_facts(persona: PersonaCard, scene: SceneState) -> list[str]:
    """Facts that must never reach player-facing output: persona secrets,
    forbidden_knowledge, private_description, and the scene's GM-only summary."""
    facts = [*persona.secrets, *persona.forbidden_knowledge]
    if persona.private_description:
        facts.append(persona.private_description)
    if scene.gm_private_summary:
        facts.append(scene.gm_private_summary)
    return facts


@dataclass(frozen=True)
class ReplyContainmentReport:
    text: str  # the reply with any verbatim hidden-fact echoes redacted
    verbatim_redacted: bool  # an exact/sentence echo was found and removed
    paraphrased_facts: tuple[str, ...]  # facts the reply likely paraphrases (flag only)

    @property
    def flagged(self) -> bool:
        return self.verbatim_redacted or bool(self.paraphrased_facts)


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in _WORD.findall(text.lower())
        if len(token) > 2 and token not in _STOPWORDS
    }


def scan_reply(
    reply: str,
    hidden_facts: Sequence[str],
    *,
    paraphrase_overlap_threshold: float = DEFAULT_PARAPHRASE_OVERLAP,
) -> ReplyContainmentReport:
    """Output-side containment backstop for the actor reply.

    Two layers behind the LLM critic: (1) deterministically redact any verbatim or
    per-sentence echo of a hidden fact; (2) flag (do not edit) facts the reply likely
    paraphrases, measured by the fraction of the fact's content words present in the
    reply. The bake-off showed models confabulate secrets the prompt never contained, so
    verbatim redaction alone is insufficient — the overlap flag catches paraphrase.
    """
    units = _matchable_units(hidden_facts)
    redacted, verbatim = _redact_one(reply, units) if units else (reply, False)

    reply_tokens = _content_tokens(redacted)
    paraphrased: list[str] = []
    for fact in hidden_facts:
        fact_tokens = _content_tokens(fact)
        if not fact_tokens:
            continue
        coverage = len(fact_tokens & reply_tokens) / len(fact_tokens)
        if coverage >= paraphrase_overlap_threshold:
            paraphrased.append(fact)
    return ReplyContainmentReport(
        text=redacted,
        verbatim_redacted=verbatim,
        paraphrased_facts=tuple(paraphrased),
    )
