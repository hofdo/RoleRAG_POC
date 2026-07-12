"""Conservative deterministic extraction of explicit durable events.

This is the fallback path for the live failure documented in
docs/13_live_model_quality_assessment.md: model-based memory curation missed
an explicit player promise. Patterns are intentionally narrow (first-person,
explicit trigger verbs) so that normal narration never produces memories.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.domain import MemoryCandidate, Visibility
from app.rag.ranking import content_terms

DETERMINISTIC_EVENT_IMPORTANCE = 4
# Default for the deterministic-fallback "did the curator already cover this
# explicit event" check (stages/memory.py) — unchanged at 0.5.
COVERAGE_THRESHOLD = 0.5
# Write-dedup drop threshold (stages/memory_dedup.py). At the old shared 0.5 a
# terse distinct fact was silently dropped: "The player gave Iria Vale a silver
# compass to keep until his return" hit 0.56 coverage against a compass-free
# promise memory sharing only {player, iria, vale, keep, return} (docs/22 P2.2
# live finding, 2026-07-12; run-dependent, since extractor phrasing length
# varies). 0.75 keeps verbatim/near-verbatim repeats dropping (~1.0 coverage)
# while a terse distinct fact survives.
WRITE_DEDUP_COVERAGE_THRESHOLD = 0.75

# Constant framing every deterministic candidate is wrapped in. Kept verbatim in
# the stored summary (it keeps the exact event keys retrievable by the lexical
# boost), but stripped before coverage-term math: otherwise the shared
# {player, stated} vocabulary inflates every write-dedup overlap toward the drop
# threshold and false-drops distinct facts (docs/22 N1).
_FRAMING_LEAD = "The player stated:"

_REVERSAL_MARKERS = re.compile(
    r"\b(?:not|never|no\s+longer|stopped|refused?|betray(?:ed|s)?|broke|broken"
    r"|revoked?|withdrew|withdrawn|abandoned|died|dead|ended)\b",
    re.IGNORECASE,
)


def _reversal_markers(text: str) -> set[str]:
    return {" ".join(match.split()).lower() for match in _REVERSAL_MARKERS.findall(text)}

_PROMISE_PATTERN = re.compile(
    # Explicit first-person commitment verbs, "(you have) my word", and narrow
    # negative commitments that are durable only because of the never/not before
    # a high-stakes verb ("I will never tell" — but not plain "I will tell you").
    r"\bi\s+(?:promise|swear|vow|pledge)\b"
    r"|\b(?:i\s+give\s+(?:you\s+)?my\s+word|you\s+have\s+my\s+word)\b"
    r"|\bi\s+will\s+(?:never|not)\s+"
    r"(?:tell|reveal|betray|forget|abandon|harm|break|leave)\b"
    r"|\bi'll\s+(?:never|not)\s+"
    r"(?:tell|reveal|betray|forget|abandon|harm|break|leave)\b"
    r"|\bi\s+won'?t\s+(?:tell|reveal|betray|forget|abandon|harm|break|leave)\b",
    re.IGNORECASE,
)
_ENTRUST_PATTERN = re.compile(
    r"\bi\s+entrust\b|\bi\s+hand\s+(?:you|over)\b",
    re.IGNORECASE,
)
_AGREEMENT_PATTERN = re.compile(
    # Mutual agreements and established standing facts (rules, signals, codes,
    # passwords, deals). High precision: declarative "the X is" / "we agreed".
    r"\bwe\s+(?:agree|agreed)\b"
    r"|\bwe\s+have\s+(?:a\s+deal|an\s+agreement)\b"
    r"|\blet'?s\s+agree\b"
    r"|\bthe\s+(?:rule|signal|code|password|deal)\s+is\b"
    r"|\bour\s+(?:deal|agreement|rule|signal|code|password)\b",
    re.IGNORECASE,
)
_DEADLINE_PATTERN = re.compile(
    r"\bbefore\s+(?:dawn|sunrise|midnight|nightfall|sunset)\b",
    re.IGNORECASE,
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_DURABLE_TERM_PATTERN = re.compile(
    r"\b(?:promise[sd]?|swear[s]?|swore|vow[sed]*|pledge[sd]?|oath|entrust[sed]*"
    r"|agree[sd]?|agreement|rule[s]?|signal[s]?|code[s]?|arrange[sd]?|protocol[s]?"
    r"|deal[s]?|password[s]?)\b",
    re.IGNORECASE,
)


def contains_durable_event_terms(text: str) -> bool:
    """Cheap lexical check whether a message hints at a durable commitment."""
    return bool(_DURABLE_TERM_PATTERN.search(text))


def extract_explicit_durable_events(
    *,
    user_message: str,
    scene_id: str | None,
    actor_id: str | None,
) -> list[MemoryCandidate]:
    candidates: list[MemoryCandidate] = []
    for raw_sentence in _SENTENCE_BOUNDARY.split(user_message.strip()):
        sentence = raw_sentence.strip()
        if not sentence:
            continue
        tags: list[str] = []
        if _PROMISE_PATTERN.search(sentence):
            tags.append("promise")
        elif _ENTRUST_PATTERN.search(sentence):
            tags.append("entrusted")
        elif _AGREEMENT_PATTERN.search(sentence):
            tags.append("agreement")
        else:
            continue
        if _DEADLINE_PATTERN.search(sentence):
            tags.append("deadline")
        candidates.append(
            MemoryCandidate(
                # Verbatim sentence keeps the exact event keys retrievable by
                # the lexical ranking boost.
                summary=f'{_FRAMING_LEAD} "{sentence}"',
                visibility=Visibility.PLAYER,
                importance=DETERMINISTIC_EVENT_IMPORTANCE,
                tags=tags,
                scene_id=scene_id,
                actor_id=actor_id,
            )
        )
    return candidates


def _strip_framing(summary: str) -> str:
    """Drop the constant `The player stated:` framing (and its wrapping quotes) so
    coverage terms reflect the event, not the shared framing vocabulary (docs/22 N1).
    A no-op for model-curated or author summaries, which carry no such prefix."""
    text = summary.strip()
    if text.lower().startswith(_FRAMING_LEAD.lower()):
        text = text[len(_FRAMING_LEAD) :].strip()
        if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
            text = text[1:-1].strip()
    return text


def is_covered_by_summaries(
    candidate_summary: str,
    summaries: Sequence[str],
    *,
    threshold: float = COVERAGE_THRESHOLD,
) -> bool:
    """True when an existing summary already carries most of the candidate's terms."""
    candidate_terms = content_terms(_strip_framing(candidate_summary))
    if not candidate_terms:
        return True
    candidate_markers = _reversal_markers(candidate_summary)
    for summary in summaries:
        overlap = len(candidate_terms & content_terms(_strip_framing(summary)))
        if overlap / len(candidate_terms) < threshold:
            continue
        # A reversal marker present in the candidate but absent from the covering
        # summary means this is a state CHANGE, not a duplicate — always write it.
        if candidate_markers - _reversal_markers(summary):
            continue
        return True
    return False
