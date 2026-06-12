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
COVERAGE_THRESHOLD = 0.5

_PROMISE_PATTERN = re.compile(
    r"\bi\s+(?:promise|swear|vow|pledge)\b|\bi\s+give\s+(?:you\s+)?my\s+word\b",
    re.IGNORECASE,
)
_ENTRUST_PATTERN = re.compile(
    r"\bi\s+entrust\b|\bi\s+hand\s+(?:you|over)\b",
    re.IGNORECASE,
)
_DEADLINE_PATTERN = re.compile(
    r"\bbefore\s+(?:dawn|sunrise|midnight|nightfall|sunset)\b",
    re.IGNORECASE,
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_DURABLE_TERM_PATTERN = re.compile(
    r"\b(?:promise[sd]?|swear[s]?|swore|vow[sed]*|pledge[sd]?|oath|entrust[sed]*)\b",
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
        else:
            continue
        if _DEADLINE_PATTERN.search(sentence):
            tags.append("deadline")
        candidates.append(
            MemoryCandidate(
                # Verbatim sentence keeps the exact event keys retrievable by
                # the lexical ranking boost.
                summary=f'The player stated: "{sentence}"',
                visibility=Visibility.PLAYER,
                importance=DETERMINISTIC_EVENT_IMPORTANCE,
                tags=tags,
                scene_id=scene_id,
                actor_id=actor_id,
            )
        )
    return candidates


def is_covered_by_summaries(
    candidate_summary: str,
    summaries: Sequence[str],
    *,
    threshold: float = COVERAGE_THRESHOLD,
) -> bool:
    """True when an existing summary already carries most of the candidate's terms."""
    candidate_terms = content_terms(candidate_summary)
    if not candidate_terms:
        return True
    for summary in summaries:
        overlap = len(candidate_terms & content_terms(summary))
        if overlap / len(candidate_terms) >= threshold:
            return True
    return False
