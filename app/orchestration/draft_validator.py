"""Deterministic draft validation: unsupported entities and evaded player actions.

Zero-LLM checks that run between generation and critique. Flags are report-only
warnings; they also feed the critique stage so flagged drafts can be repaired.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.rag.ranking import content_terms

_CAPITALISED_WORD = re.compile(r"^[A-Z][a-z]{2,}$")
_SENTENCE_SPLIT = re.compile(r"[.!?]+")
_DIRECT_ACTION_PREFIX = re.compile(r"^i['\s]", re.IGNORECASE)

UNSUPPORTED_ENTITY_FLAG_PREFIX = "draft validation: unsupported entity"
EVADED_ACTION_FLAG = "draft validation: player action unaddressed"


@dataclass(frozen=True)
class DraftValidationResult:
    unsupported_entities: tuple[str, ...] = ()
    evaded_action: bool = False
    flags: list[str] = field(default_factory=list)


def validate_draft(
    *,
    draft: str,
    player_message: str,
    visible_texts: list[str],
) -> DraftValidationResult:
    vocabulary = _vocabulary([*visible_texts, player_message])
    unsupported = tuple(
        entity
        for entity in _entity_candidates(draft)
        if not _entity_is_known(entity, vocabulary)
    )
    evaded = _player_action_evaded(draft=draft, player_message=player_message)

    flags = [f"{UNSUPPORTED_ENTITY_FLAG_PREFIX} {entity}" for entity in unsupported]
    if evaded:
        flags.append(EVADED_ACTION_FLAG)
    return DraftValidationResult(
        unsupported_entities=unsupported,
        evaded_action=evaded,
        flags=flags,
    )


def _vocabulary(texts: list[str]) -> set[str]:
    vocabulary: set[str] = set()
    for text in texts:
        for raw_token in text.lower().split():
            token = "".join(char for char in raw_token if char.isalpha())
            if not token:
                continue
            vocabulary.add(token)
            vocabulary.update(content_terms(token))
    return vocabulary


def _entity_candidates(draft: str) -> list[str]:
    candidates: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(draft):
        words = sentence.split()
        index = 0
        while index < len(words):
            stripped = words[index].strip("\"'(),;:")
            if _CAPITALISED_WORD.match(stripped):
                group = [stripped]
                cursor = index + 1
                while cursor < len(words):
                    next_word = words[cursor].strip("\"'(),;:")
                    if not _CAPITALISED_WORD.match(next_word):
                        break
                    group.append(next_word)
                    cursor += 1
                # A lone capitalised word opening a sentence is ordinary prose;
                # a capitalised run there is still a name ("Duke Erran left...").
                if index > 0 or len(group) > 1:
                    candidate = " ".join(group)
                    if candidate not in candidates:
                        candidates.append(candidate)
                index = cursor
            else:
                index += 1
    return candidates


def _entity_is_known(entity: str, vocabulary: set[str]) -> bool:
    for word in entity.lower().split():
        if word in vocabulary or content_terms(word) & vocabulary:
            return True
    return False


def _player_action_evaded(*, draft: str, player_message: str) -> bool:
    is_direct = "?" in player_message or bool(
        _DIRECT_ACTION_PREFIX.match(player_message.strip())
    )
    if not is_direct:
        return False
    message_terms = content_terms(player_message)
    # A single content term gives the actor almost nothing concrete to echo;
    # only treat richer messages as evadable to avoid false positives.
    if len(message_terms) < 2:
        return False
    return not (message_terms & content_terms(draft))
