from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any


class StructuredOutputParseError(ValueError):
    pass


def parse_single_json_object(text: str) -> dict[str, Any]:
    """Parse strict JSON, or extract one wrapped top-level JSON object."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        pass
    else:
        if isinstance(payload, dict):
            return payload
        raise StructuredOutputParseError("structured output must be a JSON object")

    candidates = list(_top_level_object_candidates(text))
    if len(candidates) != 1:
        raise StructuredOutputParseError("structured output must contain exactly one JSON object")

    try:
        payload = json.loads(candidates[0])
    except json.JSONDecodeError as exc:
        raise StructuredOutputParseError("structured output is malformed JSON") from exc
    if not isinstance(payload, dict):
        raise StructuredOutputParseError("structured output must be a JSON object")
    return payload


def _top_level_object_candidates(text: str) -> Iterator[str]:
    depth = 0
    start: int | None = None
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
            continue
        if char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start : index + 1]
                start = None
