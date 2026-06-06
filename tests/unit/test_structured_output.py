from __future__ import annotations

import pytest

from app.llm.structured_output import StructuredOutputParseError, parse_single_json_object


def test_parse_single_json_object_accepts_strict_json() -> None:
    assert parse_single_json_object('{"accepted": true, "issues": []}') == {
        "accepted": True,
        "issues": [],
    }


def test_parse_single_json_object_accepts_fenced_json() -> None:
    assert parse_single_json_object(
        '```json\n{"accepted": true, "issues": [], "repair_instruction": null}\n```'
    )["accepted"] is True


def test_parse_single_json_object_accepts_prefixed_json() -> None:
    assert parse_single_json_object(
        'Here is the requested JSON:\n{"write_memory": false, "memories": [], "reason": "trivial"}'
    ) == {"write_memory": False, "memories": [], "reason": "trivial"}


def test_parse_single_json_object_rejects_malformed_json() -> None:
    with pytest.raises(StructuredOutputParseError):
        parse_single_json_object('{"accepted": true')


def test_parse_single_json_object_rejects_multiple_objects() -> None:
    with pytest.raises(StructuredOutputParseError):
        parse_single_json_object('{"accepted": true}{"accepted": false}')


def test_parse_single_json_object_rejects_arrays() -> None:
    with pytest.raises(StructuredOutputParseError):
        parse_single_json_object('[{"accepted": true}]')
