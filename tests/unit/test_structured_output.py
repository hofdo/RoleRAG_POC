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


def test_inline_schema_refs_resolves_nested_defs() -> None:
    from app.llm.structured_output import inline_schema_refs

    schema = {
        "$defs": {
            "Visibility": {"enum": ["player", "gm"], "type": "string"},
            "Candidate": {
                "type": "object",
                "properties": {"visibility": {"$ref": "#/$defs/Visibility"}},
                "required": ["visibility"],
            },
        },
        "type": "object",
        "properties": {
            "memories": {"type": "array", "items": {"$ref": "#/$defs/Candidate"}},
        },
    }

    inlined = inline_schema_refs(schema)

    items = inlined["properties"]["memories"]["items"]
    assert items["properties"]["visibility"] == {"enum": ["player", "gm"], "type": "string"}
    assert "$defs" not in inlined
    assert "$ref" not in str(inlined)


def test_inline_schema_refs_returns_ref_free_schema_unchanged() -> None:
    from app.llm.structured_output import inline_schema_refs

    schema = {"type": "object", "properties": {"accepted": {"type": "boolean"}}}

    assert inline_schema_refs(schema) == schema
