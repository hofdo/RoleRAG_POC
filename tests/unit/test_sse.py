from __future__ import annotations

import json

from app.api.sse import build_turn_stream_frames, serialize_error_frame, serialize_stage_frame
from app.domain import TurnResult
from app.llm.router import ModelProviderName, ModelRoute


def _result(text: str) -> TurnResult:
    return TurnResult(
        text=text,
        route=ModelRoute(
            provider=ModelProviderName.LOCAL,
            model="local",
            max_tokens=100,
            temperature=0.5,
            reason="default local route",
        ),
    )


def _text_payloads(frames: list[str]) -> list[str]:
    payloads = []
    for frame in frames:
        if frame.startswith("event: text"):
            data = frame.split("data: ", 1)[1].split("\n", 1)[0]
            payloads.append(json.loads(data)["text"])
    return payloads


def test_single_text_frame_by_default() -> None:
    frames = build_turn_stream_frames(_result("Hello, gallery."))

    assert _text_payloads(frames) == ["Hello, gallery."]
    assert any(frame.startswith("event: final") for frame in frames)


def test_validated_text_is_split_into_fragments_when_enabled() -> None:
    frames = build_turn_stream_frames(_result("abcdefg"), text_chunk_chars=3)

    fragments = _text_payloads(frames)
    assert fragments == ["abc", "def", "g"]
    # Concatenation reconstructs the validated text; a final frame still follows.
    assert "".join(fragments) == "abcdefg"
    assert any(frame.startswith("event: final") for frame in frames)


def test_short_text_stays_single_frame_even_when_chunking_enabled() -> None:
    frames = build_turn_stream_frames(_result("hi"), text_chunk_chars=10)

    assert _text_payloads(frames) == ["hi"]


def test_serialize_stage_frame() -> None:
    assert serialize_stage_frame("generation") == 'event: stage\ndata: {"stage":"generation"}\n\n'


def test_serialize_error_frame() -> None:
    frame = serialize_error_frame(code="provider_timeout", message="timed out", status=504)
    assert frame.startswith("event: error\n")
    assert '"code":"provider_timeout"' in frame
