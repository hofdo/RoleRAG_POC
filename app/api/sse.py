from __future__ import annotations

import json

from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

from app.api.schemas import (
    RouteResponse,
    StreamConfirmationPayload,
    StreamFailurePayload,
    StreamFinalPayload,
    StreamTextPayload,
    to_retrieval_diagnostics_response,
)
from app.domain import TurnOutcome, TurnResult


def build_turn_stream_frames(result: TurnResult, *, text_chunk_chars: int = 0) -> list[str]:
    route = RouteResponse(
        provider=result.route.provider.value,
        model=result.route.model,
        reason=result.route.reason,
    )
    retrieval = to_retrieval_diagnostics_response(result.retrieval)
    if result.outcome == TurnOutcome.CONFIRMATION_REQUIRED:
        return [
            _serialize_frame(
                "confirmation_required",
                StreamConfirmationPayload(
                    route=route,
                    warnings=result.warnings,
                ),
            )
        ]
    if result.outcome == TurnOutcome.CONTROLLED_FAILURE:
        return [
            _serialize_frame(
                "failure",
                StreamFailurePayload(
                    text=result.text,
                    route=route,
                    finish_reason=result.finish_reason,
                    memory_written=result.memory_written,
                    critic_status=result.critic_status.value,
                    warnings=result.warnings,
                    retrieval=retrieval,
                    stage_timings=result.stage_timings,
                ),
            )
        ]
    return [
        *_text_frames(result.text, text_chunk_chars),
        _serialize_frame(
            "final",
            StreamFinalPayload(
                route=route,
                finish_reason=result.finish_reason,
                memory_written=result.memory_written,
                critic_status=result.critic_status.value,
                warnings=result.warnings,
                retrieval=retrieval,
                stage_timings=result.stage_timings,
            ),
        ),
    ]


def _text_frames(text: str, chunk_chars: int) -> list[str]:
    """Emit the validated text as one frame, or as ordered fragments the client
    concatenates. Fragments are slices of the already-validated text, so the
    critic-before-emission boundary is preserved."""
    if chunk_chars <= 0 or len(text) <= chunk_chars:
        return [_serialize_frame("text", StreamTextPayload(text=text))]
    return [
        _serialize_frame("text", StreamTextPayload(text=text[index : index + chunk_chars]))
        for index in range(0, len(text), chunk_chars)
    ]


def _serialize_frame(event: str, payload: BaseModel) -> str:
    data = json.dumps(jsonable_encoder(payload), separators=(",", ":"))
    return f"event: {event}\ndata: {data}\n\n"
