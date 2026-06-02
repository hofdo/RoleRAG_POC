from __future__ import annotations

import json

from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

from app.api.schemas import (
    RouteResponse,
    StreamFailurePayload,
    StreamFinalPayload,
    StreamTextPayload,
)
from app.domain import TurnOutcome, TurnResult


def build_turn_stream_frames(result: TurnResult) -> list[str]:
    route = RouteResponse(
        provider=result.route.provider.value,
        model=result.route.model,
        reason=result.route.reason,
    )
    if result.outcome == TurnOutcome.CONTROLLED_FAILURE:
        return [
            _serialize_frame(
                "failure",
                StreamFailurePayload(
                    text=result.text,
                    route=route,
                    memory_written=result.memory_written,
                    warnings=result.warnings,
                ),
            )
        ]
    return [
        _serialize_frame("text", StreamTextPayload(text=result.text)),
        _serialize_frame(
            "final",
            StreamFinalPayload(
                route=route,
                memory_written=result.memory_written,
                warnings=result.warnings,
            ),
        ),
    ]


def _serialize_frame(event: str, payload: BaseModel) -> str:
    data = json.dumps(jsonable_encoder(payload), separators=(",", ":"))
    return f"event: {event}\ndata: {data}\n\n"
