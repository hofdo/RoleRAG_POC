from __future__ import annotations

import json

from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

from app.api.schemas import (
    RouteResponse,
    StreamFailurePayload,
    StreamFinalPayload,
    StreamTextPayload,
    to_retrieval_diagnostics_response,
)
from app.domain import TurnOutcome, TurnResult


def build_turn_stream_frames(result: TurnResult) -> list[str]:
    route = RouteResponse(
        provider=result.route.provider.value,
        model=result.route.model,
        reason=result.route.reason,
    )
    retrieval = to_retrieval_diagnostics_response(result.retrieval)
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
                ),
            )
        ]
    return [
        _serialize_frame("text", StreamTextPayload(text=result.text)),
        _serialize_frame(
            "final",
            StreamFinalPayload(
                route=route,
                finish_reason=result.finish_reason,
                memory_written=result.memory_written,
                critic_status=result.critic_status.value,
                warnings=result.warnings,
                retrieval=retrieval,
            ),
        ),
    ]


def _serialize_frame(event: str, payload: BaseModel) -> str:
    data = json.dumps(jsonable_encoder(payload), separators=(",", ":"))
    return f"event: {event}\ndata: {data}\n\n"
