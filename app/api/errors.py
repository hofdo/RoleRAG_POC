from __future__ import annotations

from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.schemas import ErrorBody, ErrorDetail, ErrorResponse


class ApiError(Exception):
    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        details: list[ErrorDetail] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or []


def api_error_handler(_: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, ApiError):
        raise exc
    return _error_response(
        status_code=exc.status_code,
        code=exc.code,
        message=exc.message,
        details=exc.details,
    )


def request_validation_error_handler(_: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, RequestValidationError):
        raise exc
    return _error_response(
        status_code=422,
        code="validation_error",
        message="Request validation failed",
        details=[
            ErrorDetail(
                loc=[_sanitize_error_location(segment) for segment in error["loc"]],
                type=str(error["type"]),
                message="Request field validation failed",
            )
            for error in exc.errors()
        ],
    )


def _error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    details: list[ErrorDetail],
) -> JSONResponse:
    payload = ErrorResponse(
        error=ErrorBody(
            code=code,
            message=message,
            details=details,
        )
    )
    return JSONResponse(status_code=status_code, content=jsonable_encoder(payload))


def _sanitize_error_location(segment: str | int) -> str | int:
    if isinstance(segment, str) and ("/" in segment or "\\" in segment):
        return "<field>"
    return segment
