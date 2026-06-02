from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from app import __version__
from app.api import router as api_router
from app.api.errors import ApiError, api_error_handler, request_validation_error_handler


def get_app_metadata() -> dict[str, str]:
    return {"name": "rolerag-poc", "version": __version__}


app = FastAPI(title="rolerag-poc", version=__version__)
app.add_exception_handler(ApiError, api_error_handler)
app.add_exception_handler(RequestValidationError, request_validation_error_handler)
app.include_router(api_router)
