from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api import router as api_router
from app.api.errors import ApiError, api_error_handler, request_validation_error_handler
from app.web import ASSETS_DIRECTORY
from app.web import router as web_router


def get_app_metadata() -> dict[str, str]:
    return {"name": "rolerag-poc", "version": __version__}


app = FastAPI(title="rolerag-poc", version=__version__)
app.add_exception_handler(ApiError, api_error_handler)
app.add_exception_handler(RequestValidationError, request_validation_error_handler)
app.include_router(api_router)
app.include_router(web_router)
app.mount("/play/assets", StaticFiles(directory=ASSETS_DIRECTORY), name="play-assets")
