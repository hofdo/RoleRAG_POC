from pathlib import Path

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response
from starlette.types import Scope

from app import __version__
from app.api import router as api_router
from app.api.errors import ApiError, api_error_handler, request_validation_error_handler
from app.web import ASSETS_DIRECTORY
from app.web import router as web_router

# Angular SPA build output, mounted at /app when present (built via
# `ng build --base-href=/app/`). Guarded by isdir so a checkout without a
# frontend build still boots.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SPA_DIRECTORY = _REPO_ROOT / "frontend" / "dist" / "frontend" / "browser"


class _SpaStaticFiles(StaticFiles):
    """Serve index.html for unknown nested paths so client-side routes (e.g.
    /app/inspector) survive a hard refresh instead of 404-ing."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404:
                return await super().get_response("index.html", scope)
            raise


def get_app_metadata() -> dict[str, str]:
    return {"name": "rolerag-poc", "version": __version__}


app = FastAPI(title="rolerag-poc", version=__version__)
app.add_exception_handler(ApiError, api_error_handler)
app.add_exception_handler(RequestValidationError, request_validation_error_handler)
app.include_router(api_router)
app.include_router(web_router)
app.mount("/play/assets", StaticFiles(directory=ASSETS_DIRECTORY), name="play-assets")


@app.get("/", include_in_schema=False)
def root_redirect() -> RedirectResponse:
    # The SPA is the default UI when built; fall back to the legacy /play page otherwise.
    return RedirectResponse("/app/" if _SPA_DIRECTORY.is_dir() else "/play")

if _SPA_DIRECTORY.is_dir():
    # html=True serves index.html at /app/; _SpaStaticFiles adds the deep-link fallback so a
    # refresh on /app/inspector (a client-side route) returns index.html instead of 404.
    app.mount("/app", _SpaStaticFiles(directory=_SPA_DIRECTORY, html=True), name="spa")
