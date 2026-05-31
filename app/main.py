from fastapi import FastAPI

from app import __version__
from app.api import router as api_router


def get_app_metadata() -> dict[str, str]:
    return {"name": "rolerag-poc", "version": __version__}


app = FastAPI(title="rolerag-poc", version=__version__)
app.include_router(api_router)
