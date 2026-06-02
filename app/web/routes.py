from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

WEB_DIRECTORY = Path(__file__).parent
ASSETS_DIRECTORY = WEB_DIRECTORY / "assets"

router = APIRouter()


@router.get("/play", include_in_schema=False, response_class=FileResponse)
def get_play_page() -> FileResponse:
    return FileResponse(WEB_DIRECTORY / "index.html")
