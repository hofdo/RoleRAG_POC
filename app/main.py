from app import __version__


def get_app_metadata() -> dict[str, str]:
    return {"name": "rolerag-poc", "version": __version__}
