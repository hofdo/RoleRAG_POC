from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_get_play_serves_the_html_page() -> None:
    client = TestClient(app)

    response = client.get("/play")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    body = response.text
    assert "<!doctype html>" in body.lower()
    assert "RoleRAG Play" in body


def test_play_asset_is_served() -> None:
    client = TestClient(app)

    response = client.get("/play/assets/play-ui.mjs")

    assert response.status_code == 200
    assert response.text  # served a non-empty module


def test_missing_play_asset_returns_404() -> None:
    client = TestClient(app)

    response = client.get("/play/assets/does-not-exist.mjs")

    assert response.status_code == 404


def test_play_module_asset_is_served_with_javascript_mime() -> None:
    # Browsers refuse to execute <script type="module"> unless the MIME is a JavaScript type;
    # some environments lack a .mjs mapping and serve application/octet-stream, breaking the UI.
    client = TestClient(app)

    response = client.get("/play/assets/play-ui.mjs")

    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]


def test_play_stylesheet_is_served_with_css_mime() -> None:
    client = TestClient(app)

    response = client.get("/play/assets/styles.css")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/css")


def test_play_asset_path_traversal_does_not_escape_the_assets_directory() -> None:
    # A traversal attempt must not serve files outside app/web/assets (e.g. the app source).
    client = TestClient(app)

    response = client.get("/play/assets/../routes.py")

    assert response.status_code != 200
    assert "get_play_page" not in response.text
