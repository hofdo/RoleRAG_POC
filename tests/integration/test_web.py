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
