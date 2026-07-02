from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app


def test_fastapi_openapi_exposes_mvp_route_shape() -> None:
    response = TestClient(app).get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "rolerag-poc"
    assert set(schema["paths"]) >= {
        "/runtime/status",
        "/content/catalog",
        "/sessions",
        "/sessions/{session_id}",
        "/sessions/{session_id}/turns",
        "/sessions/{session_id}/turns/stream",
    }
    assert set(schema["paths"]["/runtime/status"]) == {"get"}
    assert set(schema["paths"]["/content/catalog"]) == {"get"}
    assert set(schema["paths"]["/sessions"]) == {"get", "post"}
    assert set(schema["paths"]["/sessions/{session_id}"]) == {"get"}
    assert set(schema["paths"]["/sessions/{session_id}/turns"]) == {"post"}
    assert set(schema["paths"]["/sessions/{session_id}/turns/stream"]) == {"post"}
    assert schema["components"]["schemas"]["CreateSessionRequest"]["required"] == [
        "world_id",
        "scene_id",
        "player_name",
        "active_persona_id",
    ]
    assert schema["components"]["schemas"]["CreateTurnRequest"]["required"] == ["message"]
    assert schema["components"]["schemas"]["ErrorResponse"]["required"] == ["error"]
    assert schema["components"]["schemas"]["ErrorBody"]["required"] == [
        "code",
        "message",
        "details",
    ]
    assert (
        schema["paths"]["/content/catalog"]["get"]["responses"]["400"]["content"][
            "application/json"
        ]["schema"]["$ref"]
        == "#/components/schemas/ErrorResponse"
    )
    assert (
        schema["paths"]["/sessions"]["post"]["responses"]["400"]["content"][
            "application/json"
        ]["schema"]["$ref"]
        == "#/components/schemas/ErrorResponse"
    )
    assert (
        schema["paths"]["/sessions"]["post"]["responses"]["422"]["content"][
            "application/json"
        ]["schema"]["$ref"]
        == "#/components/schemas/ErrorResponse"
    )
    assert (
        schema["paths"]["/sessions/{session_id}"]["get"]["responses"]["404"]["content"][
            "application/json"
        ]["schema"]["$ref"]
        == "#/components/schemas/ErrorResponse"
    )
    stream_responses = schema["paths"]["/sessions/{session_id}/turns/stream"]["post"][
        "responses"
    ]
    assert "text/event-stream" in stream_responses["200"]["content"]
    # Once streaming starts, errors travel as `event: error` SSE frames on HTTP 200
    # rather than HTTP error responses, so only pre-stream 422 validation remains.
    assert set(stream_responses) == {"200", "422"}
    assert (
        stream_responses["422"]["content"]["application/json"]["schema"]["$ref"]
        == "#/components/schemas/ErrorResponse"
    )


def test_spa_assets_are_served_with_javascript_mime_when_built() -> None:
    # Browsers refuse to execute module scripts without a JavaScript MIME type; some
    # environments lack a .js mapping and serve application/octet-stream, breaking the UI.
    spa_directory = Path(__file__).parents[2] / "frontend" / "dist" / "frontend" / "browser"
    if not spa_directory.is_dir():
        pytest.skip("SPA not built")
    client = TestClient(app)

    index = client.get("/app/")
    script = next(iter(sorted(spa_directory.glob("*.js"))), None)
    assert index.status_code == 200
    assert index.headers["content-type"].startswith("text/html")
    assert script is not None
    asset = client.get(f"/app/{script.name}")
    assert asset.status_code == 200
    assert "javascript" in asset.headers["content-type"]


def test_root_serves_spa_redirect_or_build_hint_outside_openapi() -> None:
    client = TestClient(app)

    response = client.get("/", follow_redirects=False)
    schema = client.get("/openapi.json").json()

    # With a frontend build present the root redirects into the SPA; without one it
    # returns an actionable 503 instead of a bare 404.
    if response.status_code == 307:
        assert response.headers["location"] == "/app/"
    else:
        assert response.status_code == 503
        assert "npm" in response.text
    assert "/" not in schema["paths"]
    assert "/app" not in schema["paths"]
