from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_fastapi_openapi_exposes_mvp_route_shape() -> None:
    response = TestClient(app).get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "rolerag-poc"
    assert set(schema["paths"]) >= {
        "/content/catalog",
        "/sessions",
        "/sessions/{session_id}",
        "/sessions/{session_id}/turns",
        "/sessions/{session_id}/turns/stream",
    }
    assert set(schema["paths"]["/content/catalog"]) == {"get"}
    assert set(schema["paths"]["/sessions"]) == {"post"}
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
    assert "text/event-stream" in schema["paths"]["/sessions/{session_id}/turns/stream"]["post"][
        "responses"
    ]["200"]["content"]
    for status_code in ("400", "404", "422"):
        assert (
            schema["paths"]["/sessions/{session_id}/turns/stream"]["post"]["responses"][
                status_code
            ]["content"]["application/json"]["schema"]["$ref"]
            == "#/components/schemas/ErrorResponse"
        )


def test_play_page_and_assets_are_served_outside_openapi() -> None:
    client = TestClient(app)

    page = client.get("/play")
    css = client.get("/play/assets/styles.css")
    api_client = client.get("/play/assets/api-client.mjs")
    play_model = client.get("/play/assets/play-model.mjs")
    play_ui = client.get("/play/assets/play-ui.mjs")
    schema = client.get("/openapi.json").json()

    assert page.status_code == 200
    assert '<script type="module" src="/play/assets/play-ui.mjs"></script>' in page.text
    assert 'id="resume-form"' in page.text
    assert 'id="resume-session-id"' in page.text
    assert "Resume session" in page.text
    assert css.status_code == 200
    assert api_client.status_code == 200
    assert play_model.status_code == 200
    assert play_ui.status_code == 200
    assert "/play" not in schema["paths"]
