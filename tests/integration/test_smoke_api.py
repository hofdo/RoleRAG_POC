from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_fastapi_openapi_exposes_mvp_route_shape() -> None:
    response = TestClient(app).get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "rolerag-poc"
    assert set(schema["paths"]) >= {
        "/sessions",
        "/sessions/{session_id}",
        "/sessions/{session_id}/turns",
    }
    assert set(schema["paths"]["/sessions"]) == {"post"}
    assert set(schema["paths"]["/sessions/{session_id}"]) == {"get"}
    assert set(schema["paths"]["/sessions/{session_id}/turns"]) == {"post"}
    assert schema["components"]["schemas"]["CreateSessionRequest"]["required"] == [
        "world_id",
        "scene_id",
        "player_name",
        "active_persona_id",
    ]
    assert schema["components"]["schemas"]["CreateTurnRequest"]["required"] == ["message"]
