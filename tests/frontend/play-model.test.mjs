import assert from "node:assert/strict";
import test from "node:test";

import {
  appendSuccessfulTurn,
  buildDebugState,
  buildSessionRequest,
  buildTurnRequest,
  createPlayState,
} from "../../app/web/assets/play-model.mjs";

test("session form request preserves editable defaults and player name", () => {
  assert.deepEqual(buildSessionRequest({
    worldId: "demo_world",
    sceneId: "rose-gallery",
    personaId: "archivist",
    playerName: "Avery",
  }), {
    world_id: "demo_world",
    scene_id: "rose-gallery",
    active_persona_id: "archivist",
    player_name: "Avery",
  });
});

test("turn request contains only message and request_cloud", () => {
  assert.deepEqual(buildTurnRequest("I listen.", true), {
    message: "I listen.",
    request_cloud: true,
  });
});

test("transcript appends player and assistant messages atomically after success", () => {
  const state = createPlayState();

  assert.deepEqual(state.transcript, []);
  const next = appendSuccessfulTurn(state, "I listen.", {
    text: "The gallery is quiet.",
    route: { provider: "local", model: "local-model", reason: "default local route" },
    memory_written: false,
    warnings: [],
  });

  assert.deepEqual(state.transcript, []);
  assert.deepEqual(next.transcript, [
    { role: "player", text: "I listen." },
    { role: "assistant", text: "The gallery is quiet." },
  ]);
});

test("debug state includes route metadata, warnings, memory, transport, and cloud request", () => {
  assert.deepEqual(buildDebugState({
    sessionId: "session-1",
    transport: "buffered-sse",
    requestCloud: true,
    turn: {
      text: "The gallery is quiet.",
      route: { provider: "local", model: "local-model", reason: "default local route" },
      memory_written: true,
      warnings: ["index delayed"],
    },
  }), {
    sessionId: "session-1",
    transport: "buffered-sse",
    requestCloud: true,
    routeProvider: "local",
    routeModel: "local-model",
    routeReason: "default local route",
    memoryWritten: true,
    warnings: ["index delayed"],
  });
});
