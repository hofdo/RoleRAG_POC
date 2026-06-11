import assert from "node:assert/strict";
import test from "node:test";

import {
  appendSuccessfulTurn,
  buildDebugState,
  buildCatalogSessionRequest,
  buildSessionRequest,
  buildTurnRequest,
  createCatalogSelection,
  createPlayState,
  describeCatalogSetupStatus,
  describeRecentSessionsStatus,
  describeRuntimeStatus,
  formatRecentSessionOption,
  RECENT_SESSIONS_UNAVAILABLE_TEXT,
  RUNTIME_STATUS_UNAVAILABLE_TEXT,
  resumeSession,
  selectedRecentSessionId,
  startNewSession,
} from "../../app/web/assets/play-model.mjs";

const CATALOG = Object.freeze({
  worlds: [
    {
      id: "demo_world",
      name: "Winter Palace Intrigue",
      default_scene_id: "rose-gallery",
      scene_ids: ["rose-gallery", "armory"],
      persona_ids: ["archivist", "guard"],
    },
    {
      id: "fallback_world",
      name: "Fallback",
      default_scene_id: "missing-scene",
      scene_ids: ["armory"],
      persona_ids: ["missing-persona", "guard"],
    },
  ],
  scenes: [
    {
      id: "armory",
      title: "Armory",
      location: "Keep",
      player_visible_summary: "Steel lines the walls.",
      active_personas: ["guard"],
    },
    {
      id: "rose-gallery",
      title: "Rose Gallery",
      location: "Palace",
      player_visible_summary: "Roses climb mirrored walls.",
      active_personas: ["archivist"],
    },
  ],
  personas: [
    {
      id: "archivist",
      name: "Iria Vale",
      role: "npc",
      public_description: "A composed archivist.",
      speaking_style: "Precise.",
    },
    {
      id: "guard",
      name: "Gate Guard",
      role: "npc",
      public_description: "A watchful guard.",
      speaking_style: "Blunt.",
    },
  ],
});

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

test("catalog selection filters scenes and personas through the selected world", () => {
  const selection = createCatalogSelection(CATALOG, "demo_world");

  assert.equal(selection.world.id, "demo_world");
  assert.deepEqual(selection.scenes.map((scene) => scene.id), ["rose-gallery", "armory"]);
  assert.deepEqual(selection.personas.map((persona) => persona.id), ["archivist", "guard"]);
  assert.equal(selection.sceneId, "rose-gallery");
  assert.equal(selection.personaId, "archivist");
});

test("catalog selection falls back to first valid referenced scene and persona", () => {
  const selection = createCatalogSelection(CATALOG, "fallback_world");

  assert.equal(selection.world.id, "fallback_world");
  assert.deepEqual(selection.scenes.map((scene) => scene.id), ["armory"]);
  assert.deepEqual(selection.personas.map((persona) => persona.id), ["guard"]);
  assert.equal(selection.sceneId, "armory");
  assert.equal(selection.personaId, "guard");
});

test("catalog session request uses selected catalog IDs", () => {
  const selection = createCatalogSelection(CATALOG, "demo_world");

  assert.deepEqual(buildCatalogSessionRequest(selection, "Avery"), {
    world_id: "demo_world",
    scene_id: "rose-gallery",
    active_persona_id: "archivist",
    player_name: "Avery",
  });
});

test("catalog setup status shows selected catalog display names", () => {
  const selection = createCatalogSelection(CATALOG, "demo_world");

  assert.equal(describeCatalogSetupStatus({
    catalogLoaded: true,
    catalogLoadFailed: false,
    selection,
    manualFallbackOpen: false,
  }), "Catalog selection: Winter Palace Intrigue / Rose Gallery (Palace) / Iria Vale (npc).");
});

test("catalog setup status explains manual fallback behavior", () => {
  const selection = createCatalogSelection(CATALOG, "demo_world");

  assert.equal(describeCatalogSetupStatus({
    catalogLoaded: true,
    catalogLoadFailed: false,
    selection,
    manualFallbackOpen: true,
  }), "Manual fallback is open; session creation will use manual IDs.");
});

test("catalog setup status explains catalog load failure", () => {
  assert.equal(describeCatalogSetupStatus({
    catalogLoaded: false,
    catalogLoadFailed: true,
    selection: null,
    manualFallbackOpen: true,
  }), "Catalog failed to load. Manual fallback is open; session creation will use manual IDs.");
});

test("runtime status description formats healthy shallow metadata", () => {
  assert.deepEqual(describeRuntimeStatus({
    app_name: "rolerag-poc",
    app_version: "0.1.0",
    environment: "local",
    cloud_mode: "ask",
    retrieval_configured: true,
    content_catalog_available: true,
    local_provider_configured: true,
    cloud_provider_configured: false,
  }), {
    warning: "",
    rows: [
      ["App", "rolerag-poc 0.1.0"],
      ["Environment", "local"],
      ["Cloud mode", "ask"],
      ["Retrieval", "yes"],
      ["Catalog", "yes"],
      ["Local route", "yes"],
      ["Cloud route", "no"],
    ],
  });
});

test("runtime status description gives non-blocking load failure text", () => {
  assert.deepEqual(describeRuntimeStatus(null), {
    warning: RUNTIME_STATUS_UNAVAILABLE_TEXT,
    rows: [],
  });
});

test("recent sessions status formats loading, empty, populated, and failure states", () => {
  assert.equal(describeRecentSessionsStatus({
    recentSessionsLoaded: false,
    recentSessionsLoadFailed: false,
    recentSessions: [],
  }), "Loading recent sessions.");
  assert.equal(describeRecentSessionsStatus({
    recentSessionsLoaded: true,
    recentSessionsLoadFailed: false,
    recentSessions: [],
  }), "No recent sessions yet.");
  assert.equal(describeRecentSessionsStatus({
    recentSessionsLoaded: true,
    recentSessionsLoadFailed: false,
    recentSessions: [{ session_id: "session-1" }, { session_id: "session-2" }],
  }), "2 recent sessions available.");
  assert.equal(describeRecentSessionsStatus({
    recentSessionsLoaded: false,
    recentSessionsLoadFailed: true,
    recentSessions: [],
  }), RECENT_SESSIONS_UNAVAILABLE_TEXT);
});

test("recent session option includes safe visible identifiers and updated time", () => {
  const label = formatRecentSessionOption({
    session_id: "session-1",
    world_id: "demo_world",
    active_scene_id: "rose-gallery",
    active_persona_id: "archivist",
    player_name: "Avery",
    created_at: "2026-06-03T10:00:00Z",
    updated_at: "2026-06-03T10:05:00Z",
  });

  assert.match(label, /Avery/);
  assert.match(label, /demo_world/);
  assert.match(label, /rose-gallery/);
  assert.match(label, /archivist/);
});

test("recent session selection returns the chosen session id", () => {
  assert.equal(selectedRecentSessionId("session-1"), "session-1");
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
    { role: "player", text: "I listen.", source: "new" },
    { role: "assistant", text: "The gallery is quiet.", source: "new" },
  ]);
});

test("new session state starts with an empty transcript", () => {
  const state = startNewSession({
    session_id: "session-1",
    world_id: "demo_world",
    active_scene_id: "rose-gallery",
    active_persona_id: "archivist",
  });

  assert.equal(state.session.session_id, "session-1");
  assert.deepEqual(state.transcript, []);
  assert.equal(state.sessionSource, "new");
});

test("resume state converts recent turns into ordered transcript entries", () => {
  const state = resumeSession({
    session_id: "session-1",
    world_id: "demo_world",
    active_scene_id: "rose-gallery",
    active_persona_id: "archivist",
    recent_turns: [
      {
        turn_index: 2,
        user_message: "What do I hear?",
        assistant_message: "Rain at the glass.",
        created_at: "2026-06-03T10:01:00Z",
      },
      {
        turn_index: 1,
        user_message: "I listen.",
        assistant_message: "The gallery is quiet.",
        created_at: "2026-06-03T10:00:00Z",
      },
    ],
  });

  assert.equal(state.sessionSource, "resumed");
  assert.deepEqual(state.transcript, [
    { role: "player", text: "I listen.", label: "Resumed turn #1", source: "resumed" },
    { role: "assistant", text: "The gallery is quiet.", label: "Resumed turn #1", source: "resumed" },
    { role: "player", text: "What do I hear?", label: "Resumed turn #2", source: "resumed" },
    { role: "assistant", text: "Rain at the glass.", label: "Resumed turn #2", source: "resumed" },
  ]);
});

test("new appended turns stay distinguishable and do not mutate resumed state", () => {
  const resumed = resumeSession({
    session_id: "session-1",
    world_id: "demo_world",
    active_scene_id: "rose-gallery",
    active_persona_id: "archivist",
    recent_turns: [
      {
        turn_index: 3,
        user_message: "I enter.",
        assistant_message: "The room waits.",
        created_at: "2026-06-03T10:00:00Z",
      },
    ],
  });

  const next = appendSuccessfulTurn(resumed, "I bow.", {
    text: "The archivist nods.",
    route: { provider: "local", model: "local-model", reason: "default local route" },
    memory_written: false,
    warnings: [],
  });

  assert.deepEqual(resumed.transcript, [
    { role: "player", text: "I enter.", label: "Resumed turn #3", source: "resumed" },
    { role: "assistant", text: "The room waits.", label: "Resumed turn #3", source: "resumed" },
  ]);
  assert.deepEqual(next.transcript, [
    { role: "player", text: "I enter.", label: "Resumed turn #3", source: "resumed" },
    { role: "assistant", text: "The room waits.", label: "Resumed turn #3", source: "resumed" },
    { role: "player", text: "I bow.", source: "new" },
    { role: "assistant", text: "The archivist nods.", source: "new" },
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
      finish_reason: "stop",
      memory_written: true,
      critic_status: "repaired",
      warnings: ["index delayed"],
    },
  }), {
    sessionId: "session-1",
    transport: "buffered-sse",
    requestCloud: true,
    routeProvider: "local",
    routeModel: "local-model",
    routeReason: "default local route",
    finishReason: "stop",
    memoryWritten: true,
    criticStatus: "repaired",
    warnings: ["index delayed"],
  });
});
