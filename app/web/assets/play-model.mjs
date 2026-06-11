export const SESSION_DEFAULTS = Object.freeze({
  worldId: "demo_world",
  sceneId: "rose-gallery",
  personaId: "archivist",
  playerName: "",
});

export function createPlayState() {
  return {
    session: null,
    sessionSource: null,
    transcript: [],
    debug: null,
  };
}

export function buildSessionRequest(form) {
  return {
    world_id: form.worldId,
    scene_id: form.sceneId,
    active_persona_id: form.personaId,
    player_name: form.playerName,
  };
}

export function createCatalogSelection(catalog, selectedWorldId = undefined) {
  const worlds = catalog.worlds;
  const scenesById = new Map(catalog.scenes.map((scene) => [scene.id, scene]));
  const personasById = new Map(catalog.personas.map((persona) => [persona.id, persona]));
  const world =
    worlds.find((candidate) => candidate.id === selectedWorldId) ??
    worlds[0] ??
    null;
  if (!world) {
    return {
      world: null,
      scenes: [],
      personas: [],
      sceneId: "",
      personaId: "",
    };
  }
  const scenes = world.scene_ids.map((id) => scenesById.get(id)).filter(Boolean);
  const personas = world.persona_ids.map((id) => personasById.get(id)).filter(Boolean);
  const defaultScene = scenes.find((scene) => scene.id === world.default_scene_id) ?? scenes[0];
  return {
    world,
    scenes,
    personas,
    sceneId: defaultScene?.id ?? "",
    personaId: personas[0]?.id ?? "",
  };
}

export function buildCatalogSessionRequest(selection, playerName) {
  return buildSessionRequest({
    worldId: selection.world?.id ?? "",
    sceneId: selection.sceneId,
    personaId: selection.personaId,
    playerName,
  });
}

export function describeCatalogSetupStatus({
  catalogLoaded,
  catalogLoadFailed,
  selection,
  manualFallbackOpen,
}) {
  if (catalogLoadFailed) {
    return "Catalog failed to load. Manual fallback is open; session creation will use manual IDs.";
  }
  if (manualFallbackOpen) {
    return "Manual fallback is open; session creation will use manual IDs.";
  }
  if (!catalogLoaded) {
    return "Loading catalog selectors.";
  }
  if (!selection?.world) {
    return "Catalog loaded, but no world is available. Open manual fallback to enter IDs.";
  }

  const scene = selection.scenes.find((candidate) => candidate.id === selection.sceneId);
  const persona = selection.personas.find((candidate) => candidate.id === selection.personaId);
  const sceneName = scene ? `${scene.title} (${scene.location})` : selection.sceneId;
  const personaName = persona ? `${persona.name} (${persona.role})` : selection.personaId;
  return `Catalog selection: ${selection.world.name} / ${sceneName} / ${personaName}.`;
}

export const RUNTIME_STATUS_UNAVAILABLE_TEXT =
  "Runtime status unavailable; gameplay controls remain available.";

export const RECENT_SESSIONS_UNAVAILABLE_TEXT =
  "Recent sessions unavailable; use Session ID fallback.";

function yesNo(value) {
  return value ? "yes" : "no";
}

export function describeRuntimeStatus(status) {
  if (!status) {
    return {
      warning: RUNTIME_STATUS_UNAVAILABLE_TEXT,
      rows: [],
    };
  }
  return {
    warning: "",
    rows: [
      ["App", `${status.app_name} ${status.app_version}`],
      ["Environment", status.environment],
      ["Cloud mode", status.cloud_mode],
      ["Retrieval", yesNo(status.retrieval_configured)],
      ["Catalog", yesNo(status.content_catalog_available)],
      ["Local route", yesNo(status.local_provider_configured)],
      ["Cloud route", yesNo(status.cloud_provider_configured)],
    ],
  };
}

export function describeRecentSessionsStatus({
  recentSessionsLoaded,
  recentSessionsLoadFailed,
  recentSessions,
}) {
  if (recentSessionsLoadFailed) {
    return RECENT_SESSIONS_UNAVAILABLE_TEXT;
  }
  if (!recentSessionsLoaded) {
    return "Loading recent sessions.";
  }
  if (recentSessions.length === 0) {
    return "No recent sessions yet.";
  }
  return `${recentSessions.length} recent session${recentSessions.length === 1 ? "" : "s"} available.`;
}

export function formatRecentSessionOption(session) {
  const updated = new Date(session.updated_at);
  const updatedLabel = Number.isNaN(updated.getTime())
    ? session.updated_at
    : updated.toLocaleString();
  return [
    session.player_name,
    session.world_id,
    session.active_scene_id,
    session.active_persona_id,
    updatedLabel,
  ].join(" / ");
}

export function selectedRecentSessionId(selectedSessionId) {
  return selectedSessionId;
}

export function buildTurnRequest(message, requestCloud) {
  return {
    message,
    request_cloud: requestCloud,
  };
}

export function buildDebugState({ sessionId, transport, requestCloud, turn }) {
  return {
    sessionId,
    transport,
    requestCloud,
    routeProvider: turn.route.provider,
    routeModel: turn.route.model,
    routeReason: turn.route.reason,
    finishReason: turn.finish_reason ?? null,
    memoryWritten: turn.memory_written,
    criticStatus: turn.critic_status,
    warnings: turn.warnings,
  };
}

export function startNewSession(sessionResponse) {
  return {
    ...createPlayState(),
    session: sessionResponse,
    sessionSource: "new",
  };
}

export function resumeSession(sessionResponse) {
  const recentTurns = [...sessionResponse.recent_turns].sort(
    (left, right) => left.turn_index - right.turn_index,
  );
  return {
    ...createPlayState(),
    session: sessionResponse,
    sessionSource: "resumed",
    transcript: recentTurns.flatMap((turn) => {
      const label = `Resumed turn #${turn.turn_index}`;
      return [
        { role: "player", text: turn.user_message, label, source: "resumed" },
        { role: "assistant", text: turn.assistant_message, label, source: "resumed" },
      ];
    }),
  };
}

export function appendSuccessfulTurn(state, playerText, turn) {
  return {
    ...state,
    transcript: [
      ...state.transcript,
      { role: "player", text: playerText, source: "new" },
      { role: "assistant", text: turn.text, source: "new" },
    ],
  };
}
