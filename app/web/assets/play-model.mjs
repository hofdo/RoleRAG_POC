export const SESSION_DEFAULTS = Object.freeze({
  worldId: "demo_world",
  sceneId: "rose-gallery",
  personaId: "archivist",
  playerName: "",
});

export function createPlayState() {
  return {
    session: null,
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
    memoryWritten: turn.memory_written,
    warnings: turn.warnings,
  };
}

export function appendSuccessfulTurn(state, playerText, turn) {
  return {
    ...state,
    transcript: [
      ...state.transcript,
      { role: "player", text: playerText },
      { role: "assistant", text: turn.text },
    ],
  };
}
