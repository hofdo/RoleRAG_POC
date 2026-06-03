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
