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
