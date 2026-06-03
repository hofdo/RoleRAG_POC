import {
  ApiError,
  createBufferedTurn,
  createSession,
  createTurn,
  getContentCatalog,
  getSession,
} from "./api-client.mjs";
import {
  appendSuccessfulTurn,
  buildDebugState,
  buildCatalogSessionRequest,
  buildSessionRequest,
  buildTurnRequest,
  createCatalogSelection,
  createPlayState,
  describeCatalogSetupStatus,
  resumeSession,
  startNewSession,
} from "./play-model.mjs";

const elements = {
  error: document.querySelector("#error-message"),
  setupPanel: document.querySelector("#setup-panel"),
  playPanel: document.querySelector("#play-panel"),
  sessionForm: document.querySelector("#session-form"),
  resumeForm: document.querySelector("#resume-form"),
  catalogWorld: document.querySelector("#catalog-world"),
  catalogScene: document.querySelector("#catalog-scene"),
  catalogPersona: document.querySelector("#catalog-persona"),
  setupStatus: document.querySelector("#setup-status"),
  manualSessionPanel: document.querySelector("#manual-session-panel"),
  worldId: document.querySelector("#world-id"),
  sceneId: document.querySelector("#scene-id"),
  personaId: document.querySelector("#persona-id"),
  playerName: document.querySelector("#player-name"),
  resumeSessionId: document.querySelector("#resume-session-id"),
  sessionSummary: document.querySelector("#session-summary"),
  newSession: document.querySelector("#new-session"),
  turnForm: document.querySelector("#turn-form"),
  turnMessage: document.querySelector("#turn-message"),
  requestCloud: document.querySelector("#request-cloud"),
  useStream: document.querySelector("#use-stream"),
  sendTurn: document.querySelector("#send-turn"),
  transcript: document.querySelector("#transcript"),
  debugState: document.querySelector("#debug-state"),
};

let state = createPlayState();
let contentCatalog = null;
let catalogSelection = null;
let catalogLoadFailed = false;

function showError(error) {
  const prefix = error instanceof ApiError ? `${error.code}: ` : "";
  elements.error.textContent = `${prefix}${error.message}`;
  elements.error.hidden = false;
}

function clearError() {
  elements.error.textContent = "";
  elements.error.hidden = true;
}

function setSelectOptions(select, options, selectedId, labelForOption) {
  select.replaceChildren();
  for (const optionItem of options) {
    const option = document.createElement("option");
    option.value = optionItem.id;
    option.textContent = labelForOption(optionItem);
    option.selected = optionItem.id === selectedId;
    select.append(option);
  }
}

function syncManualIds() {
  if (!catalogSelection?.world) {
    return;
  }
  elements.worldId.value = catalogSelection.world.id;
  elements.sceneId.value = catalogSelection.sceneId;
  elements.personaId.value = catalogSelection.personaId;
}

function renderSetupStatus() {
  elements.setupStatus.textContent = describeCatalogSetupStatus({
    catalogLoaded: Boolean(contentCatalog),
    catalogLoadFailed,
    selection: catalogSelection,
    manualFallbackOpen: elements.manualSessionPanel.open,
  });
}

function renderCatalogSelection() {
  if (!contentCatalog || !catalogSelection) {
    elements.catalogWorld.disabled = true;
    elements.catalogScene.disabled = true;
    elements.catalogPersona.disabled = true;
    renderSetupStatus();
    return;
  }
  setSelectOptions(
    elements.catalogWorld,
    contentCatalog.worlds,
    catalogSelection.world?.id,
    (world) => world.name,
  );
  setSelectOptions(
    elements.catalogScene,
    catalogSelection.scenes,
    catalogSelection.sceneId,
    (scene) => `${scene.title} (${scene.location})`,
  );
  setSelectOptions(
    elements.catalogPersona,
    catalogSelection.personas,
    catalogSelection.personaId,
    (persona) => `${persona.name} (${persona.role})`,
  );
  elements.catalogWorld.disabled = false;
  elements.catalogScene.disabled = catalogSelection.scenes.length === 0;
  elements.catalogPersona.disabled = catalogSelection.personas.length === 0;
  syncManualIds();
  renderSetupStatus();
}

async function loadCatalog() {
  try {
    contentCatalog = await getContentCatalog();
    catalogSelection = createCatalogSelection(contentCatalog);
    catalogLoadFailed = false;
    renderCatalogSelection();
  } catch (error) {
    contentCatalog = null;
    catalogSelection = null;
    catalogLoadFailed = true;
    renderCatalogSelection();
    elements.manualSessionPanel.open = true;
    renderSetupStatus();
    showError(error);
  }
}

function renderSession() {
  const session = state.session;
  elements.setupPanel.hidden = Boolean(session);
  elements.playPanel.hidden = !session;
  if (!session) {
    return;
  }
  elements.sessionSummary.textContent =
    `${session.world_id} / ${session.active_scene_id} / ${session.active_persona_id}`;
}

function renderTranscript() {
  elements.transcript.replaceChildren();
  for (const entry of state.transcript) {
    const item = document.createElement("li");
    item.className = `transcript-entry ${entry.role}`;
    const role = document.createElement("strong");
    role.textContent = entry.role === "player" ? "You" : "Assistant";
    const text = document.createElement("p");
    text.textContent = entry.text;
    item.append(role);
    if (entry.label) {
      const label = document.createElement("span");
      label.className = "transcript-label";
      label.textContent = entry.label;
      item.append(label);
    }
    item.append(text);
    elements.transcript.append(item);
  }
}

function addDebugRow(label, value) {
  const term = document.createElement("dt");
  term.textContent = label;
  const description = document.createElement("dd");
  description.textContent = String(value);
  elements.debugState.append(term, description);
}

function renderDebugState() {
  elements.debugState.replaceChildren();
  if (!state.debug) {
    return;
  }
  addDebugRow("Session ID", state.debug.sessionId);
  addDebugRow("Transport", state.debug.transport);
  addDebugRow("request_cloud", state.debug.requestCloud);
  addDebugRow("Route provider", state.debug.routeProvider);
  addDebugRow("Route model", state.debug.routeModel);
  addDebugRow("Route reason", state.debug.routeReason);
  addDebugRow("memory_written", state.debug.memoryWritten);
  addDebugRow("Warnings", state.debug.warnings.length ? state.debug.warnings.join("; ") : "none");
}

elements.sessionForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();
  const submit = elements.sessionForm.querySelector("button[type='submit']");
  submit.disabled = true;
  try {
    const useManualIds = elements.manualSessionPanel.open || !catalogSelection?.world;
    const request = useManualIds
      ? buildSessionRequest({
        worldId: elements.worldId.value,
        sceneId: elements.sceneId.value,
        personaId: elements.personaId.value,
        playerName: elements.playerName.value,
      })
      : buildCatalogSessionRequest(catalogSelection, elements.playerName.value);
    state = startNewSession(
      await createSession(request),
    );
    renderSession();
    renderTranscript();
    renderDebugState();
    elements.turnMessage.focus();
  } catch (error) {
    showError(error);
  } finally {
    submit.disabled = false;
  }
});

elements.catalogWorld.addEventListener("change", () => {
  if (!contentCatalog) {
    return;
  }
  catalogSelection = createCatalogSelection(contentCatalog, elements.catalogWorld.value);
  renderCatalogSelection();
});

elements.catalogScene.addEventListener("change", () => {
  if (!catalogSelection) {
    return;
  }
  catalogSelection = { ...catalogSelection, sceneId: elements.catalogScene.value };
  syncManualIds();
  renderSetupStatus();
});

elements.catalogPersona.addEventListener("change", () => {
  if (!catalogSelection) {
    return;
  }
  catalogSelection = { ...catalogSelection, personaId: elements.catalogPersona.value };
  syncManualIds();
  renderSetupStatus();
});

elements.manualSessionPanel.addEventListener("toggle", () => {
  renderSetupStatus();
});

elements.resumeForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();
  const submit = elements.resumeForm.querySelector("button[type='submit']");
  submit.disabled = true;
  try {
    state = resumeSession(await getSession(elements.resumeSessionId.value));
    renderSession();
    renderTranscript();
    renderDebugState();
    elements.turnMessage.focus();
  } catch (error) {
    showError(error);
  } finally {
    submit.disabled = false;
  }
});

elements.turnForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();
  const session = state.session;
  if (!session) {
    return;
  }
  const message = elements.turnMessage.value;
  const requestCloud = elements.requestCloud.checked;
  const stream = elements.useStream.checked;
  elements.sendTurn.disabled = true;
  try {
    const turnRequest = buildTurnRequest(message, requestCloud);
    const turn = stream
      ? await createBufferedTurn(session.session_id, turnRequest)
      : await createTurn(session.session_id, turnRequest);
    state = {
      ...appendSuccessfulTurn(state, message, turn),
      debug: buildDebugState({
        sessionId: session.session_id,
        transport: stream ? "buffered-sse" : "json",
        requestCloud,
        turn,
      }),
    };
    elements.turnMessage.value = "";
    renderTranscript();
    renderDebugState();
    elements.turnMessage.focus();
  } catch (error) {
    showError(error);
  } finally {
    elements.sendTurn.disabled = false;
  }
});

elements.newSession.addEventListener("click", () => {
  clearError();
  state = createPlayState();
  elements.turnMessage.value = "";
  elements.resumeSessionId.value = "";
  elements.requestCloud.checked = false;
  elements.useStream.checked = false;
  if (contentCatalog) {
    catalogSelection = createCatalogSelection(contentCatalog);
    renderCatalogSelection();
  } else {
    renderSetupStatus();
  }
  renderSession();
  renderTranscript();
  renderDebugState();
  elements.playerName.focus();
});

renderCatalogSelection();
void loadCatalog();
