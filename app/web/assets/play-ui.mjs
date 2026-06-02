import { ApiError, createBufferedTurn, createSession, createTurn } from "./api-client.mjs";
import {
  appendSuccessfulTurn,
  buildDebugState,
  buildSessionRequest,
  buildTurnRequest,
  createPlayState,
} from "./play-model.mjs";

const elements = {
  error: document.querySelector("#error-message"),
  setupPanel: document.querySelector("#setup-panel"),
  playPanel: document.querySelector("#play-panel"),
  sessionForm: document.querySelector("#session-form"),
  worldId: document.querySelector("#world-id"),
  sceneId: document.querySelector("#scene-id"),
  personaId: document.querySelector("#persona-id"),
  playerName: document.querySelector("#player-name"),
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

function showError(error) {
  const prefix = error instanceof ApiError ? `${error.code}: ` : "";
  elements.error.textContent = `${prefix}${error.message}`;
  elements.error.hidden = false;
}

function clearError() {
  elements.error.textContent = "";
  elements.error.hidden = true;
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
    item.append(role, text);
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
    state = {
      ...createPlayState(),
      session: await createSession(buildSessionRequest({
        worldId: elements.worldId.value,
        sceneId: elements.sceneId.value,
        personaId: elements.personaId.value,
        playerName: elements.playerName.value,
      })),
    };
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
  elements.requestCloud.checked = false;
  elements.useStream.checked = false;
  renderSession();
  renderTranscript();
  renderDebugState();
  elements.playerName.focus();
});
