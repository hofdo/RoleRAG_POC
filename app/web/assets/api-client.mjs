/**
 * @typedef {{
 *   world_id: string,
 *   scene_id: string,
 *   player_name: string,
 *   active_persona_id: string,
 * }} CreateSessionRequest
 */

/**
 * @typedef {{
 *   session_id: string,
 *   world_id: string,
 *   active_scene_id: string,
 *   active_persona_id: string,
 * }} CreateSessionResponse
 */

/**
 * @typedef {{
 *   id: string,
 *   name: string,
 *   default_scene_id: string,
 *   scene_ids: string[],
 *   persona_ids: string[],
 * }} CatalogWorld
 */

/**
 * @typedef {{
 *   id: string,
 *   title: string,
 *   location: string,
 *   player_visible_summary: string,
 *   active_personas: string[],
 * }} CatalogScene
 */

/**
 * @typedef {{
 *   id: string,
 *   name: string,
 *   role: string,
 *   public_description: string,
 *   speaking_style: string,
 * }} CatalogPersona
 */

/**
 * @typedef {{
 *   worlds: CatalogWorld[],
 *   scenes: CatalogScene[],
 *   personas: CatalogPersona[],
 * }} ContentCatalog
 */

/** @typedef {{ message: string, request_cloud: boolean }} CreateTurnRequest */
/** @typedef {{ provider: string, model: string, reason: string }} RouteResponse */

/**
 * @typedef {{
 *   turn_index: number,
 *   user_message: string,
 *   assistant_message: string,
 *   created_at: string,
 * }} RecentTurnResponse
 */

/**
 * @typedef {{
 *   session_id: string,
 *   world_id: string,
 *   active_scene_id: string,
 *   active_persona_id: string,
 *   recent_turns: RecentTurnResponse[],
 * }} GetSessionResponse
 */

/**
 * @typedef {{
 *   text: string,
 *   route: RouteResponse,
 *   memory_written: boolean,
 *   warnings: string[],
 * }} CreateTurnResponse
 */

export class ApiError extends Error {
  constructor(code, message, status) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
  }
}

async function throwApiError(response) {
  let body;
  try {
    body = await response.json();
  } catch {
    throw new ApiError("request_failed", "The backend request failed.", response.status);
  }
  const error = body?.error;
  throw new ApiError(
    typeof error?.code === "string" ? error.code : "request_failed",
    typeof error?.message === "string" ? error.message : "The backend request failed.",
    response.status,
  );
}

async function requestJson(url, { method = "GET", body = undefined } = {}, fetchImpl) {
  const options = { method };
  if (body !== undefined) {
    options.headers = { "content-type": "application/json" };
    options.body = JSON.stringify(body);
  }
  const response = await fetchImpl(url, options);
  if (!response.ok) {
    await throwApiError(response);
  }
  return response.json();
}

/**
 * @param {CreateSessionRequest} request
 * @param {{ fetchImpl?: typeof fetch }} options
 * @returns {Promise<CreateSessionResponse>}
 */
export function createSession(request, { fetchImpl = fetch } = {}) {
  return requestJson(
    "/sessions",
    {
      method: "POST",
      body: {
        world_id: request.world_id,
        scene_id: request.scene_id,
        player_name: request.player_name,
        active_persona_id: request.active_persona_id,
      },
    },
    fetchImpl,
  );
}

/**
 * @param {{ fetchImpl?: typeof fetch }} options
 * @returns {Promise<ContentCatalog>}
 */
export function getContentCatalog({ fetchImpl = fetch } = {}) {
  return requestJson("/content/catalog", { method: "GET" }, fetchImpl);
}

/**
 * @param {string} sessionId
 * @param {{ fetchImpl?: typeof fetch }} options
 * @returns {Promise<GetSessionResponse>}
 */
export function getSession(sessionId, { fetchImpl = fetch } = {}) {
  return requestJson(`/sessions/${encodeURIComponent(sessionId)}`, { method: "GET" }, fetchImpl);
}

/**
 * @param {string} sessionId
 * @param {CreateTurnRequest} request
 * @param {{ fetchImpl?: typeof fetch }} options
 * @returns {Promise<CreateTurnResponse>}
 */
export function createTurn(sessionId, request, { fetchImpl = fetch } = {}) {
  return requestJson(
    `/sessions/${encodeURIComponent(sessionId)}/turns`,
    {
      method: "POST",
      body: {
        message: request.message,
        request_cloud: request.request_cloud,
      },
    },
    fetchImpl,
  );
}

function applyEvent(result, eventName, payload) {
  if (eventName === "text") {
    result.text += payload.text;
    return false;
  }
  if (eventName === "final" || eventName === "failure") {
    if (eventName === "failure") {
      result.text = payload.text;
    }
    result.route = payload.route;
    result.memory_written = payload.memory_written;
    result.warnings = payload.warnings;
    return true;
  }
  return false;
}

function parseFrame(result, frame) {
  let eventName = "message";
  const dataLines = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) {
      eventName = line.slice("event:".length).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).trimStart());
    }
  }
  if (dataLines.length === 0) {
    return false;
  }
  return applyEvent(result, eventName, JSON.parse(dataLines.join("\n")));
}

async function parseEventStream(response) {
  if (!response.body) {
    throw new ApiError("invalid_stream", "The backend returned an empty event stream.", 502);
  }
  const result = { text: "", route: null, memory_written: false, warnings: [] };
  const decoder = new TextDecoder();
  let buffer = "";
  let terminal = false;
  for await (const chunk of response.body) {
    buffer += decoder.decode(chunk, { stream: true });
    buffer = buffer.replaceAll("\r\n", "\n");
    let boundary = buffer.indexOf("\n\n");
    while (boundary >= 0) {
      terminal = parseFrame(result, buffer.slice(0, boundary)) || terminal;
      buffer = buffer.slice(boundary + 2);
      boundary = buffer.indexOf("\n\n");
    }
  }
  buffer += decoder.decode();
  if (buffer.trim()) {
    terminal = parseFrame(result, buffer.trim()) || terminal;
  }
  if (!terminal || !result.route) {
    throw new ApiError("invalid_stream", "The backend event stream ended unexpectedly.", 502);
  }
  return result;
}

/**
 * @param {string} sessionId
 * @param {CreateTurnRequest} request
 * @param {{ fetchImpl?: typeof fetch }} options
 * @returns {Promise<CreateTurnResponse>}
 */
export async function createBufferedTurn(sessionId, request, { fetchImpl = fetch } = {}) {
  const response = await fetchImpl(`/sessions/${encodeURIComponent(sessionId)}/turns/stream`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      message: request.message,
      request_cloud: request.request_cloud,
    }),
  });
  if (!response.ok) {
    await throwApiError(response);
  }
  return parseEventStream(response);
}
