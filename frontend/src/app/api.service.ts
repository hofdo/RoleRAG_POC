import { Injectable } from '@angular/core';
import type {
  AddCanonFactRequest,
  CanonFact,
  CanonFactsResponse,
  ContentCatalog,
  CreateSessionRequest,
  CreateSessionResponse,
  CreateTurnRequest,
  DeleteLastTurnResponse,
  EvalRunDetail,
  EvalRunsResponse,
  GetSessionResponse,
  RecentSessionsResponse,
  RuntimeStatus,
  SessionMemoriesResponse,
  SessionTurnDetailsResponse,
  TurnDetailResponse,
  TurnResult,
} from './models';

// Ported near-verbatim from app/web/assets/api-client.mjs. Kept on fetch (not HttpClient)
// because /turns/stream is POST + SSE — fetch + ReadableStream is the proven path and
// EventSource cannot do POST. See the original module for the design rationale.

export class ApiError extends Error {
  override name = 'ApiError';
  constructor(
    public code: string,
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

async function throwApiError(response: Response): Promise<never> {
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    throw new ApiError('request_failed', 'The backend request failed.', response.status);
  }
  const error = (body as { error?: { code?: unknown; message?: unknown } })?.error;
  throw new ApiError(
    typeof error?.code === 'string' ? error.code : 'request_failed',
    typeof error?.message === 'string' ? error.message : 'The backend request failed.',
    response.status,
  );
}

async function requestJson<T>(
  url: string,
  { method = 'GET', body }: { method?: string; body?: unknown } = {},
): Promise<T> {
  const options: RequestInit = { method };
  if (body !== undefined) {
    options.headers = { 'content-type': 'application/json' };
    options.body = JSON.stringify(body);
  }
  const response = await fetch(url, options);
  if (!response.ok) {
    await throwApiError(response);
  }
  return response.json() as Promise<T>;
}

// --- SSE buffered-turn parsing (ported from api-client.mjs:247-333) ---

function applyEvent(
  result: TurnResult,
  eventName: string,
  payload: any,
  onStage?: (stage: string) => void,
): boolean {
  if (eventName === 'stage') {
    onStage?.(String((payload as { stage?: unknown }).stage ?? ''));
    return false;
  }
  if (eventName === 'error') {
    const p = payload as { code?: string; message?: string; status?: number };
    throw new ApiError(p.code ?? 'stream_error', p.message ?? 'Turn failed.', p.status ?? 502);
  }
  if (eventName === 'text') {
    result.text += payload.text;
    return false;
  }
  if (eventName === 'confirmation_required') {
    result.status = 'confirmation_required';
    result.route = payload.route;
    result.warnings = payload.warnings;
    return true;
  }
  if (eventName === 'final' || eventName === 'failure') {
    if (eventName === 'failure') {
      result.text = payload.text;
    }
    result.status = 'completed';
    result.route = payload.route;
    result.finish_reason = payload.finish_reason ?? null;
    result.memory_written = payload.memory_written;
    result.critic_status = payload.critic_status;
    result.warnings = payload.warnings;
    result.retrieval = payload.retrieval ?? null;
    result.stage_timings = payload.stage_timings;
    return true;
  }
  return false;
}

function parseFrame(
  result: TurnResult,
  frame: string,
  onStage?: (stage: string) => void,
): boolean {
  let eventName = 'message';
  const dataLines: string[] = [];
  for (const line of frame.split('\n')) {
    if (line.startsWith('event:')) {
      eventName = line.slice('event:'.length).trim();
    } else if (line.startsWith('data:')) {
      dataLines.push(line.slice('data:'.length).trimStart());
    }
  }
  if (dataLines.length === 0) {
    return false;
  }
  let payload: unknown;
  try {
    payload = JSON.parse(dataLines.join('\n'));
  } catch {
    // A corrupt frame must surface as a typed stream error, not a raw SyntaxError.
    throw new ApiError('invalid_stream', 'The backend event stream sent a malformed frame.', 502);
  }
  return applyEvent(result, eventName, payload, onStage);
}

async function parseEventStream(
  response: Response,
  onStage?: (stage: string) => void,
): Promise<TurnResult> {
  if (!response.body) {
    throw new ApiError('invalid_stream', 'The backend returned an empty event stream.', 502);
  }
  const result: TurnResult = {
    status: 'completed',
    text: '',
    route: null,
    finish_reason: null,
    memory_written: false,
    critic_status: 'skipped',
    warnings: [],
    retrieval: null,
  };
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let terminal = false;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    buffer = buffer.replaceAll('\r\n', '\n');
    let boundary = buffer.indexOf('\n\n');
    while (boundary >= 0) {
      terminal = parseFrame(result, buffer.slice(0, boundary), onStage) || terminal;
      buffer = buffer.slice(boundary + 2);
      boundary = buffer.indexOf('\n\n');
    }
  }
  buffer += decoder.decode();
  if (buffer.trim()) {
    terminal = parseFrame(result, buffer.trim(), onStage) || terminal;
  }
  if (!terminal || !result.route) {
    throw new ApiError('invalid_stream', 'The backend event stream ended unexpectedly.', 502);
  }
  return result;
}

// Long local turns can take minutes (generation + critic + repair + memory).
const STREAM_TIMEOUT_MS = 600000;

@Injectable({ providedIn: 'root' })
export class ApiService {
  getRuntimeStatus(): Promise<RuntimeStatus> {
    return requestJson('/runtime/status');
  }

  getContentCatalog(): Promise<ContentCatalog> {
    return requestJson('/content/catalog');
  }

  getEvalRuns(): Promise<EvalRunsResponse> {
    return requestJson('/diagnostics/eval-runs');
  }

  getRecentSessions(): Promise<RecentSessionsResponse> {
    return requestJson('/sessions');
  }

  createSession(request: CreateSessionRequest): Promise<CreateSessionResponse> {
    return requestJson('/sessions', {
      method: 'POST',
      body: {
        world_id: request.world_id,
        scene_id: request.scene_id,
        player_name: request.player_name,
        active_persona_id: request.active_persona_id,
      },
    });
  }

  getSession(sessionId: string): Promise<GetSessionResponse> {
    return requestJson(`/sessions/${encodeURIComponent(sessionId)}`);
  }

  getSessionMemories(sessionId: string): Promise<SessionMemoriesResponse> {
    return requestJson(`/sessions/${encodeURIComponent(sessionId)}/memories`);
  }

  getTurnDetail(sessionId: string, turnIndex: number): Promise<TurnDetailResponse> {
    return requestJson(`/sessions/${encodeURIComponent(sessionId)}/turns/${turnIndex}`);
  }

  // One call for every turn's diagnostics (Analytics/Inspector) instead of N per-turn fetches.
  getSessionTurnDetails(sessionId: string): Promise<SessionTurnDetailsResponse> {
    return requestJson(`/sessions/${encodeURIComponent(sessionId)}/turn-details`);
  }

  getEvalRun(runId: string): Promise<EvalRunDetail> {
    return requestJson(`/diagnostics/eval-runs/${encodeURIComponent(runId)}`);
  }

  getCanon(sessionId: string): Promise<CanonFactsResponse> {
    return requestJson(`/sessions/${encodeURIComponent(sessionId)}/canon`);
  }

  addCanonFact(sessionId: string, request: AddCanonFactRequest): Promise<CanonFact> {
    return requestJson(`/sessions/${encodeURIComponent(sessionId)}/canon`, {
      method: 'POST',
      body: { text: request.text },
    });
  }

  async deleteCanonFact(sessionId: string, factId: string): Promise<void> {
    const response = await fetch(
      `/sessions/${encodeURIComponent(sessionId)}/canon/${encodeURIComponent(factId)}`,
      { method: 'DELETE' },
    );
    if (!response.ok) {
      await throwApiError(response);
    }
  }

  deleteLastTurn(sessionId: string): Promise<DeleteLastTurnResponse> {
    return requestJson(`/sessions/${encodeURIComponent(sessionId)}/turns/last`, {
      method: 'DELETE',
    });
  }

  // Buffered turn over SSE (/turns/stream). Resolves once the terminal event arrives.
  async createBufferedTurn(
    sessionId: string,
    request: CreateTurnRequest,
    { timeoutMs = STREAM_TIMEOUT_MS, onStage }: { timeoutMs?: number; onStage?: (stage: string) => void } = {},
  ): Promise<TurnResult> {
    const response = await fetch(`/sessions/${encodeURIComponent(sessionId)}/turns/stream`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      signal: AbortSignal.timeout(timeoutMs),
      body: JSON.stringify({
        message: request.message,
        active_persona_id: request.active_persona_id ?? null,
      }),
    });
    if (!response.ok) {
      await throwApiError(response);
    }
    return parseEventStream(response, onStage);
  }

  updateSessionScene(sessionId: string, sceneId: string): Promise<CreateSessionResponse> {
    return requestJson(`/sessions/${encodeURIComponent(sessionId)}/scene`, {
      method: 'POST',
      body: { scene_id: sceneId },
    });
  }
}
