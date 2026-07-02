import { TestBed } from '@angular/core/testing';
import { ApiError, ApiService } from './api.service';
import { SessionStore } from './session-store';
import type {
  CreateTurnRequest,
  DeleteLastTurnResponse,
  GetSessionResponse,
  RecentSessionsResponse,
  SessionTurnDetailsResponse,
  TurnResult,
} from './models';

// SessionStore is the logic hub (turn execution, cloud-confirm, memory). Tested against a fake
// ApiService so the turn flow is exercised without a backend.

function completedTurn(text: string): TurnResult {
  return {
    status: 'completed',
    text,
    route: { provider: 'local', model: 'm', reason: 'r' },
    finish_reason: 'stop',
    memory_written: false,
    critic_status: 'accepted',
    warnings: [],
    retrieval: null,
    stage_timings: { generation: 1 },
  };
}

const confirmationTurn: TurnResult = {
  status: 'confirmation_required',
  text: '',
  route: { provider: 'cloud', model: 'c', reason: 'ask' },
  finish_reason: null,
  memory_written: false,
  critic_status: 'skipped',
  warnings: [],
  retrieval: null,
};

const sessionDetailFixture: GetSessionResponse = {
  session_id: 's1',
  world_id: 'w',
  active_scene_id: 'sc',
  active_persona_id: 'p',
  recent_turns: Array.from({ length: 8 }, (_, i) => ({
    turn_index: i,
    user_message: `recent player ${i}`,
    assistant_message: `recent assistant ${i}`,
    created_at: '2026-01-01T00:00:00Z',
  })),
};

const turnDetailsFixture: SessionTurnDetailsResponse = {
  session_id: 's1',
  turns: Array.from({ length: 20 }, (_, i) => ({
    turn_index: i,
    scene_id: 'sc',
    persona_id: 'p',
    user_message: `full player ${i}`,
    assistant_message: `full assistant ${i}`,
    route: { provider: 'local', model: 'm', reason: 'r' },
    created_at: '2026-01-01T00:00:00Z',
    finish_reason: 'stop',
    memory_written: false,
    critic_status: 'accepted',
    warnings: [],
    errors: [],
    stage_timings: {},
    retrieval: null,
  })),
};

class FakeApi {
  calls: { sessionId: string; request: CreateTurnRequest }[] = [];
  next: TurnResult = completedTurn('hello');
  failPanels = false;
  stagesToEmit: string[] = [];

  createBufferedTurn(
    sessionId: string,
    request: CreateTurnRequest,
    options?: { onStage?: (stage: string) => void },
  ): Promise<TurnResult> {
    this.calls.push({ sessionId, request });
    for (const stage of this.stagesToEmit) {
      options?.onStage?.(stage);
    }
    return Promise.resolve(this.next);
  }
  getSessionMemories(): Promise<{ session_id: string; memories: [] }> {
    if (this.failPanels) return Promise.reject(new Error('backend down'));
    return Promise.resolve({ session_id: 's1', memories: [] });
  }
  getCanon(_sessionId: string): Promise<{ session_id: string; facts: [] }> {
    if (this.failPanels) return Promise.reject(new Error('backend down'));
    return Promise.resolve({ session_id: 's1', facts: [] });
  }
  addCanonFact(
    _sessionId: string,
    request: { text: string },
  ): Promise<{ id: string; text: string }> {
    if (this.failPanels) return Promise.reject(new Error('backend down'));
    return Promise.resolve({ id: 'f1', text: request.text });
  }
  deleteCanonFact(_sessionId: string, _factId: string): Promise<void> {
    if (this.failPanels) return Promise.reject(new Error('backend down'));
    return Promise.resolve();
  }
  deleteLastTurnCalls: string[] = [];
  deleteLastTurnShouldFail = false;
  deleteLastTurn(sessionId: string): Promise<DeleteLastTurnResponse> {
    this.deleteLastTurnCalls.push(sessionId);
    if (this.deleteLastTurnShouldFail) return Promise.reject(new Error('delete failed'));
    return Promise.resolve({
      session_id: sessionId,
      deleted_turn_index: 1,
      user_message: 'a question',
      deleted_memory_count: 0,
    });
  }
  getSession(_sessionId: string): Promise<GetSessionResponse> {
    return Promise.resolve(sessionDetailFixture);
  }
  getSessionTurnDetails(_sessionId: string): Promise<SessionTurnDetailsResponse> {
    return Promise.resolve(turnDetailsFixture);
  }
  getRecentSessions(): Promise<RecentSessionsResponse> {
    return Promise.resolve({ sessions: [] });
  }
}

function setup(): { store: SessionStore; api: FakeApi } {
  const api = new FakeApi();
  TestBed.configureTestingModule({
    providers: [SessionStore, { provide: ApiService, useValue: api }],
  });
  const store = TestBed.inject(SessionStore);
  store.session.set({
    session_id: 's1',
    world_id: 'w',
    active_scene_id: 'sc',
    active_persona_id: 'p',
  });
  return { store, api };
}

describe('SessionStore turn flow', () => {
  it('appends player + assistant lines on a completed turn and records debug', async () => {
    const { store, api } = setup();
    api.next = completedTurn('an answer');

    await store.sendMessage('a question', false);

    expect(store.transcript().map((e) => [e.role, e.text])).toEqual([
      ['player', 'a question'],
      ['assistant', 'an answer'],
    ]);
    expect(store.pendingConfirm()).toBeNull();
    expect(store.lastDebug()?.provider).toBe('local');
    expect(store.busy()).toBe(false);
  });

  it('holds the message and does not append when confirmation is required', async () => {
    const { store, api } = setup();
    api.next = confirmationTurn;

    await store.sendMessage('go cloud?', true);

    expect(store.pendingConfirm()).toEqual({ message: 'go cloud?' });
    expect(store.transcript()).toEqual([]);
  });

  it('confirmCloud resubmits the held message with cloud_confirmed and clears the prompt', async () => {
    const { store, api } = setup();
    api.next = confirmationTurn;
    await store.sendMessage('go cloud?', true);

    api.next = completedTurn('cloud reply');
    await store.confirmCloud();

    const last = api.calls.at(-1)!.request;
    expect(last).toEqual({ message: 'go cloud?', request_cloud: true, cloud_confirmed: true, force_local: false });
    expect(store.pendingConfirm()).toBeNull();
    expect(store.transcript().at(-1)).toEqual({ role: 'assistant', text: 'cloud reply', source: 'new' });
  });

  it('forceLocal resubmits the held message pinned to local', async () => {
    const { store, api } = setup();
    api.next = confirmationTurn;
    await store.sendMessage('go cloud?', true);

    api.next = completedTurn('local reply');
    await store.forceLocal();

    const last = api.calls.at(-1)!.request;
    expect(last).toEqual({ message: 'go cloud?', request_cloud: false, cloud_confirmed: false, force_local: true });
    expect(store.pendingConfirm()).toBeNull();
  });

  it('is a no-op without an active session', async () => {
    const { store, api } = setup();
    store.session.set(null);
    await store.sendMessage('nope', false);
    expect(api.calls).toEqual([]);
  });

  it('clears currentStage after the turn completes', async () => {
    const { store, api } = setup();
    api.stagesToEmit = ['session', 'retrieval', 'generation'];

    await store.sendMessage('a question', false);

    expect(store.currentStage()).toBeNull();
  });

  it('clears currentStage even when the turn fails', async () => {
    const { store, api } = setup();
    api.createBufferedTurn = () => Promise.reject(new Error('boom'));

    await store.sendMessage('a question', false);

    expect(store.currentStage()).toBeNull();
    expect(store.turnError()).toContain('boom');
  });

  it('keeps returning false from sendMessage when the turn fails', async () => {
    const { store, api } = setup();
    api.createBufferedTurn = () =>
      Promise.reject(new ApiError('provider_timeout', 'boom', 504));

    const ok = await store.sendMessage('hello', false);

    expect(ok).toBeFalse();
    expect(store.turnError()).toContain('provider_timeout');
  });

  it('returns true from sendMessage when the turn completes', async () => {
    const { store } = setup();
    const ok = await store.sendMessage('hello', false);
    expect(ok).toBeTrue();
  });

  it('returns true from sendMessage when confirmation is required', async () => {
    const { store, api } = setup();
    api.next = confirmationTurn;
    const ok = await store.sendMessage('go cloud?', true);
    expect(ok).toBeTrue();
  });

  it('confirmCloud/forceLocal resolve true with no pending confirmation', async () => {
    const { store } = setup();
    expect(await store.confirmCloud()).toBeTrue();
    expect(await store.forceLocal()).toBeTrue();
  });
});

describe('SessionStore rerollLast', () => {
  it('deletes the last turn, drops it from the transcript, and resends the message', async () => {
    const { store, api } = setup();
    api.next = completedTurn('an answer');
    await store.sendMessage('a question', false);
    expect(store.transcript().length).toBe(2);

    api.next = completedTurn('a reroll answer');
    await store.rerollLast();

    expect(api.deleteLastTurnCalls).toEqual(['s1']);
    expect(store.transcript().map((e) => [e.role, e.text])).toEqual([
      ['player', 'a question'],
      ['assistant', 'a reroll answer'],
    ]);
    expect(store.busy()).toBe(false);
  });

  it('is a no-op without an active session', async () => {
    const { store, api } = setup();
    store.session.set(null);

    await store.rerollLast();

    expect(api.deleteLastTurnCalls).toEqual([]);
  });

  it('is a no-op when there is no player turn in the transcript', async () => {
    const { store, api } = setup();

    await store.rerollLast();

    expect(api.deleteLastTurnCalls).toEqual([]);
  });

  it('is a no-op while busy', async () => {
    const { store, api } = setup();
    api.next = completedTurn('an answer');
    await store.sendMessage('a question', false);
    store.busy.set(true);

    await store.rerollLast();

    expect(api.deleteLastTurnCalls).toEqual([]);
  });

  it('surfaces a delete failure, leaves the transcript intact, and does not resend', async () => {
    const { store, api } = setup();
    api.next = completedTurn('an answer');
    await store.sendMessage('a question', false);
    api.deleteLastTurnShouldFail = true;

    await store.rerollLast();

    expect(store.turnError()).toContain('delete failed');
    expect(store.transcript().length).toBe(2);
    expect(store.busy()).toBe(false);
    expect(api.calls.length).toBe(1); // no resend attempted
  });
});

describe('SessionStore resume', () => {
  it('resume loads the full transcript from turn-details', async () => {
    const { store } = setup();

    await store.resume('s1');

    expect(store.transcript().length).toBe(40); // 20 player + 20 assistant entries
    expect(store.transcript()[0]).toEqual({
      role: 'player',
      text: 'full player 0',
      label: 'Resumed turn #0',
      source: 'resumed',
    });
  });

  it('loadRecentSessions populates recentSessions from the API', async () => {
    const { store, api } = setup();
    api.getRecentSessions = () =>
      Promise.resolve({
        sessions: [
          {
            session_id: 's9',
            world_id: 'w',
            active_scene_id: 'sc',
            active_persona_id: 'p',
            player_name: 'Ada',
            created_at: '2026-01-01T00:00:00Z',
            updated_at: '2026-01-02T00:00:00Z',
          },
        ],
      });

    await store.loadRecentSessions();

    expect(store.recentSessions().length).toBe(1);
    expect(store.recentSessions()[0].session_id).toBe('s9');
  });

  it('loadRecentSessions clears the list on failure instead of throwing', async () => {
    const { store, api } = setup();
    api.getRecentSessions = () => Promise.reject(new Error('backend down'));

    await store.loadRecentSessions();

    expect(store.recentSessions()).toEqual([]);
  });
});

describe('SessionStore panel errors', () => {
  it('surfaces a memory refresh failure and keeps the previous list', async () => {
    const { store, api } = setup();
    store.memories.set([{ summary: 'kept' } as never]);

    api.failPanels = true;
    await store.refreshMemories();

    expect(store.memoryError()).toContain('backend down');
    expect(store.memories().length).toBe(1);
  });

  it('clears the memory error on a later successful refresh', async () => {
    const { store, api } = setup();
    api.failPanels = true;
    await store.refreshMemories();

    api.failPanels = false;
    await store.refreshMemories();

    expect(store.memoryError()).toBeNull();
  });

  it('surfaces a rejected canon add without appending the fact', async () => {
    const { store, api } = setup();
    api.failPanels = true;

    await store.addCanon('the door is locked');

    expect(store.canonError()).toContain('Fact not added');
    expect(store.canonFacts()).toEqual([]);
  });

  it('appends a canon fact only after the server confirms it', async () => {
    const { store } = setup();

    await store.addCanon('the door is locked');

    expect(store.canonFacts().map((fact) => fact.text)).toEqual(['the door is locked']);
    expect(store.canonError()).toBeNull();
  });

  it('surfaces a rejected canon delete and keeps the fact', async () => {
    const { store, api } = setup();
    await store.addCanon('the door is locked');

    api.failPanels = true;
    await store.deleteCanon('f1');

    expect(store.canonError()).toContain('Fact not deleted');
    expect(store.canonFacts().length).toBe(1);
  });
});
