import { TestBed } from '@angular/core/testing';
import { ApiService } from './api.service';
import { SessionStore } from './session-store';
import type { CreateTurnRequest, TurnResult } from './models';

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

class FakeApi {
  calls: { sessionId: string; request: CreateTurnRequest }[] = [];
  next: TurnResult = completedTurn('hello');
  failPanels = false;

  createBufferedTurn(sessionId: string, request: CreateTurnRequest): Promise<TurnResult> {
    this.calls.push({ sessionId, request });
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
