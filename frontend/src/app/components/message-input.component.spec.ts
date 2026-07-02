import { TestBed } from '@angular/core/testing';
import { ApiService } from '../api.service';
import { SessionStore } from '../session-store';
import { MessageInputComponent } from './message-input.component';
import type { CreateTurnRequest, TurnResult } from '../models';

class FakeApi {
  calls: CreateTurnRequest[] = [];
  createBufferedTurn(_sessionId: string, request: CreateTurnRequest): Promise<TurnResult> {
    this.calls.push(request);
    return Promise.resolve({
      status: 'completed',
      text: 'reply',
      route: { provider: 'local', model: 'm', reason: 'r' },
      finish_reason: 'stop',
      memory_written: false,
      critic_status: 'accepted',
      warnings: [],
      retrieval: null,
    });
  }
  getSessionMemories(): Promise<{ session_id: string; memories: [] }> {
    return Promise.resolve({ session_id: 's1', memories: [] });
  }
}

function make(): { fixture: ReturnType<typeof TestBed.createComponent<MessageInputComponent>>; store: SessionStore; api: FakeApi } {
  const api = new FakeApi();
  TestBed.configureTestingModule({
    imports: [MessageInputComponent],
    providers: [SessionStore, { provide: ApiService, useValue: api }],
  });
  const fixture = TestBed.createComponent(MessageInputComponent);
  const store = TestBed.inject(SessionStore);
  return { fixture, store, api };
}

describe('MessageInputComponent', () => {
  it('disables Send with no session, empty draft, or while busy', () => {
    const { fixture, store } = make();
    const c = fixture.componentInstance;

    expect(c.sendDisabled()).toBe(true); // no session

    store.session.set({ session_id: 's1', world_id: 'w', active_scene_id: 'sc', active_persona_id: 'p' });
    expect(c.sendDisabled()).toBe(true); // empty draft

    c.draft.set('hi');
    expect(c.sendDisabled()).toBe(false);

    store.busy.set(true);
    expect(c.sendDisabled()).toBe(true);
  });

  it('send() submits the trimmed draft and clears it', async () => {
    const { fixture, store, api } = make();
    const c = fixture.componentInstance;
    store.session.set({ session_id: 's1', world_id: 'w', active_scene_id: 'sc', active_persona_id: 'p' });
    c.draft.set('  hello  ');

    c.send();
    await Promise.resolve();
    await Promise.resolve();

    expect(api.calls.at(-1)?.message).toBe('hello');
    expect(c.draft()).toBe('');
  });

  it('shows the cloud-confirm banner when a confirmation is pending', () => {
    const { fixture, store } = make();
    store.pendingConfirm.set({ message: 'go cloud?' });
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('.confirm-banner')).not.toBeNull();
  });

  it('shows the current stage label while busy', () => {
    const { fixture, store } = make();
    store.busy.set(true);
    store.currentStage.set('retrieval');
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('.stage')?.textContent).toContain('Retrieving memories');
  });

  it('hides the stage line when not busy', () => {
    const { fixture, store } = make();
    store.busy.set(false);
    store.currentStage.set('retrieval');
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('.stage')).toBeNull();
  });

  it('falls back to the raw stage name for an unknown stage', () => {
    const { fixture } = make();
    const c = fixture.componentInstance;
    expect(c.stageLabel('mystery')).toBe('mystery');
  });
});
