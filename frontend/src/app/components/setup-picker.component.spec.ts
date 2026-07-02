import { TestBed } from '@angular/core/testing';
import { ApiService } from '../api.service';
import { SessionStore } from '../session-store';
import { SetupPickerComponent } from './setup-picker.component';
import type { GetSessionResponse, RecentSessionsResponse } from '../models';

class FakeApi {
  recentSessions: RecentSessionsResponse = { sessions: [] };
  getRecentSessions(): Promise<RecentSessionsResponse> {
    return Promise.resolve(this.recentSessions);
  }
  getSession(sessionId: string): Promise<GetSessionResponse> {
    return Promise.resolve({
      session_id: sessionId,
      world_id: 'w',
      active_scene_id: 'sc',
      active_persona_id: 'p',
      recent_turns: [],
    });
  }
  getSessionTurnDetails(sessionId: string): Promise<{ session_id: string; turns: [] }> {
    return Promise.resolve({ session_id: sessionId, turns: [] });
  }
  getSessionMemories(): Promise<{ session_id: string; memories: [] }> {
    return Promise.resolve({ session_id: 's1', memories: [] });
  }
  getCanon(): Promise<{ session_id: string; facts: [] }> {
    return Promise.resolve({ session_id: 's1', facts: [] });
  }
}

function make(): { fixture: ReturnType<typeof TestBed.createComponent<SetupPickerComponent>>; store: SessionStore; api: FakeApi } {
  const api = new FakeApi();
  TestBed.configureTestingModule({
    imports: [SetupPickerComponent],
    providers: [SessionStore, { provide: ApiService, useValue: api }],
  });
  const fixture = TestBed.createComponent(SetupPickerComponent);
  const store = TestBed.inject(SessionStore);
  return { fixture, store, api };
}

describe('SetupPickerComponent', () => {
  it('loads recent sessions on construction', async () => {
    const { fixture, store } = make();
    fixture.componentInstance;
    await Promise.resolve();
    await Promise.resolve();

    expect(store.recentSessions()).toEqual([]);
  });

  it('shows a resume select once recent sessions load, and resumes on click', async () => {
    const { fixture, store, api } = make();
    api.recentSessions = {
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
    };
    // Reconstruct so the constructor picks up the fixture list.
    const fixture2 = TestBed.createComponent(SetupPickerComponent);
    await Promise.resolve();
    await Promise.resolve();
    fixture2.detectChanges();

    const el = fixture2.nativeElement as HTMLElement;
    const select = el.querySelector('select#resume-select') as HTMLSelectElement | null;
    expect(select).not.toBeNull();
    expect(select?.textContent).toContain('Ada');

    const button = Array.from(el.querySelectorAll('button')).find((b) => b.textContent?.trim() === 'Resume');
    expect(button).toBeDefined();
    button?.click();
    await Promise.resolve();
    await Promise.resolve();

    expect(store.sessionId()).toBe('s9');
    void fixture;
    void store;
  });

  it('hides the resume block when there are no recent sessions', () => {
    const { fixture } = make();
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('select#resume-select')).toBeNull();
  });
});
