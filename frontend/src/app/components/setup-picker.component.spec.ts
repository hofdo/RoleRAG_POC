import { TestBed } from '@angular/core/testing';
import { ApiService } from '../api.service';
import { SessionStore } from '../session-store';
import { SetupPickerComponent } from './setup-picker.component';
import type { ContentCatalog, GetSessionResponse, RecentSessionsResponse } from '../models';

const catalogFixture: ContentCatalog = {
  worlds: [{ id: 'w', name: 'World', default_scene_id: 'sc', scene_ids: ['sc', 'east-wing'], persona_ids: ['p', 'warden'] }],
  scenes: [
    { id: 'sc', title: 'Rose Gallery', location: 'Winter Palace', player_visible_summary: '', active_personas: [] },
    { id: 'east-wing', title: 'East Wing', location: 'Winter Palace', player_visible_summary: '', active_personas: [] },
  ],
  personas: [
    { id: 'p', name: 'Archivist', role: 'npc', public_description: '', speaking_style: '' },
    { id: 'warden', name: 'Warden', role: 'npc', public_description: '', speaking_style: '' },
  ],
};

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
  updateSessionSceneCalls: { sessionId: string; sceneId: string }[] = [];
  updateSessionScene(sessionId: string, sceneId: string): Promise<GetSessionResponse> {
    this.updateSessionSceneCalls.push({ sessionId, sceneId });
    return Promise.resolve({
      session_id: sessionId,
      world_id: 'w',
      active_scene_id: sceneId,
      active_persona_id: 'p',
      recent_turns: [],
    });
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

describe('SetupPickerComponent scene/persona switch controls', () => {
  it('shows switch controls once a session is active, and switches scene on click', async () => {
    const { fixture, store, api } = make();
    store.catalog.set(catalogFixture);
    store.session.set({
      session_id: 's1',
      world_id: 'w',
      active_scene_id: 'sc',
      active_persona_id: 'p',
    });
    fixture.detectChanges();

    const el = fixture.nativeElement as HTMLElement;
    const sceneSelect = el.querySelector('select') as HTMLSelectElement;
    expect(sceneSelect).not.toBeNull();
    sceneSelect.value = 'east-wing';
    sceneSelect.dispatchEvent(new Event('change'));

    const button = Array.from(el.querySelectorAll('button')).find(
      (b) => b.textContent?.trim() === 'Switch scene',
    );
    expect(button).toBeDefined();
    button?.click();
    await Promise.resolve();
    await Promise.resolve();

    expect(api.updateSessionSceneCalls).toEqual([{ sessionId: 's1', sceneId: 'east-wing' }]);
  });

  it('sets the persona override signal when the persona select changes', () => {
    const { fixture, store } = make();
    store.catalog.set(catalogFixture);
    store.session.set({
      session_id: 's1',
      world_id: 'w',
      active_scene_id: 'sc',
      active_persona_id: 'p',
    });
    fixture.detectChanges();

    const el = fixture.nativeElement as HTMLElement;
    const selects = Array.from(el.querySelectorAll('select'));
    const personaSelect = selects[1];
    expect(personaSelect).toBeDefined();
    personaSelect.value = 'warden';
    personaSelect.dispatchEvent(new Event('change'));

    expect(store.personaOverride()).toBe('warden');
  });

  it('does not show switch controls before a session is active', () => {
    const { fixture } = make();
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    const button = Array.from(el.querySelectorAll('button')).find(
      (b) => b.textContent?.trim() === 'Switch scene',
    );
    expect(button).toBeUndefined();
  });
});
