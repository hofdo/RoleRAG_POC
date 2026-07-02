import {
  buildCatalogSessionRequest,
  buildSessionRequest,
  buildTurnRequest,
  createCatalogSelection,
  formatRetrievalDiagnostics,
  formatStageTimings,
  isConfirmationRequired,
} from './play-model';
import type { ContentCatalog, RetrievalDiagnostics } from './models';

// One runnable check for the non-trivial ported logic (catalog wiring, diagnostics shaping).
// Mirrors tests/frontend/play-model.test.mjs in the vanilla source.

const catalog: ContentCatalog = {
  worlds: [
    { id: 'w1', name: 'World One', default_scene_id: 's2', scene_ids: ['s1', 's2'], persona_ids: ['p1'] },
  ],
  scenes: [
    { id: 's1', title: 'Scene One', location: 'Hall', player_visible_summary: '', active_personas: [] },
    { id: 's2', title: 'Scene Two', location: 'Garden', player_visible_summary: '', active_personas: [] },
  ],
  personas: [
    { id: 'p1', name: 'Archivist', role: 'narrator', public_description: '', speaking_style: '' },
  ],
};

describe('play-model', () => {
  it('selects the default scene and first persona from the catalog', () => {
    const sel = createCatalogSelection(catalog);
    expect(sel.world?.id).toBe('w1');
    expect(sel.scenes.length).toBe(2);
    expect(sel.sceneId).toBe('s2'); // default_scene_id wins
    expect(sel.personaId).toBe('p1');
  });

  it('returns empty selection when no worlds exist', () => {
    const sel = createCatalogSelection({ worlds: [], scenes: [], personas: [] });
    expect(sel.world).toBeNull();
    expect(sel.sceneId).toBe('');
  });

  it('builds a turn request with just the message', () => {
    expect(buildTurnRequest('hi')).toEqual({
      message: 'hi',
      active_persona_id: undefined,
    });
  });

  it('builds a turn request with a persona override', () => {
    expect(buildTurnRequest('hi', { personaId: 'warden' })).toEqual({
      message: 'hi',
      active_persona_id: 'warden',
    });
  });

  it('builds a session request defaulting to the local provider', () => {
    expect(
      buildSessionRequest({ worldId: 'w1', sceneId: 's2', personaId: 'p1', playerName: 'Avery' }),
    ).toEqual({
      world_id: 'w1',
      scene_id: 's2',
      active_persona_id: 'p1',
      player_name: 'Avery',
      provider: 'local',
    });
  });

  it('builds a session request carrying the chosen provider', () => {
    expect(
      buildSessionRequest({
        worldId: 'w1',
        sceneId: 's2',
        personaId: 'p1',
        playerName: 'Avery',
        provider: 'cloud',
      }),
    ).toEqual({
      world_id: 'w1',
      scene_id: 's2',
      active_persona_id: 'p1',
      player_name: 'Avery',
      provider: 'cloud',
    });
  });

  it('builds a catalog session request carrying the chosen provider', () => {
    const sel = createCatalogSelection(catalog);
    const request = buildCatalogSessionRequest(sel, 'Avery', 'cloud');
    expect(request.provider).toBe('cloud');
  });

  it('builds a catalog session request defaulting to local when no provider given', () => {
    const sel = createCatalogSelection(catalog);
    const request = buildCatalogSessionRequest(sel, 'Avery');
    expect(request.provider).toBe('local');
  });

  it('detects confirmation-required turns', () => {
    expect(isConfirmationRequired({ status: 'confirmation_required' })).toBe(true);
    expect(isConfirmationRequired({ status: 'completed' })).toBe(false);
    expect(isConfirmationRequired(null)).toBe(false);
  });

  it('formats stage timings and retrieval boosts', () => {
    expect(formatStageTimings({ retrieval: 1.23, generation: 4.5 })).toBe('retrieval 1.2s; generation 4.5s');
    expect(formatStageTimings(undefined)).toBe('none');

    const retrieval: RetrievalDiagnostics = {
      query: 'q',
      selected: [
        {
          selected_rank: 1,
          source: 'lore',
          collection: 'canon_lore',
          visibility: 'player',
          original_score: 0.8,
          adjusted_score: 0.84,
          applied_boosts: { recency: 0.04 },
        },
      ],
      rejected: [],
    };
    const shaped = formatRetrievalDiagnostics(retrieval);
    expect(shaped.selected[0].boosts).toEqual(['recency 0.040']);
    expect(shaped.query).toBe('q');
  });
});
