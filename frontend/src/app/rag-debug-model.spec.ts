import { buildDebugView, newMemoryIds } from './rag-debug-model';
import type { RetrievalCandidate, TurnResult } from './models';

// Pure view-model — see play-model.spec.ts for the sibling convention this mirrors.

function candidate(overrides: Partial<RetrievalCandidate> = {}): RetrievalCandidate {
  return {
    selected_rank: 1,
    source: 'scene:rose-gallery',
    collection: 'scenes',
    visibility: 'player',
    original_score: 0.5,
    adjusted_score: 0.58,
    applied_boosts: {},
    ...overrides,
  };
}

function fullTurn(overrides: Partial<TurnResult> = {}): TurnResult {
  return {
    status: 'completed',
    text: 'a reply',
    route: { provider: 'local', model: 'gemma-27b', reason: 'session-bound' },
    finish_reason: 'stop',
    memory_written: true,
    critic_status: 'accepted',
    warnings: [],
    retrieval: {
      query: 'what did the archivist say about dawn',
      selected: [
        candidate({
          id: 'c1',
          selected_rank: 1,
          applied_boosts: { recency: 0.08, canon: -0.02, novelty: 0 },
          slice_score: 1.42,
          slice_matched_terms: ['dawn', 'promise'],
        }),
      ],
      rejected: [candidate({ id: 'c2', selected_rank: null, visibility: 'gm' })],
    },
    stage_timings: { retrieval: 0.2, generation: 1.1 },
    ...overrides,
  };
}

describe('rag-debug-model buildDebugView', () => {
  it('returns null when there is no turn', () => {
    expect(buildDebugView(null)).toBeNull();
  });

  it('builds a full view from a turn with retrieval diagnostics', () => {
    const view = buildDebugView(fullTurn());
    expect(view).not.toBeNull();
    expect(view!.query).toBe('what did the archivist say about dawn');
    expect(view!.route).toBe('local/gemma-27b');
    expect(view!.memoryWritten).toBe(true);
    expect(view!.criticStatus).toBe('accepted');
    expect(view!.stageTimings).toBe('retrieval 0.2s; generation 1.1s');
    expect(view!.selected.length).toBe(1);
    expect(view!.rejected.length).toBe(1);
  });

  it('marks non-player visibility as hidden', () => {
    const view = buildDebugView(fullTurn())!;
    expect(view.selected[0].hidden).toBe(false);
    expect(view.rejected[0].hidden).toBe(true);
  });

  it('formats boosts with signed 3-decimal values and skips zero weights', () => {
    const view = buildDebugView(fullTurn())!;
    expect(view.selected[0].boosts).toEqual(['recency +0.080', 'canon -0.020']);
  });

  it('formats a slice label when slice fields are present', () => {
    const view = buildDebugView(fullTurn())!;
    expect(view.selected[0].sliceLabel).toBe('lexical slice 1.42 (terms: dawn, promise)');
  });

  it('leaves sliceLabel null when no slice fields are present', () => {
    const view = buildDebugView(fullTurn())!;
    expect(view.rejected[0].sliceLabel).toBeNull();
  });

  it('still builds a view for a turn with no retrieval (e.g. a failed turn)', () => {
    const view = buildDebugView(fullTurn({ retrieval: null, memory_written: false }));
    expect(view).not.toBeNull();
    expect(view!.query).toBeNull();
    expect(view!.selected).toEqual([]);
    expect(view!.rejected).toEqual([]);
    expect(view!.stageTimings).toBe('retrieval 0.2s; generation 1.1s');
  });

  it('leaves route null when the turn has no route', () => {
    const view = buildDebugView(fullTurn({ route: null }));
    expect(view!.route).toBeNull();
  });
});

describe('rag-debug-model newMemoryIds', () => {
  it('returns memory ids that were not present before the turn', () => {
    const before = new Set(['m1', 'm2']);
    const after = [{ id: 'm1' }, { id: 'm2' }, { id: 'm3' }];
    expect(newMemoryIds(before, after)).toEqual(['m3']);
  });

  it('returns an empty list when nothing new was written', () => {
    const before = new Set(['m1']);
    expect(newMemoryIds(before, [{ id: 'm1' }])).toEqual([]);
  });

  it('returns everything when starting from an empty snapshot', () => {
    expect(newMemoryIds(new Set(), [{ id: 'm1' }, { id: 'm2' }])).toEqual(['m1', 'm2']);
  });
});
