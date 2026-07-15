import {
  buildScatterSeries,
  escapeHtml,
  filterPoints,
  highlightIdsFromTurn,
  tooltipLines,
  type MapFilters,
} from './vector-map-model';
import type { RagMapPoint, TurnResult } from './models';

function point(partial: Partial<RagMapPoint> & { id: string }): RagMapPoint {
  return {
    id: partial.id,
    collection: partial.collection ?? 'canon_lore',
    x: partial.x ?? 0,
    y: partial.y ?? 0,
    source: partial.source ?? 'src',
    source_type: partial.source_type ?? 'lore',
    visibility: partial.visibility ?? 'player',
    tags: partial.tags ?? [],
    world_id: partial.world_id ?? null,
    scene_id: partial.scene_id ?? null,
    persona_id: partial.persona_id ?? null,
    session_id: partial.session_id ?? null,
    actor_id: partial.actor_id ?? null,
    importance: partial.importance ?? null,
    created_at: partial.created_at ?? null,
    text_preview: partial.text_preview ?? 'preview text',
    text_redacted: partial.text_redacted ?? false,
  };
}

function emptyFilters(): MapFilters {
  return { collections: new Set(), visibilities: new Set(), sessionId: null, tag: null };
}

describe('filterPoints', () => {
  const points = [
    point({ id: 'a', collection: 'canon_lore', visibility: 'player', session_id: 's1', tags: ['x'] }),
    point({ id: 'b', collection: 'session_memory', visibility: 'gm', session_id: 's2', tags: ['y'] }),
    point({
      id: 'c',
      collection: 'persona_memory',
      visibility: 'character_private',
      session_id: 's1',
      tags: ['x', 'y'],
    }),
  ];

  it('returns all points when filters are empty', () => {
    expect(filterPoints(points, emptyFilters())).toEqual(points);
  });

  it('filters by collection set', () => {
    const filters = { ...emptyFilters(), collections: new Set(['canon_lore', 'persona_memory']) };
    expect(filterPoints(points, filters).map((p) => p.id)).toEqual(['a', 'c']);
  });

  it('filters by visibility set', () => {
    const filters = { ...emptyFilters(), visibilities: new Set(['gm']) };
    expect(filterPoints(points, filters).map((p) => p.id)).toEqual(['b']);
  });

  it('filters by exact sessionId', () => {
    const filters = { ...emptyFilters(), sessionId: 's1' };
    expect(filterPoints(points, filters).map((p) => p.id)).toEqual(['a', 'c']);
  });

  it('filters by tag membership', () => {
    const filters = { ...emptyFilters(), tag: 'y' };
    expect(filterPoints(points, filters).map((p) => p.id)).toEqual(['b', 'c']);
  });

  it('combines multiple filter dimensions (AND)', () => {
    const filters: MapFilters = {
      collections: new Set(['persona_memory']),
      visibilities: new Set(['character_private']),
      sessionId: 's1',
      tag: 'x',
    };
    expect(filterPoints(points, filters).map((p) => p.id)).toEqual(['c']);
  });

  it('excludes everything when a filter dimension matches nothing', () => {
    const filters = { ...emptyFilters(), tag: 'zzz' };
    expect(filterPoints(points, filters)).toEqual([]);
  });
});

describe('highlightIdsFromTurn', () => {
  it('is null-safe', () => {
    expect(highlightIdsFromTurn(null)).toEqual({ selected: new Set(), rejected: new Set() });
  });

  it('is safe when the turn has no retrieval diagnostics', () => {
    const turn = { retrieval: null } as unknown as TurnResult;
    expect(highlightIdsFromTurn(turn)).toEqual({ selected: new Set(), rejected: new Set() });
  });

  it('extracts selected and rejected candidate ids', () => {
    const turn = {
      retrieval: {
        query: 'q',
        selected: [
          { id: 'a', source: 's', collection: 'c', visibility: 'player', original_score: 0, adjusted_score: 0, applied_boosts: {} },
          { source: 's', collection: 'c', visibility: 'player', original_score: 0, adjusted_score: 0, applied_boosts: {} }, // no id
        ],
        rejected: [
          { id: 'b', source: 's', collection: 'c', visibility: 'player', original_score: 0, adjusted_score: 0, applied_boosts: {} },
        ],
      },
    } as unknown as TurnResult;
    const result = highlightIdsFromTurn(turn);
    expect(result.selected).toEqual(new Set(['a']));
    expect(result.rejected).toEqual(new Set(['b']));
  });
});

describe('buildScatterSeries', () => {
  const noHighlights = { selected: new Set<string>(), rejected: new Set<string>() };

  it('splits points into one series per collection', () => {
    const points = [
      point({ id: 'a', collection: 'canon_lore', x: 1, y: 2 }),
      point({ id: 'b', collection: 'session_memory', x: 3, y: 4 }),
    ];
    const series = buildScatterSeries(points, noHighlights, null);
    expect(series.map((s) => s.name)).toEqual(['Canon lore', 'Session memory']);
    expect(series[0].data).toEqual([{ value: [1, 2], pointId: 'a' }]);
    expect(series[0].symbolSize).toBe(7);
  });

  it('adds a selected-highlight series on top without removing the base point', () => {
    const points = [point({ id: 'a', collection: 'canon_lore', x: 1, y: 2 })];
    const highlights = { selected: new Set(['a']), rejected: new Set<string>() };
    const series = buildScatterSeries(points, highlights, null);
    const names = series.map((s) => s.name);
    expect(names).toContain('Canon lore');
    expect(names).toContain('Last turn: selected');
    const highlightSeries = series.find((s) => s.name === 'Last turn: selected')!;
    expect(highlightSeries.data).toEqual([{ value: [1, 2], pointId: 'a' }]);
    expect(highlightSeries.symbolSize).toBe(14);
    expect(highlightSeries.symbol).toBe('diamond');
  });

  it('adds a rejected-highlight series', () => {
    const points = [point({ id: 'a', collection: 'canon_lore', x: 1, y: 2 })];
    const highlights = { selected: new Set<string>(), rejected: new Set(['a']) };
    const series = buildScatterSeries(points, highlights, null);
    const rejectedSeries = series.find((s) => s.name === 'Last turn: rejected')!;
    expect(rejectedSeries.data).toEqual([{ value: [1, 2], pointId: 'a' }]);
    expect(rejectedSeries.symbol).toBe('triangle');
  });

  it('omits highlight series when nothing is highlighted', () => {
    const points = [point({ id: 'a' })];
    const series = buildScatterSeries(points, noHighlights, null);
    expect(series.some((s) => s.name === 'Last turn: selected')).toBeFalse();
    expect(series.some((s) => s.name === 'Last turn: rejected')).toBeFalse();
  });

  it('adds a query series only when a query point is present', () => {
    const points = [point({ id: 'a' })];
    expect(buildScatterSeries(points, noHighlights, null).some((s) => s.name === 'Query')).toBeFalse();
    const withQuery = buildScatterSeries(points, noHighlights, [5, 6]);
    const querySeries = withQuery.find((s) => s.name === 'Query')!;
    expect(querySeries.data).toEqual([{ value: [5, 6], pointId: null }]);
    expect(querySeries.symbolSize).toBe(20);
  });

  it('returns no series for an empty point list and no query', () => {
    expect(buildScatterSeries([], noHighlights, null)).toEqual([]);
  });
});

describe('escapeHtml', () => {
  it('escapes the five reserved HTML characters', () => {
    expect(escapeHtml(`<a href="x">it's & "that"</a>`)).toBe(
      '&lt;a href=&quot;x&quot;&gt;it&#39;s &amp; &quot;that&quot;&lt;/a&gt;',
    );
  });

  it('leaves plain text untouched', () => {
    expect(escapeHtml('hello world')).toBe('hello world');
  });
});

describe('tooltipLines', () => {
  it('formats a full point', () => {
    const p = point({
      id: 'a1',
      collection: 'session_memory',
      visibility: 'gm',
      source: 'turn-3',
      tags: ['combat', 'npc'],
      importance: 0.8,
      text_preview: 'The dragon roared.',
      text_redacted: false,
    });
    expect(tooltipLines(p)).toEqual([
      'id: a1',
      'collection: Session memory',
      'visibility: gm',
      'source: turn-3',
      'tags: combat, npc',
      'importance: 0.8',
      'preview: The dragon roared.',
    ]);
  });

  it('shows [hidden] for redacted text and — for empty tags/importance', () => {
    const p = point({
      id: 'a2',
      tags: [],
      importance: null,
      text_preview: null,
      text_redacted: true,
    });
    const lines = tooltipLines(p);
    expect(lines).toContain('tags: —');
    expect(lines).toContain('importance: —');
    expect(lines).toContain('preview: [hidden]');
  });
});
