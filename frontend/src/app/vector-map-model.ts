import type { RagMapPoint, TurnResult } from './models';

// Pure model for the Vector Map page (backlog #85): a 2D PCA scatter of every point in the
// vector store (canon lore / session memory / persona memory), colored by collection, with
// optional highlighting of the last turn's retrieval outcome. No Angular or echarts imports
// here on purpose — this file returns plain, chart-library-agnostic data shapes so it stays
// unit-testable without a DOM or a chart instance.

export const COLLECTION_LABELS: Record<string, string> = {
  canon_lore: 'Canon lore',
  session_memory: 'Session memory',
  persona_memory: 'Persona memory',
};

// Colorblind-safe qualitative triad (Tol/Okabe-Ito adjacent hues).
export const COLLECTION_COLORS: Record<string, string> = {
  canon_lore: '#4477aa',
  session_memory: '#ee7733',
  persona_memory: '#009988',
};

export interface MapFilters {
  collections: Set<string>;
  visibilities: Set<string>;
  sessionId: string | null;
  tag: string | null;
}

// Empty set on a dimension means "no filtering on that dimension" (show everything).
export function filterPoints(points: RagMapPoint[], filters: MapFilters): RagMapPoint[] {
  return points.filter((point) => {
    if (filters.collections.size > 0 && !filters.collections.has(point.collection)) {
      return false;
    }
    if (filters.visibilities.size > 0 && !filters.visibilities.has(point.visibility)) {
      return false;
    }
    if (filters.sessionId && point.session_id !== filters.sessionId) {
      return false;
    }
    if (filters.tag && !point.tags.includes(filters.tag)) {
      return false;
    }
    return true;
  });
}

export interface TurnHighlights {
  selected: Set<string>;
  rejected: Set<string>;
}

// Extracts the point ids the last turn's retrieval selected/rejected, so the map can highlight
// what actually made it into (or was passed over for) the most recent prompt. Candidates without
// an id (older fixtures / pre-#85 responses) are simply not highlightable.
export function highlightIdsFromTurn(turn: TurnResult | null): TurnHighlights {
  const selected = new Set<string>();
  const rejected = new Set<string>();
  const retrieval = turn?.retrieval;
  if (retrieval) {
    for (const candidate of retrieval.selected) {
      if (candidate.id) selected.add(candidate.id);
    }
    for (const candidate of retrieval.rejected) {
      if (candidate.id) rejected.add(candidate.id);
    }
  }
  return { selected, rejected };
}

export interface SeriesPoint {
  value: [number, number];
  pointId: string | null;
}

export interface SeriesSpec {
  name: string;
  color: string;
  symbolSize: number;
  symbol?: string;
  zLevel?: number;
  data: SeriesPoint[];
}

const SELECTED_SERIES_NAME = 'Last turn: selected';
const REJECTED_SERIES_NAME = 'Last turn: rejected';
const QUERY_SERIES_NAME = 'Query';

// One series per collection (base layer), plus optional highlight/query overlays drawn on top
// (higher zLevel). Highlight overlays duplicate the underlying point's coordinates rather than
// replacing the base marker, so the base collection dot stays visible underneath.
export function buildScatterSeries(
  points: RagMapPoint[],
  highlights: TurnHighlights,
  queryPoint: number[] | null,
): SeriesSpec[] {
  const byCollection = new Map<string, SeriesPoint[]>();
  const selectedData: SeriesPoint[] = [];
  const rejectedData: SeriesPoint[] = [];

  for (const point of points) {
    const entry: SeriesPoint = { value: [point.x, point.y], pointId: point.id };
    const bucket = byCollection.get(point.collection);
    if (bucket) {
      bucket.push(entry);
    } else {
      byCollection.set(point.collection, [entry]);
    }
    if (highlights.selected.has(point.id)) {
      selectedData.push(entry);
    } else if (highlights.rejected.has(point.id)) {
      rejectedData.push(entry);
    }
  }

  const series: SeriesSpec[] = [];
  for (const [collection, data] of byCollection.entries()) {
    series.push({
      name: COLLECTION_LABELS[collection] ?? collection,
      color: COLLECTION_COLORS[collection] ?? '#888888',
      symbolSize: 7,
      symbol: 'circle',
      zLevel: 1,
      data,
    });
  }

  if (rejectedData.length) {
    series.push({
      name: REJECTED_SERIES_NAME,
      color: '#bb5566',
      symbolSize: 11,
      symbol: 'triangle',
      zLevel: 2,
      data: rejectedData,
    });
  }

  if (selectedData.length) {
    series.push({
      name: SELECTED_SERIES_NAME,
      color: '#2be6d6',
      symbolSize: 14,
      symbol: 'diamond',
      zLevel: 3,
      data: selectedData,
    });
  }

  if (queryPoint && queryPoint.length === 2) {
    series.push({
      name: QUERY_SERIES_NAME,
      color: '#c9a24b',
      symbolSize: 20,
      symbol: 'path://M12,1 L15,9 L23,9 L16.5,14 L19,22 L12,17 L5,22 L7.5,14 L1,9 L9,9 Z',
      zLevel: 4,
      data: [{ value: [queryPoint[0], queryPoint[1]], pointId: null }],
    });
  }

  return series;
}

export function escapeHtml(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

// Plain-text lines for a point's tooltip; the caller escapes + joins (e.g. with <br/>).
export function tooltipLines(point: RagMapPoint): string[] {
  return [
    `id: ${point.id}`,
    `collection: ${COLLECTION_LABELS[point.collection] ?? point.collection}`,
    `visibility: ${point.visibility}`,
    `source: ${point.source}`,
    `tags: ${point.tags.length ? point.tags.join(', ') : '—'}`,
    `importance: ${point.importance ?? '—'}`,
    `preview: ${point.text_redacted ? '[hidden]' : (point.text_preview ?? '—')}`,
  ];
}
