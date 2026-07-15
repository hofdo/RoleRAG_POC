// Pure view-model helpers for the live RAG debug panel (Play page, #85). No framework deps —
// mirrors the play-model.ts convention so this stays directly unit-testable.

import type { RetrievalCandidate, TurnResult } from './models';
import { formatStageTimings } from './play-model';

export interface DebugChunkRow {
  id: string | null;
  collection: string;
  source: string;
  visibility: string;
  rank: number | null;
  originalScore: number;
  adjustedScore: number;
  boosts: string[];
  sliceLabel: string | null;
  hidden: boolean;
}

export interface DebugView {
  query: string | null;
  selected: DebugChunkRow[];
  rejected: DebugChunkRow[];
  stageTimings: string;
  memoryWritten: boolean;
  criticStatus: string | null;
  warnings: string[];
  route: string | null;
}

// `+0.080` / `-0.020`, 3 decimals, zero-weight boosts are noise and are dropped.
function formatBoost(name: string, weight: number): string | null {
  if (weight === 0) return null;
  const sign = weight > 0 ? '+' : '-';
  return `${name} ${sign}${Math.abs(weight).toFixed(3)}`;
}

function formatSliceLabel(candidate: RetrievalCandidate): string | null {
  if (!candidate.slice_guaranteed && candidate.slice_score == null) {
    return null;
  }
  const score = (candidate.slice_score ?? 0).toFixed(2);
  const terms = candidate.slice_matched_terms?.length
    ? ` (terms: ${candidate.slice_matched_terms.join(', ')})`
    : '';
  return `lexical slice ${score}${terms}`;
}

function toRow(candidate: RetrievalCandidate): DebugChunkRow {
  return {
    id: candidate.id ?? null,
    collection: candidate.collection,
    source: candidate.source,
    visibility: candidate.visibility,
    rank: candidate.selected_rank ?? null,
    originalScore: candidate.original_score,
    adjustedScore: candidate.adjusted_score,
    boosts: Object.entries(candidate.applied_boosts ?? {})
      .map(([name, weight]) => formatBoost(name, weight))
      .filter((entry): entry is string => entry !== null),
    sliceLabel: formatSliceLabel(candidate),
    hidden: candidate.visibility !== 'player',
  };
}

// Builds the panel's view whenever a turn exists, even one whose retrieval is missing (a
// failed/short-circuited turn) — the panel still has timings/route/warnings worth showing.
export function buildDebugView(turn: TurnResult | null): DebugView | null {
  if (!turn) return null;
  const retrieval = turn.retrieval;
  return {
    query: retrieval?.query ?? null,
    selected: (retrieval?.selected ?? []).map(toRow),
    rejected: (retrieval?.rejected ?? []).map(toRow),
    stageTimings: formatStageTimings(turn.stage_timings),
    memoryWritten: turn.memory_written,
    criticStatus: turn.critic_status ?? null,
    warnings: turn.warnings ?? [],
    route: turn.route ? `${turn.route.provider}/${turn.route.model}` : null,
  };
}

// Memory ids present after the turn that weren't present before it — "written this turn".
export function newMemoryIds(before: ReadonlySet<string>, after: { id: string }[]): string[] {
  return after.filter((memory) => !before.has(memory.id)).map((memory) => memory.id);
}
