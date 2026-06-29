import type { TurnDetailResponse } from './models';

// Pure aggregation of a session's turn diagnostics into dashboard series. Derived entirely from
// fields the API already returns (stage_timings, retrieval, memory_written, warnings, errors) —
// there is no server-side "recall score", so retrieval activity is reported as selected-chunk
// count per turn, not a recall metric.

export interface NamedValue {
  name: string;
  value: number;
}

export interface AnalyticsSummary {
  turnCount: number;
  avgStageSeconds: NamedValue[]; // per stage, sorted desc
  perTurnTotalSeconds: number[];
  retrievedPerTurn: number[]; // selected chunks per turn
  cumulativeMemories: number[];
  memoriesWritten: number;
  warningCount: number;
  errorsByCategory: NamedValue[]; // sorted desc
}

export function aggregateTurns(turns: TurnDetailResponse[]): AnalyticsSummary {
  const sorted = [...turns].sort((a, b) => a.turn_index - b.turn_index);
  const stageTotals = new Map<string, number>();
  const stageCounts = new Map<string, number>();
  const errorsByCategoryMap = new Map<string, number>();
  const perTurnTotalSeconds: number[] = [];
  const retrievedPerTurn: number[] = [];
  const cumulativeMemories: number[] = [];
  let memoriesWritten = 0;
  let warningCount = 0;

  for (const turn of sorted) {
    let total = 0;
    for (const [stage, seconds] of Object.entries(turn.stage_timings ?? {})) {
      stageTotals.set(stage, (stageTotals.get(stage) ?? 0) + seconds);
      stageCounts.set(stage, (stageCounts.get(stage) ?? 0) + 1);
      total += seconds;
    }
    perTurnTotalSeconds.push(total);
    retrievedPerTurn.push(turn.retrieval?.selected?.length ?? 0);
    if (turn.memory_written) memoriesWritten += 1;
    cumulativeMemories.push(memoriesWritten);
    warningCount += turn.warnings?.length ?? 0;
    for (const error of turn.errors ?? []) {
      errorsByCategoryMap.set(error.category, (errorsByCategoryMap.get(error.category) ?? 0) + 1);
    }
  }

  const avgStageSeconds: NamedValue[] = [...stageTotals.entries()]
    .map(([name, total]) => ({ name, value: total / (stageCounts.get(name) || 1) }))
    .sort((a, b) => b.value - a.value);

  const errorsByCategory: NamedValue[] = [...errorsByCategoryMap.entries()]
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => b.value - a.value);

  return {
    turnCount: sorted.length,
    avgStageSeconds,
    perTurnTotalSeconds,
    retrievedPerTurn,
    cumulativeMemories,
    memoriesWritten,
    warningCount,
    errorsByCategory,
  };
}
