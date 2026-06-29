import { aggregateTurns } from './analytics-model';
import type { TurnDetailResponse } from './models';

function turn(partial: Partial<TurnDetailResponse> & { turn_index: number }): TurnDetailResponse {
  return {
    turn_index: partial.turn_index,
    scene_id: 's',
    persona_id: 'p',
    user_message: 'u',
    assistant_message: 'a',
    route: { provider: 'local', model: 'm', reason: 'r' },
    created_at: '',
    finish_reason: 'stop',
    memory_written: partial.memory_written ?? false,
    critic_status: 'accepted',
    warnings: partial.warnings ?? [],
    errors: partial.errors ?? [],
    stage_timings: partial.stage_timings ?? {},
    retrieval: partial.retrieval ?? null,
  };
}

describe('aggregateTurns', () => {
  it('averages stage timings, tracks totals and cumulative memory', () => {
    const summary = aggregateTurns([
      turn({
        turn_index: 1,
        stage_timings: { retrieval: 1, generation: 3 },
        memory_written: true,
        warnings: ['w'],
      }),
      turn({
        turn_index: 0, // out of order on purpose — aggregate sorts
        stage_timings: { retrieval: 3, generation: 5 },
        retrieval: { query: 'q', selected: [{} as never, {} as never], rejected: [] },
      }),
    ]);

    expect(summary.turnCount).toBe(2);
    // sorted by index: turn 0 then turn 1. Totals: 8s then 4s.
    expect(summary.perTurnTotalSeconds).toEqual([8, 4]);
    expect(summary.retrievedPerTurn).toEqual([2, 0]);
    // avg retrieval = (3+1)/2 = 2, generation = (5+3)/2 = 4; sorted desc by value.
    expect(summary.avgStageSeconds).toEqual([
      { name: 'generation', value: 4 },
      { name: 'retrieval', value: 2 },
    ]);
    expect(summary.cumulativeMemories).toEqual([0, 1]);
    expect(summary.memoriesWritten).toBe(1);
    expect(summary.warningCount).toBe(1);
  });

  it('tallies errors by category, sorted desc', () => {
    const summary = aggregateTurns([
      turn({ turn_index: 0, errors: [{ category: 'degraded', stage: 'retrieval', message: 'm', suggestion: null }] }),
      turn({
        turn_index: 1,
        errors: [
          { category: 'degraded', stage: 'generation', message: 'm', suggestion: null },
          { category: 'security', stage: 'critique', message: 'm', suggestion: null },
        ],
      }),
    ]);
    expect(summary.errorsByCategory).toEqual([
      { name: 'degraded', value: 2 },
      { name: 'security', value: 1 },
    ]);
  });

  it('handles an empty session', () => {
    const summary = aggregateTurns([]);
    expect(summary.turnCount).toBe(0);
    expect(summary.avgStageSeconds).toEqual([]);
    expect(summary.errorsByCategory).toEqual([]);
  });
});
