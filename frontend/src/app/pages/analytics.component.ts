import { Component, computed, inject, OnInit, signal } from '@angular/core';
import { ApiService } from '../api.service';
import { aggregateTurns, type AnalyticsSummary } from '../analytics-model';
import { formatRecentSessionOption } from '../play-model';
import type { RecentSessionResponse } from '../models';

// Phase 3: aggregate a session's turn diagnostics into latency / retrieval / memory / error
// series. Loops GET /sessions/{id}/turns/{i} per Phase-0 decision (bulk endpoint only if slow).
// Charts are plain CSS bars — ponytail: no chart lib for simple proportional bars.
@Component({
  selector: 'app-analytics',
  imports: [],
  template: `
    <section class="analytics">
      <header class="analytics__head">
        <h1>Analytics</h1>
        <label>
          Session
          <select #sel (change)="selectSession(sel.value)">
            <option value="">— pick a session —</option>
            @for (s of sessions(); track s.session_id) {
              <option [value]="s.session_id">{{ label(s) }}</option>
            }
          </select>
        </label>
        @if (error()) { <p class="error">{{ error() }}</p> }
        @if (loading()) { <p class="hint">Loading turn diagnostics…</p> }
      </header>

      @if (summary(); as sum) {
        <p class="note">Based on {{ sum.turnCount }} recent turn(s).</p>

        <div class="cards">
          <div class="card"><span class="num">{{ sum.turnCount }}</span><span class="lbl">turns</span></div>
          <div class="card"><span class="num">{{ avgTotal(sum) }}s</span><span class="lbl">avg total latency</span></div>
          <div class="card"><span class="num">{{ sum.memoriesWritten }}</span><span class="lbl">memories written</span></div>
          <div class="card"><span class="num">{{ sum.warningCount }}</span><span class="lbl">warnings</span></div>
          <div class="card"><span class="num">{{ totalErrors(sum) }}</span><span class="lbl">errors</span></div>
        </div>

        <h2 class="section-label">Avg latency per stage (s)</h2>
        @for (v of sum.avgStageSeconds; track v.name) {
          <div class="bar-row">
            <span class="bar-label">{{ v.name }}</span>
            <span class="bar"><span class="bar-fill" [style.width.%]="pct(v.value, maxStage())"></span></span>
            <span class="bar-val">{{ v.value.toFixed(2) }}</span>
          </div>
        }
        @if (!sum.avgStageSeconds.length) { <p class="hint">No stage timings recorded.</p> }

        <h2 class="section-label">Total latency per turn (s)</h2>
        @for (t of sum.perTurnTotalSeconds; track $index) {
          <div class="bar-row">
            <span class="bar-label">#{{ $index }}</span>
            <span class="bar"><span class="bar-fill" [style.width.%]="pct(t, maxTurnTotal())"></span></span>
            <span class="bar-val">{{ t.toFixed(1) }}</span>
          </div>
        }

        <h2 class="section-label">Error categories</h2>
        @for (v of sum.errorsByCategory; track v.name) {
          <div class="bar-row">
            <span class="bar-label">{{ v.name }}</span>
            <span class="bar"><span class="bar-fill err" [style.width.%]="pct(v.value, maxError())"></span></span>
            <span class="bar-val">{{ v.value }}</span>
          </div>
        }
        @if (!sum.errorsByCategory.length) { <p class="hint">No errors recorded. 🎉</p> }
      } @else if (!loading()) {
        <p class="hint">Pick a session to see its analytics.</p>
      }
    </section>
  `,
  styles: [
    `
      /* Grimoire Console styling */
      .analytics__head { display: flex; align-items: baseline; gap: 1.5rem; flex-wrap: wrap; }
      .analytics__head label { display: inline-flex; gap: 0.4rem; align-items: center; }
      .error { color: var(--danger); }
      .hint { color: var(--muted); font-style: italic; }
      .note { color: var(--muted); font-size: 0.85rem; }
      .cards { display: flex; gap: 1rem; flex-wrap: wrap; margin: 0.5rem 0 1rem; }
      .card { display: flex; flex-direction: column; padding: 0.6rem 1rem; background: var(--surface); border: 1px solid var(--border); border-radius: var(--r-card); min-width: 6rem; }
      .card .num { font-size: 1.4rem; font-weight: 700; color: var(--accent); }
      .card .lbl { font-size: 0.75rem; color: var(--muted); }
      .bar-row { display: grid; grid-template-columns: 8rem 1fr 3rem; gap: 0.5rem; align-items: center; margin: 0.2rem 0; font-size: 0.85rem; }
      .bar-label { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--muted); }
      .bar { background: color-mix(in srgb, var(--border) 50%, transparent); border-radius: var(--r-chip); height: 0.9rem; }
      .bar-fill { display: block; height: 100%; background: var(--accent); border-radius: var(--r-chip); }
      .bar-fill.err { background: var(--danger); }
      .bar-val { text-align: right; font-variant-numeric: tabular-nums; }
    `,
  ],
})
export class AnalyticsComponent implements OnInit {
  private readonly api = inject(ApiService);

  readonly sessions = signal<RecentSessionResponse[]>([]);
  readonly summary = signal<AnalyticsSummary | null>(null);
  readonly loading = signal<boolean>(false);
  readonly error = signal<string | null>(null);

  readonly maxStage = computed(() =>
    Math.max(0, ...(this.summary()?.avgStageSeconds ?? []).map((v) => v.value)),
  );
  readonly maxTurnTotal = computed(() =>
    Math.max(0, ...(this.summary()?.perTurnTotalSeconds ?? [])),
  );
  readonly maxError = computed(() =>
    Math.max(0, ...(this.summary()?.errorsByCategory ?? []).map((v) => v.value)),
  );

  ngOnInit(): void {
    void this.loadSessions();
  }

  label(session: RecentSessionResponse): string {
    return formatRecentSessionOption(session);
  }

  pct(value: number, max: number): number {
    return max > 0 ? (value / max) * 100 : 0;
  }

  avgTotal(sum: AnalyticsSummary): string {
    if (!sum.perTurnTotalSeconds.length) return '0.0';
    const total = sum.perTurnTotalSeconds.reduce((a, b) => a + b, 0);
    return (total / sum.perTurnTotalSeconds.length).toFixed(1);
  }

  totalErrors(sum: AnalyticsSummary): number {
    return sum.errorsByCategory.reduce((a, b) => a + b.value, 0);
  }

  private async loadSessions(): Promise<void> {
    try {
      const response = await this.api.getRecentSessions();
      this.sessions.set(response.sessions);
    } catch (error) {
      this.error.set(error instanceof Error ? error.message : 'Failed to load sessions.');
    }
  }

  async selectSession(sessionId: string): Promise<void> {
    this.summary.set(null);
    this.error.set(null);
    if (!sessionId) return;
    this.loading.set(true);
    try {
      const detail = await this.api.getSession(sessionId);
      const turns = await Promise.all(
        detail.recent_turns.map((t) => this.api.getTurnDetail(sessionId, t.turn_index)),
      );
      this.summary.set(aggregateTurns(turns));
    } catch (error) {
      this.error.set(error instanceof Error ? error.message : 'Failed to load analytics.');
    } finally {
      this.loading.set(false);
    }
  }
}
