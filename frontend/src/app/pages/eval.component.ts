import { Component, computed, inject, OnInit, signal } from '@angular/core';
import { ApiService } from '../api.service';
import { formatStageTimings } from '../play-model';
import type { EvalRunDetail, EvalRunSummary } from '../models';

// Phase 4: trend recall / latency / pass-fail across eval runs. Reads the backend
// GET /diagnostics/eval-runs endpoint, which scans EVAL_RESULTS_DIR for per-run
// conversation-checkpoint.json artifacts. CSS bars, no chart lib (ponytail).
@Component({
  selector: 'app-eval',
  imports: [],
  template: `
    <section class="eval">
      <h1>Eval</h1>
      @if (error()) { <p class="error">{{ error() }}</p> }
      @if (loading()) { <p class="hint">Loading eval runs…</p> }

      <p class="note">
        Results dir: <code>{{ resultsDir() }}</code>
        @if (runs().length === 0 && !loading()) {
          — no runs found. Point <code>EVAL_RESULTS_DIR</code> at a folder of run subdirs each
          holding <code>conversation-checkpoint.json</code> (or <code>raw/…</code>).
        }
      </p>

      @if (runs().length) {
        <table class="runs runs--list">
          <thead>
            <tr>
              <th>Run</th><th>Status</th><th>Turns</th>
              <th>Recall miss</th><th>Extract miss</th><th>Retr miss</th>
              <th>p50 s</th><th>p95 s</th><th>Total s</th><th>Warn</th>
            </tr>
          </thead>
          <tbody>
            @for (run of runs(); track run.id) {
              <tr
                [class.fail]="run.status !== 'pass'"
                [class.is-live]="selectedRunId() === run.id"
                (click)="openRun(run.id)"
              >
                <td>{{ run.id }}</td>
                <td>{{ run.status }}</td>
                <td>{{ run.turn_count }}</td>
                <td>{{ run.recall_misses }}</td>
                <td>{{ run.extraction_misses }}</td>
                <td>{{ run.retrieval_misses }}</td>
                <td>{{ run.p50_seconds.toFixed(2) }}</td>
                <td>{{ run.p95_seconds.toFixed(2) }}</td>
                <td>{{ run.total_seconds.toFixed(1) }}</td>
                <td>{{ run.warning_total }}</td>
              </tr>
            }
          </tbody>
        </table>

        <h2>Recall misses per run</h2>
        @for (run of runs(); track run.id) {
          <div class="bar-row">
            <span class="bar-label">{{ run.id }}</span>
            <span class="bar"><span class="bar-fill err" [style.width.%]="pct(run.recall_misses, maxRecall())"></span></span>
            <span class="bar-val">{{ run.recall_misses }}</span>
          </div>
        }

        <h2>p95 latency per run (s)</h2>
        @for (run of runs(); track run.id) {
          <div class="bar-row">
            <span class="bar-label">{{ run.id }}</span>
            <span class="bar"><span class="bar-fill" [style.width.%]="pct(run.p95_seconds, maxP95())"></span></span>
            <span class="bar-val">{{ run.p95_seconds.toFixed(1) }}</span>
          </div>
        }

        @if (selectedRunId()) {
          <div class="run-detail">
            <header class="run-detail__head">
              <h2 class="section-label">Run {{ selectedRunId() }}</h2>
              <button type="button" class="ghost" (click)="closeRun()">Close</button>
            </header>
            @if (loadingDetail()) {
              <p class="hint">Loading run…</p>
            }
            @if (selectedRun(); as run) {
              <dl class="meta">
                <div><dt>Status</dt><dd>{{ run.status ?? '—' }}</dd></div>
                <div>
                  <dt>Latency p50/p95/total (s)</dt>
                  <dd>
                    {{ run.quality_metrics?.latency?.p50_seconds ?? 0 }} /
                    {{ run.quality_metrics?.latency?.p95_seconds ?? 0 }} /
                    {{ run.quality_metrics?.latency?.total_seconds ?? 0 }}
                  </dd>
                </div>
                <div>
                  <dt>Recall / extract / retr misses</dt>
                  <dd>
                    {{ run.quality_metrics?.callback_recall_misses ?? 0 }} /
                    {{ run.quality_metrics?.memory_extraction_misses ?? 0 }} /
                    {{ run.quality_metrics?.retrieval_selection_misses ?? 0 }}
                  </dd>
                </div>
              </dl>

              @if (warningEntries(run.warning_counts).length) {
                <h3 class="section-label">Warning counts</h3>
                <ul class="warnings">
                  @for (w of warningEntries(run.warning_counts); track w.name) {
                    <li>{{ w.name }}: {{ w.value }}</li>
                  }
                </ul>
              }

              @if (run.turns?.length) {
                <h3 class="section-label">Per-turn</h3>
                <table class="runs">
                  <thead>
                    <tr><th>#</th><th>Dur s</th><th>Finish</th><th>Warn</th><th>Stages</th></tr>
                  </thead>
                  <tbody>
                    @for (t of run.turns ?? []; track $index) {
                      <tr>
                        <td>{{ t.turn_index }}</td>
                        <td>{{ (t.duration_seconds ?? 0).toFixed(1) }}</td>
                        <td>{{ t.finish_reason ?? '—' }}</td>
                        <td>{{ t.warnings?.length ?? 0 }}</td>
                        <td class="stages">{{ stageTimings(t.stage_timings) }}</td>
                      </tr>
                    }
                  </tbody>
                </table>
              }
            }
          </div>
        }
      }
    </section>
  `,
  styles: [
    `
      /* Grimoire Console styling */
      .error { color: var(--danger); }
      .hint, .note { color: var(--muted); font-size: 0.85rem; }
      .note code { font-family: var(--font-mono); }
      .runs {
        border-collapse: collapse;
        margin: 0.5rem 0 1rem;
        font-size: 0.85rem;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--r-card);
        box-shadow: var(--shadow-card);
        overflow: hidden;
      }
      .runs td { padding: 0.25rem 0.6rem; text-align: right; border-bottom: 1px solid var(--border); }
      .runs th { padding: 0.25rem 0.6rem; text-align: right; }
      .runs th:first-child, .runs td:first-child { text-align: left; }
      .runs tr.fail td { color: var(--danger); }
      .bar-row { display: grid; grid-template-columns: 12rem 1fr 3rem; gap: 0.5rem; align-items: center; margin: 0.2rem 0; font-size: 0.85rem; }
      .bar-label { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--muted); }
      .bar { background: var(--surface-2); border-radius: var(--r-chip); height: 0.9rem; }
      .bar-fill { display: block; height: 100%; background: var(--accent); border-radius: var(--r-chip); }
      .bar-fill.err { background: var(--danger); }
      .bar-val { text-align: right; font-variant-numeric: tabular-nums; }
      .runs--list tbody tr { cursor: pointer; }
      .run-detail { margin-top: 1.5rem; background: var(--surface); border: 1px solid var(--border); border-radius: var(--r-card); padding: 1rem; box-shadow: var(--shadow-card); }
      .run-detail__head { display: flex; align-items: baseline; justify-content: space-between; gap: 1rem; }
      .meta div { display: flex; gap: 0.5rem; margin: 0.15rem 0; font-size: 0.85rem; }
      .meta dt { width: 16rem; color: var(--muted); }
      .meta dd { margin: 0; }
      .warnings { margin: 0.25rem 0; padding-left: 1.25rem; font-size: 0.85rem; }
      .stages { font-family: var(--font-mono); font-size: 0.72rem; text-align: left; }
    `,
  ],
})
export class EvalComponent implements OnInit {
  private readonly api = inject(ApiService);

  readonly runs = signal<EvalRunSummary[]>([]);
  readonly resultsDir = signal<string>('');
  readonly loading = signal<boolean>(false);
  readonly error = signal<string | null>(null);

  readonly selectedRunId = signal<string | null>(null);
  readonly selectedRun = signal<EvalRunDetail | null>(null);
  readonly loadingDetail = signal<boolean>(false);

  readonly maxRecall = computed(() => Math.max(0, ...this.runs().map((r) => r.recall_misses)));
  readonly maxP95 = computed(() => Math.max(0, ...this.runs().map((r) => r.p95_seconds)));

  ngOnInit(): void {
    void this.load();
  }

  pct(value: number, max: number): number {
    return max > 0 ? (value / max) * 100 : 0;
  }

  async openRun(runId: string): Promise<void> {
    this.selectedRunId.set(runId);
    this.selectedRun.set(null);
    this.loadingDetail.set(true);
    try {
      this.selectedRun.set(await this.api.getEvalRun(runId));
    } catch (error) {
      this.error.set(error instanceof Error ? error.message : 'Failed to load run detail.');
    } finally {
      this.loadingDetail.set(false);
    }
  }

  closeRun(): void {
    this.selectedRunId.set(null);
    this.selectedRun.set(null);
  }

  stageTimings(timings: Record<string, number> | undefined): string {
    return formatStageTimings(timings);
  }

  warningEntries(counts: Record<string, number> | undefined): { name: string; value: number }[] {
    return Object.entries(counts ?? {}).map(([name, value]) => ({ name, value }));
  }

  private async load(): Promise<void> {
    this.loading.set(true);
    this.error.set(null);
    try {
      const response = await this.api.getEvalRuns();
      this.resultsDir.set(response.results_dir);
      this.runs.set(response.runs);
    } catch (error) {
      this.error.set(error instanceof Error ? error.message : 'Failed to load eval runs.');
    } finally {
      this.loading.set(false);
    }
  }
}
