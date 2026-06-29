import { Component, computed, inject, OnInit, signal } from '@angular/core';
import { ApiService } from '../api.service';
import { formatRetrievalDiagnostics, formatRecentSessionOption, formatStageTimings } from '../play-model';
import type { RecentSessionResponse, RecentTurnResponse, TurnDetailResponse } from '../models';

// Phase 2: per-session turn timeline → drill into one turn's retrieval ranking (selected vs
// rejected chunks, scores, applied boosts), stage timings, critic status and structured errors.
// Read-only; reuses the tested play-model formatters. State is component-local (no store needed).
@Component({
  selector: 'app-inspector',
  imports: [],
  template: `
    <section class="inspector">
      <header class="inspector__head">
        <h1>RAG Inspector</h1>
        <label>
          Session
          <select #sel (change)="selectSession(sel.value)">
            <option value="">— pick a session —</option>
            @for (s of sessions(); track s.session_id) {
              <option [value]="s.session_id" [selected]="s.session_id === selectedId()">
                {{ label(s) }}
              </option>
            }
          </select>
        </label>
        @if (error()) {
          <p class="error">{{ error() }}</p>
        }
      </header>

      <div class="cols">
        <ul class="timeline">
          @for (t of turns(); track t.turn_index) {
            <li>
              <button
                type="button"
                [class.active]="selectedIndex() === t.turn_index"
                [class.is-live]="selectedIndex() === t.turn_index"
                (click)="selectTurn(t.turn_index)"
              >
                <span class="idx">#{{ t.turn_index }}</span>
                <span class="msg">{{ t.user_message }}</span>
              </button>
            </li>
          }
          @if (selectedId() && turns().length === 0 && !loading()) {
            <li class="hint">No turns in this session.</li>
          }
          @if (!selectedId()) {
            <li class="hint">Pick a session to list its turns.</li>
          }
        </ul>

        <div class="detail">
          @if (selectedTurn(); as turn) {
            <h2>Turn #{{ turn.turn_index }}</h2>
            <dl class="meta">
              <div><dt>Route</dt><dd>{{ turn.route.provider }} / {{ turn.route.model }}</dd></div>
              <div><dt>Reason</dt><dd>{{ turn.route.reason }}</dd></div>
              <div><dt>Critic</dt><dd>{{ turn.critic_status }}</dd></div>
              <div><dt>Finish</dt><dd>{{ turn.finish_reason ?? '—' }}</dd></div>
              <div><dt>Memory written</dt><dd>{{ turn.memory_written ? 'yes' : 'no' }}</dd></div>
              <div><dt>Timings</dt><dd>{{ timings() }}</dd></div>
            </dl>

            @if (turn.warnings.length) {
              <h3 class="section-label">Warnings</h3>
              <ul class="warnings">
                @for (w of turn.warnings; track $index) { <li>{{ w }}</li> }
              </ul>
            }

            @if (turn.errors.length) {
              <h3 class="section-label">Errors</h3>
              <ul class="errors">
                @for (e of turn.errors; track $index) {
                  <li>
                    <strong>{{ e.category }}</strong> &#64; {{ e.stage }} — {{ e.message }}
                    @if (e.suggestion) { <em>({{ e.suggestion }})</em> }
                  </li>
                }
              </ul>
            }

            <h3 class="section-label">Retrieval</h3>
            @if (retrievalView().query) {
              <p class="query">Query: <code>{{ retrievalView().query }}</code></p>
            }
            @if (retrievalView().selected.length) {
              <h4 class="section-label">Selected</h4>
              @for (r of retrievalView().selected; track $index) {
                <div class="chunk">
                  <span class="rank">#{{ r.rank ?? '—' }}</span>
                  <span class="src">{{ r.source }} · {{ r.collection }} · {{ r.visibility }}</span>
                  <span class="score">{{ r.originalScore.toFixed(3) }} → {{ r.adjustedScore.toFixed(3) }}</span>
                  @if (r.boosts.length) { <span class="boosts">{{ r.boosts.join(', ') }}</span> }
                </div>
              }
            }
            @if (retrievalView().rejected.length) {
              <h4 class="section-label">Rejected</h4>
              @for (r of retrievalView().rejected; track $index) {
                <div class="chunk rejected">
                  <span class="src">{{ r.source }} · {{ r.collection }} · {{ r.visibility }}</span>
                  <span class="score">{{ r.originalScore.toFixed(3) }} → {{ r.adjustedScore.toFixed(3) }}</span>
                  @if (r.boosts.length) { <span class="boosts">{{ r.boosts.join(', ') }}</span> }
                </div>
              }
            }
            @if (!retrievalView().selected.length && !retrievalView().rejected.length) {
              <p class="hint">No retrieval diagnostics recorded for this turn.</p>
            }
          } @else {
            <p class="hint">Pick a turn to inspect its retrieval.</p>
          }
        </div>
      </div>
    </section>
  `,
  styles: [
    `
      /* Grimoire Console styling */
      .inspector__head { display: flex; align-items: baseline; gap: 1.5rem; flex-wrap: wrap; }
      .inspector__head label { display: inline-flex; gap: 0.4rem; align-items: center; color: var(--muted); }
      .error { color: var(--danger); }
      .cols { display: grid; grid-template-columns: 22rem minmax(0, 1fr); gap: 1.5rem; align-items: start; }
      .timeline { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 0.25rem; }
      .timeline button { display: flex; gap: 0.5rem; width: 100%; text-align: left; padding: 0.4rem 0.6rem; cursor: pointer; background: var(--surface-2); color: var(--text); border: 1px solid var(--border); border-radius: var(--r-chip); }
      .timeline button.active { font-weight: 600; }
      .timeline .idx { color: var(--muted); }
      .timeline .msg { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .detail { background: var(--surface); border: 1px solid var(--border); border-radius: var(--r-card); padding: 1rem; box-shadow: var(--shadow-card); }
      .hint { color: var(--muted); font-style: italic; }
      .query code { color: var(--text); }
      .meta div { display: flex; gap: 0.5rem; }
      .meta dt { width: 9rem; color: var(--muted); }
      .meta dd { margin: 0; }
      .chunk { display: flex; gap: 0.75rem; padding: 0.25rem 0; font-size: 0.85rem; flex-wrap: wrap; }
      .chunk.rejected { opacity: 0.5; }
      .chunk .rank { font-weight: 600; }
      .boosts { color: var(--accent); }
      @media (max-width: 800px) { .cols { grid-template-columns: 1fr; } }
    `,
  ],
})
export class InspectorComponent implements OnInit {
  private readonly api = inject(ApiService);

  readonly sessions = signal<RecentSessionResponse[]>([]);
  readonly selectedId = signal<string>('');
  readonly turns = signal<RecentTurnResponse[]>([]);
  readonly selectedIndex = signal<number | null>(null);
  readonly selectedTurn = signal<TurnDetailResponse | null>(null);
  readonly loading = signal<boolean>(false);
  readonly error = signal<string | null>(null);

  readonly retrievalView = computed(() =>
    formatRetrievalDiagnostics(this.selectedTurn()?.retrieval ?? null),
  );
  readonly timings = computed(() => formatStageTimings(this.selectedTurn()?.stage_timings));

  ngOnInit(): void {
    void this.loadSessions();
  }

  label(session: RecentSessionResponse): string {
    return formatRecentSessionOption(session);
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
    this.selectedId.set(sessionId);
    this.selectedTurn.set(null);
    this.selectedIndex.set(null);
    this.turns.set([]);
    if (!sessionId) return;
    this.loading.set(true);
    this.error.set(null);
    try {
      const detail = await this.api.getSession(sessionId);
      this.turns.set([...detail.recent_turns].sort((a, b) => a.turn_index - b.turn_index));
    } catch (error) {
      this.error.set(error instanceof Error ? error.message : 'Failed to load session.');
    } finally {
      this.loading.set(false);
    }
  }

  async selectTurn(turnIndex: number): Promise<void> {
    const sessionId = this.selectedId();
    if (!sessionId) return;
    this.selectedIndex.set(turnIndex);
    this.error.set(null);
    try {
      this.selectedTurn.set(await this.api.getTurnDetail(sessionId, turnIndex));
    } catch (error) {
      this.error.set(error instanceof Error ? error.message : 'Failed to load turn.');
      this.selectedTurn.set(null);
    }
  }
}
