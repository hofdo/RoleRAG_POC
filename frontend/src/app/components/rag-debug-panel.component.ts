import { Component, computed, inject, signal } from '@angular/core';
import { RouterLink } from '@angular/router';
import { ApiService } from '../api.service';
import { SessionStore } from '../session-store';
import { buildDebugView, newMemoryIds, type DebugChunkRow } from '../rag-debug-model';

interface ChunkFetchState {
  loading: boolean;
  text: string | null;
  redacted: boolean;
  error: string | null;
}

// Collapsible live RAG debug panel for the Play page (#85): shows the retrieval query,
// selected/rejected chunks with scores + boosts, on-demand chunk text (via the diagnostics
// chunk-texts endpoint), new memories since the last turn, stage timings and critic status.
// Collapsed by default (native <details>) so it stays out of the way during normal play.
@Component({
  selector: 'app-rag-debug-panel',
  imports: [RouterLink],
  template: `
    <details class="rag-debug">
      <summary class="rag-debug__summary">RAG debug</summary>

      @if (view(); as v) {
        <div class="rag-debug__body">
          <div class="rag-debug__query-row">
            <span class="rag-debug__label">Query</span>
            <span class="rag-debug__query" [title]="v.query ?? ''">{{ v.query ?? '—' }}</span>
          </div>

          <label class="rag-debug__reveal">
            <input type="checkbox" [checked]="revealHidden()" (change)="toggleReveal()" />
            Reveal hidden (GM) text
          </label>

          <h4 class="rag-debug__section">Selected</h4>
          @if (v.selected.length) {
            <ul class="rag-debug__list">
              @for (row of v.selected; track $index) {
                <li class="rag-debug__chunk">
                  <button
                    type="button"
                    class="rag-debug__chunk-head"
                    [class.is-disabled]="!row.id"
                    (click)="toggleRow(row)"
                  >
                    <span class="rag-debug__rank">#{{ row.rank ?? '—' }}</span>
                    <span class="rag-debug__src">{{ row.source }} · {{ row.collection }}</span>
                    <span class="rag-debug__vis" [class.is-hidden]="row.hidden">{{ row.visibility }}</span>
                    <span class="rag-debug__score">{{ row.originalScore.toFixed(3) }} → {{ row.adjustedScore.toFixed(3) }}</span>
                  </button>
                  @if (row.boosts.length) {
                    <span class="rag-debug__boosts">
                      @for (b of row.boosts; track b) { <span class="rag-debug__chip">{{ b }}</span> }
                    </span>
                  }
                  @if (row.sliceLabel) { <span class="rag-debug__slice">{{ row.sliceLabel }}</span> }
                  @if (isExpanded(row)) {
                    <div class="rag-debug__text">
                      @if (chunkState(row); as state) {
                        @if (state.loading) { <em>Loading…</em> }
                        @if (state.error) { <span class="rag-debug__error">{{ state.error }}</span> }
                        @if (!state.loading && !state.error) {
                          @if (state.redacted) {
                            <em>hidden (enable reveal)</em>
                          } @else {
                            <p>{{ state.text }}</p>
                          }
                        }
                      }
                    </div>
                  }
                </li>
              }
            </ul>
          } @else {
            <p class="rag-debug__empty">No selected chunks.</p>
          }

          <h4 class="rag-debug__section">Rejected</h4>
          @if (v.rejected.length) {
            <ul class="rag-debug__list rag-debug__list--dim">
              @for (row of v.rejected; track $index) {
                <li class="rag-debug__chunk">
                  <button
                    type="button"
                    class="rag-debug__chunk-head"
                    [class.is-disabled]="!row.id"
                    (click)="toggleRow(row)"
                  >
                    <span class="rag-debug__src">{{ row.source }} · {{ row.collection }}</span>
                    <span class="rag-debug__vis" [class.is-hidden]="row.hidden">{{ row.visibility }}</span>
                    <span class="rag-debug__score">{{ row.originalScore.toFixed(3) }} → {{ row.adjustedScore.toFixed(3) }}</span>
                  </button>
                  @if (row.boosts.length) {
                    <span class="rag-debug__boosts">
                      @for (b of row.boosts; track b) { <span class="rag-debug__chip">{{ b }}</span> }
                    </span>
                  }
                  @if (row.sliceLabel) { <span class="rag-debug__slice">{{ row.sliceLabel }}</span> }
                  @if (isExpanded(row)) {
                    <div class="rag-debug__text">
                      @if (chunkState(row); as state) {
                        @if (state.loading) { <em>Loading…</em> }
                        @if (state.error) { <span class="rag-debug__error">{{ state.error }}</span> }
                        @if (!state.loading && !state.error) {
                          @if (state.redacted) {
                            <em>hidden (enable reveal)</em>
                          } @else {
                            <p>{{ state.text }}</p>
                          }
                        }
                      }
                    </div>
                  }
                </li>
              }
            </ul>
          } @else {
            <p class="rag-debug__empty">No rejected chunks.</p>
          }

          <h4 class="rag-debug__section">New memories since last turn</h4>
          @if (newMemories().length) {
            <ul class="rag-debug__list">
              @for (m of newMemories(); track m.id) {
                <li>{{ m.summary }}</li>
              }
            </ul>
          } @else {
            <p class="rag-debug__empty">No new memories.</p>
          }
          <p class="rag-debug__meta">
            Memory written: {{ v.memoryWritten ? 'yes' : 'no' }} — memory curation may run deferred.
          </p>

          <p class="rag-debug__meta">{{ v.stageTimings }} · critic: {{ v.criticStatus ?? '—' }}</p>
          @if (v.warnings.length) {
            <ul class="rag-debug__warnings">
              @for (w of v.warnings; track $index) { <li>{{ w }}</li> }
            </ul>
          }

          <a
            class="rag-debug__link"
            [routerLink]="['/inspector']"
            [queryParams]="{ session: store.sessionId() }"
          >Open in Inspector →</a>
        </div>
      } @else {
        <p class="rag-debug__empty">No turn yet.</p>
      }
    </details>
  `,
  // Grimoire Console styling — mirrors memory-panel/canon-panel conventions.
  styles: [`
    .rag-debug {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--r-card);
      padding: 0.9rem;
      color: var(--text);
    }
    .rag-debug__summary {
      cursor: pointer;
      font-weight: 600;
      color: var(--accent);
    }
    .rag-debug__body { display: flex; flex-direction: column; gap: 0.4rem; margin-top: 0.6rem; }
    .rag-debug__section { margin: 0.4rem 0 0.1rem; color: var(--muted); font-size: 0.85rem; }
    .rag-debug__query-row { display: flex; gap: 0.4rem; align-items: baseline; min-width: 0; }
    .rag-debug__label { color: var(--muted); flex-shrink: 0; }
    .rag-debug__query {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-family: var(--font-mono);
      font-size: 0.85rem;
    }
    .rag-debug__reveal { display: flex; align-items: center; gap: 0.4rem; font-size: 0.8rem; color: var(--muted); }
    .rag-debug__list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 0.3rem; }
    .rag-debug__list--dim { opacity: 0.6; }
    .rag-debug__chunk { display: flex; flex-direction: column; gap: 0.2rem; }
    .rag-debug__chunk-head {
      display: flex;
      gap: 0.5rem;
      flex-wrap: wrap;
      align-items: center;
      width: 100%;
      text-align: left;
      background: var(--surface-2);
      border: 1px solid var(--border);
      border-radius: var(--r-chip);
      padding: 0.3rem 0.5rem;
      color: var(--text);
      font-size: 0.82rem;
      cursor: pointer;
    }
    .rag-debug__chunk-head.is-disabled { cursor: default; opacity: 0.7; }
    .rag-debug__rank { font-weight: 600; color: var(--muted); }
    .rag-debug__src { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .rag-debug__vis {
      padding: 0.05rem 0.4rem;
      border-radius: var(--r-chip);
      border: 1px solid var(--border);
      font-size: 0.72rem;
      color: var(--muted);
    }
    .rag-debug__vis.is-hidden { color: var(--danger); border-color: var(--danger); }
    .rag-debug__score { font-variant-numeric: tabular-nums; color: var(--muted); }
    .rag-debug__boosts { display: flex; gap: 0.25rem; flex-wrap: wrap; }
    .rag-debug__chip {
      font-size: 0.72rem;
      font-family: var(--font-mono);
      color: var(--accent);
      background: var(--surface-2);
      border: 1px solid var(--border);
      border-radius: var(--r-chip);
      padding: 0 0.35rem;
    }
    .rag-debug__slice { font-size: 0.72rem; color: var(--muted); }
    .rag-debug__text { background: var(--surface-2); border-radius: var(--r-chip); padding: 0.4rem 0.5rem; font-size: 0.82rem; }
    .rag-debug__text p { margin: 0; white-space: pre-wrap; }
    .rag-debug__error { color: var(--danger); }
    .rag-debug__empty { margin: 0; color: var(--muted); font-style: italic; font-size: 0.85rem; }
    .rag-debug__meta { margin: 0; color: var(--muted); font-size: 0.8rem; }
    .rag-debug__warnings { margin: 0; padding-left: 1.25rem; color: var(--danger); font-size: 0.8rem; }
    .rag-debug__link { color: var(--accent); font-size: 0.85rem; align-self: flex-start; }
  `],
})
export class RagDebugPanelComponent {
  readonly store = inject(SessionStore);
  private readonly api = inject(ApiService);

  readonly revealHidden = signal(false);
  private readonly expandedRows = signal<ReadonlySet<string>>(new Set());
  private readonly chunkCache = signal<ReadonlyMap<string, ChunkFetchState>>(new Map());

  readonly view = computed(() => buildDebugView(this.store.lastTurn()));
  readonly newMemories = computed(() => {
    const ids = new Set(newMemoryIds(this.store.memoryIdsBeforeLastTurn(), this.store.memories()));
    return this.store.memories().filter((memory) => ids.has(memory.id));
  });

  toggleRow(row: DebugChunkRow): void {
    if (!row.id) return;
    const key = this.rowKey(row);
    const expanded = new Set(this.expandedRows());
    if (expanded.has(key)) {
      expanded.delete(key);
    } else {
      expanded.add(key);
      void this.fetchChunk(row);
    }
    this.expandedRows.set(expanded);
  }

  isExpanded(row: DebugChunkRow): boolean {
    return !!row.id && this.expandedRows().has(this.rowKey(row));
  }

  chunkState(row: DebugChunkRow): ChunkFetchState | undefined {
    if (!row.id) return undefined;
    return this.chunkCache().get(this.cacheKey(row));
  }

  // Cache is keyed by the reveal flag, so toggling it never has to invalidate anything —
  // it just refetches whatever is currently expanded under the new key.
  toggleReveal(): void {
    this.revealHidden.update((value) => !value);
    const v = this.view();
    const rows = [...(v?.selected ?? []), ...(v?.rejected ?? [])];
    for (const row of rows) {
      if (this.isExpanded(row)) void this.fetchChunk(row);
    }
  }

  private rowKey(row: DebugChunkRow): string {
    return `${row.collection}:${row.id}`;
  }

  private cacheKey(row: DebugChunkRow): string {
    return `${row.collection}:${row.id}:${this.revealHidden()}`;
  }

  private setCacheEntry(key: string, entry: ChunkFetchState): void {
    this.chunkCache.update((cache) => new Map(cache).set(key, entry));
  }

  private async fetchChunk(row: DebugChunkRow): Promise<void> {
    if (!row.id) return;
    const key = this.cacheKey(row);
    if (this.chunkCache().has(key)) return;
    this.setCacheEntry(key, { loading: true, text: null, redacted: false, error: null });
    try {
      const response = await this.api.getChunkTexts({
        items: [{ collection: row.collection, id: row.id }],
        include_hidden: this.revealHidden(),
      });
      const chunk = response.chunks[0];
      this.setCacheEntry(key, {
        loading: false,
        text: chunk?.text ?? null,
        redacted: chunk?.text_redacted ?? false,
        error: chunk && chunk.found ? null : 'Chunk not found.',
      });
    } catch (error) {
      this.setCacheEntry(key, {
        loading: false,
        text: null,
        redacted: false,
        error: error instanceof Error ? error.message : 'Failed to load chunk text.',
      });
    }
  }
}
