import {
  Component,
  ElementRef,
  OnDestroy,
  OnInit,
  afterNextRender,
  computed,
  effect,
  inject,
  signal,
  viewChild,
} from '@angular/core';
import * as echarts from 'echarts/core';
import { ScatterChart } from 'echarts/charts';
import {
  GridComponent,
  LegendComponent,
  TooltipComponent,
  DataZoomComponent,
} from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';
import type { EChartsOption } from 'echarts';
import { ApiService } from '../api.service';
import { SessionStore } from '../session-store';
import {
  COLLECTION_COLORS,
  COLLECTION_LABELS,
  buildScatterSeries,
  escapeHtml,
  filterPoints,
  highlightIdsFromTurn,
  tooltipLines,
  type MapFilters,
  type SeriesSpec,
} from '../vector-map-model';
import type { RagMapPoint } from '../models';

echarts.use([
  ScatterChart,
  GridComponent,
  LegendComponent,
  TooltipComponent,
  DataZoomComponent,
  CanvasRenderer,
]);

const ALL_COLLECTIONS = Object.keys(COLLECTION_LABELS);
const ALL_VISIBILITIES = ['player', 'gm', 'character_private'];

function defaultFilters(): MapFilters {
  return {
    collections: new Set(ALL_COLLECTIONS),
    visibilities: new Set(ALL_VISIBILITIES),
    sessionId: null,
    tag: null,
  };
}

// Vector Map (backlog #85): a 2D PCA scatter of every point currently in the vector store
// (canon lore / session memory / persona memory), colored by collection, with optional
// highlighting of the last turn's retrieval outcome (selected vs. rejected chunks) and a
// star marker for the retrieval query itself. Pure aggregation/filtering lives in
// vector-map-model.ts; this component is the thin fetch + echarts + signal glue.
@Component({
  selector: 'app-vector-map',
  imports: [],
  template: `
    <section class="map">
      <header class="map__head">
        <h1>Vector Map</h1>
        <button type="button" class="ghost" (click)="refresh()" [disabled]="loading()">
          Refresh
        </button>
        @if (loading()) {
          <span class="hint">Loading…</span>
        }
        @if (error()) {
          <span class="error">{{ error() }}</span>
        }
      </header>

      <p class="note">
        {{ pointCount() }} points · PCA pc1 {{ pcaPct(0) }}% / pc2 {{ pcaPct(1) }}% · model
        {{ embeddingModel() || '—' }}
        @if (truncated()) {
          <span class="warn"> · truncated: not every point is shown</span>
        }
      </p>

      <div class="layout">
        <div class="filters">
          <h2 class="section-label">Collections</h2>
          @for (key of allCollections; track key) {
            <label class="chk">
              <input
                type="checkbox"
                [checked]="filters().collections.has(key)"
                (change)="toggleCollection(key, $any($event.target).checked)"
              />
              <span class="dot" [style.background]="collectionColor(key)"></span>
              {{ collectionLabel(key) }}
            </label>
          }

          <h2 class="section-label">Visibility</h2>
          @for (v of allVisibilities; track v) {
            <label class="chk">
              <input
                type="checkbox"
                [checked]="filters().visibilities.has(v)"
                (change)="toggleVisibility(v, $any($event.target).checked)"
              />
              {{ v }}
            </label>
          }

          <h2 class="section-label">Session id</h2>
          <input type="text" placeholder="exact session id" (change)="setSessionId($any($event.target).value)" />

          <h2 class="section-label">Tag</h2>
          <input type="text" placeholder="exact tag" (change)="setTag($any($event.target).value)" />

          <h2 class="section-label">Options</h2>
          <label class="chk">
            <input
              type="checkbox"
              [checked]="highlightLastTurn()"
              (change)="toggleHighlightLastTurn($any($event.target).checked)"
            />
            Highlight last turn
          </label>
          <label class="chk">
            <input
              type="checkbox"
              [checked]="revealHidden()"
              (change)="toggleRevealHidden($any($event.target).checked)"
            />
            Reveal hidden text
          </label>
        </div>

        <div class="chart-wrap">
          <div #chartEl class="chart"></div>
        </div>

        @if (selectedPoint(); as p) {
          <aside class="detail">
            <button type="button" class="ghost close" (click)="closeDetail()">Close</button>
            <h2 class="section-label">Point</h2>
            <dl class="meta">
              <div><dt>id</dt><dd>{{ p.id }}</dd></div>
              <div><dt>collection</dt><dd>{{ collectionLabel(p.collection) }}</dd></div>
              <div><dt>visibility</dt><dd>{{ p.visibility }}</dd></div>
              <div><dt>source</dt><dd>{{ p.source }}</dd></div>
              <div><dt>source_type</dt><dd>{{ p.source_type }}</dd></div>
              <div><dt>tags</dt><dd>{{ p.tags.join(', ') || '—' }}</dd></div>
              <div><dt>world</dt><dd>{{ p.world_id ?? '—' }}</dd></div>
              <div><dt>scene</dt><dd>{{ p.scene_id ?? '—' }}</dd></div>
              <div><dt>persona</dt><dd>{{ p.persona_id ?? '—' }}</dd></div>
              <div><dt>session</dt><dd>{{ p.session_id ?? '—' }}</dd></div>
              <div><dt>actor</dt><dd>{{ p.actor_id ?? '—' }}</dd></div>
              <div><dt>importance</dt><dd>{{ p.importance ?? '—' }}</dd></div>
              <div><dt>created_at</dt><dd>{{ p.created_at ?? '—' }}</dd></div>
            </dl>
            <h2 class="section-label">Text</h2>
            @if (selectedText() !== null) {
              <p class="text">{{ selectedText() }}</p>
            } @else {
              <p class="hint">Loading text…</p>
            }
          </aside>
        }
      </div>
    </section>
  `,
  styles: [
    `
      /* Grimoire Console styling */
      .map__head { display: flex; align-items: baseline; gap: 1rem; flex-wrap: wrap; }
      .error { color: var(--danger); }
      .hint { color: var(--muted); font-style: italic; }
      .note { color: var(--muted); font-size: 0.85rem; }
      .note .warn { color: var(--danger); }
      .layout { display: grid; grid-template-columns: 14rem minmax(0, 1fr) 20rem; gap: 1rem; align-items: start; }
      .filters { display: flex; flex-direction: column; gap: 0.3rem; background: var(--surface); border: 1px solid var(--border); border-radius: var(--r-card); padding: 0.75rem; }
      .filters .section-label { margin-top: 0.6rem; }
      .filters .section-label:first-child { margin-top: 0; }
      .chk { display: flex; align-items: center; gap: 0.4rem; font-size: 0.85rem; cursor: pointer; }
      .chk input { padding: 0; }
      .dot { width: 0.65rem; height: 0.65rem; border-radius: 50%; display: inline-block; }
      .filters input[type='text'] { width: 100%; }
      .chart-wrap { background: var(--surface); border: 1px solid var(--border); border-radius: var(--r-card); padding: 0.5rem; box-shadow: var(--shadow-card); }
      .chart { width: 100%; height: 70vh; }
      .detail { background: var(--surface); border: 1px solid var(--border); border-radius: var(--r-card); padding: 0.75rem; position: relative; max-height: 70vh; overflow-y: auto; }
      .detail .close { position: absolute; top: 0.5rem; right: 0.5rem; }
      .meta div { display: flex; gap: 0.5rem; font-size: 0.82rem; }
      .meta dt { width: 6rem; color: var(--muted); flex-shrink: 0; }
      .meta dd { margin: 0; overflow-wrap: anywhere; }
      .text { font-size: 0.85rem; white-space: pre-wrap; }
      @media (max-width: 800px) {
        .layout { grid-template-columns: 1fr; }
      }
    `,
  ],
})
export class VectorMapComponent implements OnInit, OnDestroy {
  private readonly api = inject(ApiService);
  private readonly store = inject(SessionStore);

  readonly allCollections = ALL_COLLECTIONS;
  readonly allVisibilities = ALL_VISIBILITIES;

  private readonly chartEl = viewChild<ElementRef<HTMLDivElement>>('chartEl');

  readonly points = signal<RagMapPoint[]>([]);
  readonly loading = signal<boolean>(false);
  readonly error = signal<string | null>(null);
  readonly filters = signal<MapFilters>(defaultFilters());
  readonly revealHidden = signal<boolean>(false);
  readonly highlightLastTurn = signal<boolean>(true);
  readonly selectedPoint = signal<RagMapPoint | null>(null);
  readonly selectedText = signal<string | null>(null);
  readonly truncated = signal<boolean>(false);
  readonly explainedVariance = signal<number[]>([]);
  readonly embeddingModel = signal<string>('');
  readonly pointCount = signal<number>(0);
  readonly queryPoint = signal<number[] | null>(null);

  private readonly filteredPoints = computed(() => filterPoints(this.points(), this.filters()));
  private readonly highlights = computed(() =>
    highlightIdsFromTurn(this.highlightLastTurn() ? this.store.lastTurn() : null),
  );
  private readonly pointsById = computed(() => new Map(this.points().map((p) => [p.id, p])));

  private chart: echarts.ECharts | null = null;
  private resizeObserver: ResizeObserver | null = null;

  constructor() {
    afterNextRender(() => {
      const el = this.chartEl()?.nativeElement;
      if (!el) return;
      this.chart = echarts.init(el);
      this.chart.on('click', (params) => {
        const pointId = (params.data as { pointId?: string | null } | undefined)?.pointId;
        if (pointId) void this.selectPoint(pointId);
      });
      this.resizeObserver = new ResizeObserver(() => this.chart?.resize());
      this.resizeObserver.observe(el);
      this.renderChart();
    });

    effect(() => {
      // Reactive dependency on filtered points / highlights / query point.
      this.renderChart();
    });
  }

  ngOnInit(): void {
    void this.load();
  }

  ngOnDestroy(): void {
    this.resizeObserver?.disconnect();
    this.chart?.dispose();
    this.chart = null;
  }

  async refresh(): Promise<void> {
    await this.load();
  }

  private async load(): Promise<void> {
    this.loading.set(true);
    this.error.set(null);
    try {
      const response = await this.api.getRagMap({
        include_hidden: this.revealHidden(),
        query_text: this.highlightLastTurn() ? (this.store.lastTurn()?.retrieval?.query ?? null) : null,
      });
      this.points.set(response.points);
      this.pointCount.set(response.point_count);
      this.truncated.set(response.truncated);
      this.explainedVariance.set(response.explained_variance);
      this.queryPoint.set(response.query_point);
      this.embeddingModel.set(response.embedding_model);
    } catch (error) {
      this.error.set(error instanceof Error ? error.message : 'Failed to load the vector map.');
    } finally {
      this.loading.set(false);
    }
  }

  collectionLabel(key: string): string {
    return COLLECTION_LABELS[key] ?? key;
  }

  collectionColor(key: string): string {
    return COLLECTION_COLORS[key] ?? '#888888';
  }

  pcaPct(index: number): string {
    const value = this.explainedVariance()[index];
    return value === undefined ? '—' : (value * 100).toFixed(0);
  }

  toggleCollection(key: string, checked: boolean): void {
    this.filters.update((f) => {
      const next = new Set(f.collections);
      if (checked) next.add(key);
      else next.delete(key);
      return { ...f, collections: next };
    });
  }

  toggleVisibility(value: string, checked: boolean): void {
    this.filters.update((f) => {
      const next = new Set(f.visibilities);
      if (checked) next.add(value);
      else next.delete(value);
      return { ...f, visibilities: next };
    });
  }

  setSessionId(value: string): void {
    this.filters.update((f) => ({ ...f, sessionId: value.trim() || null }));
  }

  setTag(value: string): void {
    this.filters.update((f) => ({ ...f, tag: value.trim() || null }));
  }

  toggleHighlightLastTurn(checked: boolean): void {
    this.highlightLastTurn.set(checked);
    void this.load();
  }

  toggleRevealHidden(checked: boolean): void {
    this.revealHidden.set(checked);
    void this.load();
  }

  closeDetail(): void {
    this.selectedPoint.set(null);
    this.selectedText.set(null);
  }

  private async selectPoint(pointId: string): Promise<void> {
    const point = this.pointsById().get(pointId);
    if (!point) return;
    this.selectedPoint.set(point);
    this.selectedText.set(null);
    try {
      const response = await this.api.getChunkTexts({
        items: [{ collection: point.collection, id: point.id }],
        include_hidden: this.revealHidden(),
      });
      const chunk = response.chunks[0];
      this.selectedText.set(
        !chunk || chunk.text_redacted ? '[hidden — enable reveal]' : (chunk.text ?? '—'),
      );
    } catch (error) {
      this.selectedText.set(
        error instanceof Error ? `Error: ${error.message}` : 'Failed to load text.',
      );
    }
  }

  private renderChart(): void {
    if (!this.chart) return;
    const specs = buildScatterSeries(this.filteredPoints(), this.highlights(), this.queryPoint());
    this.chart.setOption(this.buildOption(specs), { notMerge: true });
  }

  private buildOption(specs: SeriesSpec[]): EChartsOption {
    const byId = this.pointsById();
    return {
      backgroundColor: 'transparent',
      grid: { left: 56, right: 24, top: 40, bottom: 48, containLabel: true },
      legend: { top: 0, textStyle: { color: '#ece3d0' } },
      xAxis: {
        type: 'value',
        name: 'PC1',
        axisLine: { lineStyle: { color: '#38301f' } },
        axisLabel: { color: '#9a8e78' },
        splitLine: { lineStyle: { color: '#262017' } },
      },
      yAxis: {
        type: 'value',
        name: 'PC2',
        axisLine: { lineStyle: { color: '#38301f' } },
        axisLabel: { color: '#9a8e78' },
        splitLine: { lineStyle: { color: '#262017' } },
      },
      dataZoom: [
        { type: 'inside', xAxisIndex: 0 },
        { type: 'inside', yAxisIndex: 0 },
      ],
      tooltip: {
        trigger: 'item',
        formatter: (params) => {
          const p = params as { data?: { pointId?: string | null } };
          const pointId = p.data?.pointId;
          if (!pointId) return 'Query';
          const point = byId.get(pointId);
          if (!point) return escapeHtml(pointId);
          return tooltipLines(point).map((line) => escapeHtml(line)).join('<br/>');
        },
      },
      series: specs.map((spec) => ({
        type: 'scatter' as const,
        name: spec.name,
        symbol: spec.symbol ?? 'circle',
        symbolSize: spec.symbolSize,
        z: spec.zLevel ?? 1,
        itemStyle: { color: spec.color },
        data: spec.data,
      })),
    };
  }
}
