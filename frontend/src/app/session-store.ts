import { Injectable, computed, inject, signal } from '@angular/core';
import { ApiError, ApiService } from './api.service';
import {
  buildCatalogSessionRequest,
  buildTurnRequest,
  createCatalogSelection,
  describeMemories,
  describeRuntimeStatus,
  formatStageTimings,
  isConfirmationRequired,
  resumeTranscript,
  type CatalogSelection,
  type TranscriptEntry,
} from './play-model';
import type {
  CanonFact,
  ContentCatalog,
  CreateSessionResponse,
  MemoryEpisode,
  RuntimeStatus,
  TurnResult,
} from './models';

export interface TurnDebug {
  provider: string;
  model: string;
  reason: string;
  finishReason: string | null;
  memoryWritten: boolean;
  criticStatus: string;
  warnings: string[];
  stageTimings: string;
}

function errorMessage(error: unknown): string {
  if (error instanceof ApiError) return `${error.code}: ${error.message}`;
  return error instanceof Error ? error.message : 'Request failed.';
}

// Signal store for the Play module. Leaf components inject this singleton and read signals /
// call methods — no sibling @Input wiring, which keeps each component a self-contained file.
// Signals over NgRx: built-in and sufficient for a single-user POC.
@Injectable({ providedIn: 'root' })
export class SessionStore {
  private readonly api = inject(ApiService);

  readonly runtimeStatus = signal<RuntimeStatus | null>(null);
  readonly catalog = signal<ContentCatalog | null>(null);
  readonly selection = signal<CatalogSelection | null>(null);
  readonly session = signal<CreateSessionResponse | null>(null);
  readonly transcript = signal<TranscriptEntry[]>([]);
  readonly memories = signal<MemoryEpisode[]>([]);
  readonly canonFacts = signal<CanonFact[]>([]);
  readonly busy = signal<boolean>(false);
  readonly turnError = signal<string | null>(null);
  readonly loadError = signal<string | null>(null);
  // Set when CLOUD_MODE=ask and the router wants cloud: holds the message to resubmit on confirm.
  readonly pendingConfirm = signal<{ message: string } | null>(null);
  readonly lastDebug = signal<TurnDebug | null>(null);

  readonly statusView = computed(() => describeRuntimeStatus(this.runtimeStatus()));
  readonly sessionId = computed(() => this.session()?.session_id ?? null);
  readonly memoryLines = computed(() =>
    describeMemories({ session_id: this.sessionId() ?? '', memories: this.memories() }),
  );

  async loadRuntimeStatus(): Promise<void> {
    try {
      this.runtimeStatus.set(await this.api.getRuntimeStatus());
    } catch {
      this.runtimeStatus.set(null);
    }
  }

  async loadCatalog(selectedWorldId?: string): Promise<void> {
    try {
      const catalog = await this.api.getContentCatalog();
      this.catalog.set(catalog);
      this.selection.set(createCatalogSelection(catalog, selectedWorldId));
      this.loadError.set(null);
    } catch (error) {
      this.loadError.set(errorMessage(error));
    }
  }

  // Re-derive scenes/personas when the player picks a different world.
  selectWorld(worldId: string): void {
    const catalog = this.catalog();
    if (catalog) this.selection.set(createCatalogSelection(catalog, worldId));
  }

  // Patch the in-progress selection (scene/persona dropdowns) without rebuilding it.
  patchSelection(patch: Partial<Pick<CatalogSelection, 'sceneId' | 'personaId'>>): void {
    const current = this.selection();
    if (current) this.selection.set({ ...current, ...patch });
  }

  async createSessionFromSelection(playerName: string): Promise<void> {
    const selection = this.selection();
    if (!selection?.world) return;
    this.busy.set(true);
    this.turnError.set(null);
    try {
      const request = buildCatalogSessionRequest(selection, playerName);
      const session = await this.api.createSession(request);
      this.session.set(session);
      this.transcript.set([]);
      this.memories.set([]);
      this.pendingConfirm.set(null);
      await Promise.all([this.refreshMemories(), this.loadCanon()]);
    } catch (error) {
      this.turnError.set(errorMessage(error));
    } finally {
      this.busy.set(false);
    }
  }

  async resume(sessionId: string): Promise<void> {
    this.busy.set(true);
    this.turnError.set(null);
    try {
      const detail = await this.api.getSession(sessionId);
      this.session.set({
        session_id: detail.session_id,
        world_id: detail.world_id,
        active_scene_id: detail.active_scene_id,
        active_persona_id: detail.active_persona_id,
      });
      this.transcript.set(resumeTranscript(detail));
      this.pendingConfirm.set(null);
      await Promise.all([this.refreshMemories(), this.loadCanon()]);
    } catch (error) {
      this.turnError.set(errorMessage(error));
    } finally {
      this.busy.set(false);
    }
  }

  sendMessage(message: string, requestCloud: boolean): Promise<void> {
    return this.runTurn(message, buildTurnRequest(message, requestCloud));
  }

  // Resubmit the held message after the user approves cloud routing.
  confirmCloud(): Promise<void> {
    const pending = this.pendingConfirm();
    if (!pending) return Promise.resolve();
    return this.runTurn(pending.message, buildTurnRequest(pending.message, true, { cloudConfirmed: true }));
  }

  // Resubmit the held message but pin it to the local provider.
  forceLocal(): Promise<void> {
    const pending = this.pendingConfirm();
    if (!pending) return Promise.resolve();
    return this.runTurn(pending.message, buildTurnRequest(pending.message, false, { forceLocal: true }));
  }

  private async runTurn(message: string, request: ReturnType<typeof buildTurnRequest>): Promise<void> {
    const sessionId = this.sessionId();
    if (!sessionId) return;
    this.busy.set(true);
    this.turnError.set(null);
    try {
      const turn = await this.api.createBufferedTurn(sessionId, request);
      this.applyTurn(message, turn);
    } catch (error) {
      this.turnError.set(errorMessage(error));
    } finally {
      this.busy.set(false);
    }
  }

  private applyTurn(message: string, turn: TurnResult): void {
    if (isConfirmationRequired(turn)) {
      this.pendingConfirm.set({ message });
      return;
    }
    this.pendingConfirm.set(null);
    this.transcript.update((entries) => [
      ...entries,
      { role: 'player', text: message, source: 'new' },
      { role: 'assistant', text: turn.text, source: 'new' },
    ]);
    if (turn.route) {
      this.lastDebug.set({
        provider: turn.route.provider,
        model: turn.route.model,
        reason: turn.route.reason,
        finishReason: turn.finish_reason,
        memoryWritten: turn.memory_written,
        criticStatus: turn.critic_status,
        warnings: turn.warnings,
        stageTimings: formatStageTimings(turn.stage_timings),
      });
    }
    void this.refreshMemories();
  }

  async refreshMemories(): Promise<void> {
    const sessionId = this.sessionId();
    if (!sessionId) return;
    try {
      const response = await this.api.getSessionMemories(sessionId);
      this.memories.set(response.memories);
    } catch {
      // Memory panel is non-critical; leave the previous list on failure.
    }
  }

  async loadCanon(): Promise<void> {
    const sessionId = this.sessionId();
    if (!sessionId) return;
    try {
      const response = await this.api.getCanon(sessionId);
      this.canonFacts.set(response.facts);
    } catch {
      // Non-critical.
    }
  }

  async addCanon(text: string): Promise<void> {
    const sessionId = this.sessionId();
    const trimmed = text.trim();
    if (!sessionId || !trimmed) return;
    const fact = await this.api.addCanonFact(sessionId, { text: trimmed });
    this.canonFacts.update((facts) => [...facts, fact]);
  }

  async deleteCanon(factId: string): Promise<void> {
    const sessionId = this.sessionId();
    if (!sessionId) return;
    await this.api.deleteCanonFact(sessionId, factId);
    this.canonFacts.update((facts) => facts.filter((fact) => fact.id !== factId));
  }
}
