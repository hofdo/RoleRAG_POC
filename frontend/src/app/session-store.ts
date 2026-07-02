import { Injectable, computed, inject, signal } from '@angular/core';
import { ApiError, ApiService } from './api.service';
import {
  buildCatalogSessionRequest,
  buildTurnRequest,
  createCatalogSelection,
  describeMemories,
  describeRuntimeStatus,
  formatStageTimings,
  fullTranscript,
  type CatalogSelection,
  type TranscriptEntry,
} from './play-model';
import type {
  CanonFact,
  ContentCatalog,
  CreateSessionResponse,
  MemoryEpisode,
  RecentSessionResponse,
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
  readonly currentStage = signal<string | null>(null);
  readonly turnError = signal<string | null>(null);
  readonly loadError = signal<string | null>(null);
  // Side-panel failures surface here instead of silently showing stale data.
  readonly memoryError = signal<string | null>(null);
  readonly canonError = signal<string | null>(null);
  readonly lastDebug = signal<TurnDebug | null>(null);
  readonly recentSessions = signal<RecentSessionResponse[]>([]);
  // Persona to use for the *next* turn only; the backend switches the session's
  // active persona once that turn lands (see TurnSessionLoader.load).
  readonly personaOverride = signal<string | null>(null);
  // Provider chosen at session-creation time only; the session is bound to it for its
  // lifetime (see feat(session): provider field). Not consulted after createSessionFromSelection.
  readonly sessionProvider = signal<'local' | 'cloud'>('local');

  readonly statusView = computed(() => describeRuntimeStatus(this.runtimeStatus()));
  readonly sessionId = computed(() => this.session()?.session_id ?? null);
  readonly memoryLines = computed(() =>
    describeMemories({ session_id: this.sessionId() ?? '', memories: this.memories() }),
  );
  readonly cloudAvailable = computed(() => {
    const status = this.runtimeStatus();
    return !!status && status.cloud_provider_configured && status.cloud_mode !== 'off';
  });

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
    const provider = this.sessionProvider();
    if (provider === 'cloud' && this.runtimeStatus()?.cloud_mode === 'ask') {
      if (!window.confirm('Turns in this session will be sent to the cloud provider. Continue?')) {
        return;
      }
    }
    this.busy.set(true);
    this.turnError.set(null);
    try {
      const request = buildCatalogSessionRequest(selection, playerName, provider);
      const session = await this.api.createSession(request);
      this.session.set(session);
      this.transcript.set([]);
      this.memories.set([]);
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
      const details = await this.api.getSessionTurnDetails(sessionId);
      this.transcript.set(fullTranscript(details));
      await Promise.all([this.refreshMemories(), this.loadCanon()]);
    } catch (error) {
      this.turnError.set(errorMessage(error));
    } finally {
      this.busy.set(false);
    }
  }

  async loadRecentSessions(): Promise<void> {
    try {
      this.recentSessions.set((await this.api.getRecentSessions()).sessions);
    } catch {
      this.recentSessions.set([]);
    }
  }

  sendMessage(message: string): Promise<boolean> {
    return this.runTurn(message, buildTurnRequest(message, { personaId: this.personaOverride() }));
  }

  // Switch the active scene mid-campaign. Turns already store per-turn scene_id, so
  // history for prior scenes stays intact.
  async switchScene(sceneId: string): Promise<void> {
    const sessionId = this.sessionId();
    if (!sessionId || !sceneId || this.busy()) return;
    this.busy.set(true);
    try {
      this.session.set(await this.api.updateSessionScene(sessionId, sceneId));
      this.turnError.set(null);
    } catch (error) {
      this.turnError.set(errorMessage(error));
    } finally {
      this.busy.set(false);
    }
  }

  private async runTurn(message: string, request: ReturnType<typeof buildTurnRequest>): Promise<boolean> {
    const sessionId = this.sessionId();
    if (!sessionId) return false;
    this.busy.set(true);
    this.turnError.set(null);
    this.currentStage.set(null);
    try {
      const turn = await this.api.createBufferedTurn(sessionId, request, {
        onStage: (stage) => this.currentStage.set(stage),
      });
      this.applyTurn(message, turn);
      return true;
    } catch (error) {
      this.turnError.set(errorMessage(error));
      return false;
    } finally {
      this.currentStage.set(null);
      this.busy.set(false);
    }
  }

  private applyTurn(message: string, turn: TurnResult): void {
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
      this.memoryError.set(null);
    } catch (error) {
      // Keep the previous list, but say it may be stale instead of failing silently.
      this.memoryError.set(`Refresh failed — list may be stale. ${errorMessage(error)}`);
    }
  }

  async loadCanon(): Promise<void> {
    const sessionId = this.sessionId();
    if (!sessionId) return;
    try {
      const response = await this.api.getCanon(sessionId);
      this.canonFacts.set(response.facts);
      this.canonError.set(null);
    } catch (error) {
      this.canonError.set(`Load failed — list may be stale. ${errorMessage(error)}`);
    }
  }

  // Mutations update the list only after the server confirms, so a failure needs no
  // rollback — it just has to be visible.
  async addCanon(text: string): Promise<void> {
    const sessionId = this.sessionId();
    const trimmed = text.trim();
    if (!sessionId || !trimmed) return;
    try {
      const fact = await this.api.addCanonFact(sessionId, { text: trimmed });
      this.canonFacts.update((facts) => [...facts, fact]);
      this.canonError.set(null);
    } catch (error) {
      this.canonError.set(`Fact not added. ${errorMessage(error)}`);
    }
  }

  async deleteCanon(factId: string): Promise<void> {
    const sessionId = this.sessionId();
    if (!sessionId) return;
    try {
      await this.api.deleteCanonFact(sessionId, factId);
      this.canonFacts.update((facts) => facts.filter((fact) => fact.id !== factId));
      this.canonError.set(null);
    } catch (error) {
      this.canonError.set(`Fact not deleted. ${errorMessage(error)}`);
    }
  }

  // Delete the last turn (and any memories written after it) and resend the same
  // player message. The delete step manages its own busy flag; the resend delegates
  // to sendMessage, which manages busy for the turn itself.
  async rerollLast(): Promise<void> {
    const sessionId = this.sessionId();
    if (!sessionId || this.busy()) return;
    const lastPlayer = [...this.transcript()].reverse().find((e) => e.role === 'player');
    if (!lastPlayer) return;
    this.turnError.set(null);
    this.busy.set(true);
    try {
      await this.api.deleteLastTurn(sessionId);
      this.transcript.update((all) => all.slice(0, -2));
    } catch (error) {
      this.turnError.set(errorMessage(error));
      return;
    } finally {
      this.busy.set(false);
    }
    await this.sendMessage(lastPlayer.text);
    void this.refreshMemories();
  }
}
