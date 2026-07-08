// Pure view-model helpers ported from app/web/assets/play-model.mjs.
// No framework deps — pure functions, directly unit-testable (see play-model.spec.ts).

import type {
  CanonFact,
  CatalogPersona,
  CatalogScene,
  CatalogWorld,
  ContentCatalog,
  CreateSessionRequest,
  CreateTurnRequest,
  MemoryEpisode,
  RecentSessionResponse,
  RetrievalCandidate,
  RetrievalDiagnostics,
  RuntimeStatus,
  SessionMemoriesResponse,
  SessionTurnDetailsResponse,
  StageTimings,
} from './models';

export const SESSION_DEFAULTS = Object.freeze({
  worldId: 'demo_world',
  sceneId: 'rose-gallery',
  personaId: 'archivist',
  playerName: '',
});

export interface SetupForm {
  worldId: string;
  sceneId: string;
  personaId: string;
  playerName: string;
  provider?: 'local' | 'cloud';
}

export interface CatalogSelection {
  world: CatalogWorld | null;
  scenes: CatalogScene[];
  personas: CatalogPersona[];
  sceneId: string;
  personaId: string;
}

export interface TranscriptEntry {
  role: 'player' | 'assistant';
  text: string;
  label?: string;
  source: 'new' | 'resumed';
}

export function buildSessionRequest(form: SetupForm): CreateSessionRequest {
  return {
    world_id: form.worldId,
    scene_id: form.sceneId,
    active_persona_id: form.personaId,
    player_name: form.playerName,
    provider: form.provider ?? 'local',
  };
}

export function createCatalogSelection(
  catalog: ContentCatalog,
  selectedWorldId?: string,
): CatalogSelection {
  const worlds = catalog.worlds;
  const scenesById = new Map(catalog.scenes.map((scene) => [scene.id, scene] as const));
  const personasById = new Map(catalog.personas.map((persona) => [persona.id, persona] as const));
  const world = worlds.find((candidate) => candidate.id === selectedWorldId) ?? worlds[0] ?? null;
  if (!world) {
    return { world: null, scenes: [], personas: [], sceneId: '', personaId: '' };
  }
  const scenes = world.scene_ids
    .map((id) => scenesById.get(id))
    .filter((scene): scene is CatalogScene => Boolean(scene));
  const personas = world.persona_ids
    .map((id) => personasById.get(id))
    .filter((persona): persona is CatalogPersona => Boolean(persona));
  const defaultScene = scenes.find((scene) => scene.id === world.default_scene_id) ?? scenes[0];
  return {
    world,
    scenes,
    personas,
    sceneId: defaultScene?.id ?? '',
    personaId: personas[0]?.id ?? '',
  };
}

export function buildCatalogSessionRequest(
  selection: CatalogSelection,
  playerName: string,
  provider?: 'local' | 'cloud',
): CreateSessionRequest {
  return buildSessionRequest({
    worldId: selection.world?.id ?? '',
    sceneId: selection.sceneId,
    personaId: selection.personaId,
    playerName,
    provider,
  });
}

export const RUNTIME_STATUS_UNAVAILABLE_TEXT =
  'Runtime status unavailable; gameplay controls remain available.';

function yesNo(value: boolean): string {
  return value ? 'yes' : 'no';
}

export interface RuntimeStatusView {
  warning: string;
  rows: [string, string][];
}

export function describeRuntimeStatus(status: RuntimeStatus | null): RuntimeStatusView {
  if (!status) {
    return { warning: RUNTIME_STATUS_UNAVAILABLE_TEXT, rows: [] };
  }
  return {
    warning: '',
    rows: [
      ['App', `${status.app_name} ${status.app_version}`],
      ['Environment', status.environment],
      ['Cloud mode', status.cloud_mode],
      ['Retrieval', yesNo(status.retrieval_configured)],
      ['Catalog', yesNo(status.content_catalog_available)],
      ['Local route', yesNo(status.local_provider_configured)],
      ['Cloud route', yesNo(status.cloud_provider_configured)],
    ],
  };
}

export function formatRecentSessionOption(session: RecentSessionResponse): string {
  const updated = new Date(session.updated_at);
  const updatedLabel = Number.isNaN(updated.getTime())
    ? session.updated_at
    : updated.toLocaleString();
  return [
    session.player_name,
    session.world_id,
    session.active_scene_id,
    session.active_persona_id,
    updatedLabel,
  ].join(' / ');
}

export function buildTurnRequest(
  message: string,
  { personaId = null }: { personaId?: string | null } = {},
): CreateTurnRequest {
  return {
    message,
    active_persona_id: personaId ?? undefined,
  };
}

export function describeMemories(memoriesResponse: SessionMemoriesResponse | null): string[] {
  const memories: MemoryEpisode[] = memoriesResponse?.memories ?? [];
  return memories.map((memory) => {
    const tags = memory.tags?.length ? ` (${memory.tags.join(', ')})` : '';
    return `[${memory.visibility}/${memory.importance}] ${memory.summary}${tags}`;
  });
}

export interface RetrievalRow {
  rank: number | null;
  source: string;
  collection: string;
  visibility: string;
  originalScore: number;
  adjustedScore: number;
  boosts: string[];
}

export function formatRetrievalDiagnostics(retrieval: RetrievalDiagnostics | null): {
  query: string | null;
  selected: RetrievalRow[];
  rejected: RetrievalRow[];
} {
  if (!retrieval) {
    return { query: null, selected: [], rejected: [] };
  }
  const row = (candidate: RetrievalCandidate): RetrievalRow => ({
    rank: candidate.selected_rank ?? null,
    source: candidate.source,
    collection: candidate.collection,
    visibility: candidate.visibility,
    originalScore: candidate.original_score,
    adjustedScore: candidate.adjusted_score,
    boosts: Object.entries(candidate.applied_boosts ?? {}).map(
      ([name, weight]) => `${name} ${weight.toFixed(3)}`,
    ),
  });
  return {
    query: retrieval.query ?? null,
    selected: (retrieval.selected ?? []).map(row),
    rejected: (retrieval.rejected ?? []).map(row),
  };
}

export function formatStageTimings(stageTimings: StageTimings | undefined): string {
  const entries = Object.entries(stageTimings ?? {});
  if (entries.length === 0) {
    return 'none';
  }
  return entries.map(([stage, seconds]) => `${stage} ${seconds.toFixed(1)}s`).join('; ');
}

export function canonFactList(facts: CanonFact[]): CanonFact[] {
  return facts;
}

// Full transcript for resume, built from every turn's detail. Used by
// SessionStore.resume() so the resumed conversation is complete.
export function fullTranscript(details: SessionTurnDetailsResponse): TranscriptEntry[] {
  const turns = [...details.turns].sort((left, right) => left.turn_index - right.turn_index);
  return turns.flatMap((turn): TranscriptEntry[] => {
    const label = `Resumed turn #${turn.turn_index}`;
    return [
      { role: 'player', text: turn.user_message, label, source: 'resumed' },
      { role: 'assistant', text: turn.assistant_message, label, source: 'resumed' },
    ];
  });
}
