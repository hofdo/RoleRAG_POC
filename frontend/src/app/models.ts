// TypeScript interfaces for the RoleRAG HTTP API.
// Ported from the JSDoc typedefs in app/web/assets/api-client.mjs, extended with the
// turn-detail / retrieval-diagnostics / structured-error fields the inspector & analytics
// modules consume (see app/api/routes.py).

export interface CreateSessionRequest {
  world_id: string;
  scene_id: string;
  player_name: string;
  active_persona_id: string;
  provider?: 'local' | 'cloud';
}

export interface CreateSessionResponse {
  session_id: string;
  world_id: string;
  active_scene_id: string;
  active_persona_id: string;
  provider?: string;
}

export interface RecentSessionResponse {
  session_id: string;
  world_id: string;
  active_scene_id: string;
  active_persona_id: string;
  player_name: string;
  created_at: string;
  updated_at: string;
}

export interface RecentSessionsResponse {
  sessions: RecentSessionResponse[];
}

export interface CatalogWorld {
  id: string;
  name: string;
  default_scene_id: string;
  scene_ids: string[];
  persona_ids: string[];
}

export interface CatalogScene {
  id: string;
  title: string;
  location: string;
  player_visible_summary: string;
  active_personas: string[];
}

export interface CatalogPersona {
  id: string;
  name: string;
  role: string;
  public_description: string;
  speaking_style: string;
}

export interface ContentCatalog {
  worlds: CatalogWorld[];
  scenes: CatalogScene[];
  personas: CatalogPersona[];
}

export interface RuntimeStatus {
  app_name: string;
  app_version: string;
  environment: string;
  cloud_mode: string;
  retrieval_configured: boolean;
  content_catalog_available: boolean;
  local_provider_configured: boolean;
  cloud_provider_configured: boolean;
}

export interface CreateTurnRequest {
  message: string;
  active_persona_id?: string;
}

export interface RouteResponse {
  provider: string;
  model: string;
  reason: string;
}

export interface RecentTurnResponse {
  turn_index: number;
  user_message: string;
  assistant_message: string;
  created_at: string;
}

export interface GetSessionResponse {
  session_id: string;
  world_id: string;
  active_scene_id: string;
  active_persona_id: string;
  recent_turns: RecentTurnResponse[];
}

export interface TurnError {
  category: string;
  stage: string;
  message: string;
  suggestion: string | null;
}

export interface RetrievalCandidate {
  selected_rank: number | null;
  source: string;
  collection: string;
  visibility: string;
  original_score: number;
  adjusted_score: number;
  applied_boosts: Record<string, number>;
  // Additive fields the backend has always sent (RetrievalCandidateResponse) but the
  // UI ignored until the debug panel needed them (#85). Optional so older fixtures
  // and tests without them stay valid.
  id?: string;
  source_type?: string;
  tags?: string[];
  slice_score?: number | null;
  slice_matched_terms?: string[];
  slice_guaranteed?: boolean;
}

export interface RetrievalDiagnostics {
  query: string | null;
  selected: RetrievalCandidate[];
  rejected: RetrievalCandidate[];
}

export type StageTimings = Record<string, number>;

// Result of a (buffered/SSE) turn.
export interface TurnResult {
  status: 'completed';
  text: string;
  route: RouteResponse | null;
  finish_reason: string | null;
  memory_written: boolean;
  critic_status: string;
  warnings: string[];
  errors?: TurnError[];
  retrieval: RetrievalDiagnostics | null;
  stage_timings?: StageTimings;
}

export interface TurnDetailResponse {
  turn_index: number;
  scene_id: string;
  persona_id: string;
  user_message: string;
  assistant_message: string;
  route: RouteResponse;
  created_at: string;
  finish_reason: string | null;
  memory_written: boolean;
  critic_status: string;
  warnings: string[];
  errors: TurnError[];
  stage_timings: StageTimings;
  retrieval: RetrievalDiagnostics | null;
}

export interface MemoryEpisode {
  id: string;
  scene_id: string;
  actor_id: string | null;
  summary: string;
  importance: number;
  visibility: string;
  tags: string[];
}

export interface SessionMemoriesResponse {
  session_id: string;
  memories: MemoryEpisode[];
}

export interface CanonFact {
  id: string;
  text: string;
}

export interface CanonFactsResponse {
  session_id: string;
  facts: CanonFact[];
}

export interface AddCanonFactRequest {
  text: string;
}

export interface DeleteLastTurnResponse {
  session_id: string;
  deleted_turn_index: number;
  user_message: string;
  deleted_memory_count: number;
}

export interface EvalRunSummary {
  id: string;
  status: string;
  turn_count: number;
  recall_misses: number;
  extraction_misses: number;
  retrieval_misses: number;
  total_seconds: number;
  p50_seconds: number;
  p95_seconds: number;
  warning_total: number;
}

export interface EvalRunsResponse {
  results_dir: string;
  runs: EvalRunSummary[];
}

export interface SessionTurnDetailsResponse {
  session_id: string;
  turns: TurnDetailResponse[];
}

// --- RAG debug/vector-map surface (POST /diagnostics/rag-map, /diagnostics/chunk-texts, #85) ---

export interface RagMapRequest {
  query_text?: string | null;
  include_hidden?: boolean;
}

export interface RagMapPoint {
  id: string;
  collection: string;
  x: number;
  y: number;
  source: string;
  source_type: string;
  visibility: string;
  tags: string[];
  world_id: string | null;
  scene_id: string | null;
  persona_id: string | null;
  session_id: string | null;
  actor_id: string | null;
  importance: number | null;
  created_at: string | null;
  text_preview: string | null;
  text_redacted: boolean;
}

export interface RagMapResponse {
  points: RagMapPoint[];
  point_count: number;
  truncated: boolean;
  explained_variance: number[];
  query_point: number[] | null;
  embedding_model: string;
}

export interface ChunkTextItemRequest {
  collection: string;
  id: string;
}

export interface ChunkTextsRequest {
  items: ChunkTextItemRequest[];
  include_hidden?: boolean;
}

export interface ChunkTextResponse {
  id: string;
  collection: string;
  visibility: string | null;
  text: string | null;
  text_redacted: boolean;
  found: boolean;
}

export interface ChunkTextsResponse {
  chunks: ChunkTextResponse[];
}

// Eval drill-down = the raw conversation-checkpoint.json. Loosely typed: the UI reads a few
// headline fields and tolerates everything else.
export interface EvalRunDetail {
  status?: string;
  warning_counts?: Record<string, number>;
  quality_metrics?: {
    callback_recall_misses?: number;
    memory_extraction_misses?: number;
    retrieval_selection_misses?: number;
    latency?: { total_seconds?: number; p50_seconds?: number; p95_seconds?: number };
  };
  turns?: {
    turn_index?: number;
    duration_seconds?: number;
    stage_timings?: Record<string, number>;
    finish_reason?: string | null;
    warnings?: string[];
  }[];
  [key: string]: unknown;
}
