# 28 - RAG Debug Panel & Vector Map

> Reviewed: 2026-07-15 @ c69baf0

Live observability for the RAG core while playing (#85): a per-turn debug panel on the Play
page, chunk-text drill-down, and a 2-D map of every point in the vector store. Everything
here is read-only and additive — no engine behavior changed.

## What shipped

1. **Live debug panel** (`/app/play`, right column, collapsed by default). After every turn
   it shows the retrieval query, selected and rejected candidates (rank, source, collection,
   visibility, `original → adjusted` score with per-boost breakdown, lexical-slice labels),
   stage timings, critic status, warnings, and memories written since the previous turn.
   The data comes from the SSE terminal frame the backend already sent — the panel adds no
   backend calls until you expand a row (chunk text is fetched lazily). A deep link opens
   the same session in the full RAG Inspector.
2. **Vector Map page** (`/app/map`). Every real point across all three collections
   (`canon_lore`, `session_memory`, `persona_memory`) plotted as an interactive 2-D scatter
   (ECharts canvas: wheel zoom, drag pan, hover tooltips, click for a full-detail sidebar),
   with filters (collection, visibility, session, tag), the last turn's selected/rejected
   candidates highlighted, and the last retrieval query projected into the same space as a
   star marker.
3. **Two debug endpoints** backing the above (see [docs/12](12_api_contract.md) and
   `/docs` for exact shapes):
   - `POST /diagnostics/rag-map` — scrolls all collections (sentinel meta point excluded),
     fits a 2-D PCA, returns projected points + metadata + redacted-by-default 120-char text
     previews, optional `query_text` → `query_point` overlay, `explained_variance`, and a
     `truncated` flag past a 20 000-point cap. `503 retrieval_unavailable` when Qdrant is
     unreachable.
   - `POST /diagnostics/chunk-texts` — full chunk text by `(collection, id)`, batched (≤50),
     request order preserved, `found=false` for unknown ids.

## Hidden-text policy (decision record)

Persisted turn diagnostics stay **metadata-only** — `TurnRetrievalDiagnostics` never carries
chunk text; that boundary is unchanged. The debug endpoints are the deliberate, explicit
exception, and the rule is **server-enforced**:

- Default (`include_hidden=false`): any chunk whose visibility is not `player` gets
  `text`/`text_preview = null` and `text_redacted = true`. The client never receives hidden
  text it did not explicitly request — the default cannot be defeated from dev-tools.
- `include_hidden=true` (the UI's "Reveal hidden" toggle) returns hidden text, badged in
  the UI so GM/character-private content is never mistaken for player-visible content.

Why this is compatible with the visibility invariant (docs/08 §invariants, docs/21): the
invariant protects against hidden text reaching **LLM providers** (especially cloud). These
endpoints never construct a prompt and never call a provider; they return data only to the
local single-user browser — the same posture as `GET /sessions/{id}/memories`, which has
always returned all visibilities as an authoring surface. The policy is pinned by
`tests/integration/test_api_rag_debug.py::test_rag_map_redacts_hidden_text_by_default`.

## Projection notes (PCA)

`app/rag/projection.py`, pure numpy (now a direct dependency):

- Fit = column-mean centering + `numpy.linalg.svd`, top-2 right singular vectors.
- **Deterministic**: component signs are fixed svd_flip-style (largest-|value| element of
  each component forced positive), so identical data → identical layout across calls.
- **Fit over the full unfiltered point set**, always; the UI filters client-side, so
  filtering or highlighting never reshuffles the layout, and the query vector is projected
  with the same fitted mean/components (coordinate-consistent overlay).
- Degenerate inputs handled explicitly: single point / collinear / all-identical inputs get
  zero-padded components and `explained_variance` entries of `0.0` (effective-rank check,
  `matrix_rank`-style tolerance); empty input never reaches the fitter (empty index returns
  an empty 200 response).
- Recomputed per request — SVD on a few-thousand × 384 matrix is tens of milliseconds, and
  the index mutates constantly (memory writes, consolidation), so a cache would buy little
  and cost invalidation complexity.
- Qdrant stores vectors L2-normalized under cosine distance, so the map plots normalized
  directions; `InMemoryVectorStore` stores raw vectors. Same layout semantics, different
  raw magnitudes — irrelevant for reading clusters, worth knowing when comparing stores.

## Reading the map

- Color = collection; hover a point for id/source/visibility/tags/importance + preview.
- Diamonds = last turn's selected candidates, triangles = scored-but-rejected, star = the
  retrieval query. A tight query-to-selection cluster with rejected points nearby means
  ranking (not recall) decided the turn; a query star far from everything means the query
  framing found no semantic neighborhood — look at `build_retrieval_query` inputs.
- `explained_variance` in the header says how much structure the 2-D view actually captures
  (384-D → 2-D is lossy; treat proximity as a hint, not proof — the ranked scores in the
  debug panel are the ground truth).

## Files

Backend: `app/rag/vector_store.py` (`StoredPoint`, `scroll_points`, `get_chunks`, both
stores + parity tests), `app/rag/projection.py`, `app/api/debug_routes.py`,
`app/api/schemas.py`. Frontend: `components/rag-debug-panel.component.ts` +
`rag-debug-model.ts`, `pages/vector-map.component.ts` + `vector-map-model.ts` (ECharts is
lazy-loaded with the map route only). Tests: `tests/unit/test_projection.py`,
`tests/unit/test_vector_store_scroll_dump.py`, `tests/integration/test_api_rag_debug.py`,
and the frontend model specs.
