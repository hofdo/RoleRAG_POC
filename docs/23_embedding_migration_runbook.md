# 23 — Embedding-Model Migration Runbook

> Reviewed: 2026-07-10 @ 24d4aab

How to change `EMBEDDING_MODEL` safely. This is change item (1) of
[docs/22 § P1.2](22_rag_scaling_roadmap.md#p12-embedding-model-upgrade-path-multilingual);
the multilingual benchmark and any default swap stay gated on the P0.4 corpus.

## Why a runbook at all

Qdrant is a **derived, rebuildable index** — SQLite stays authoritative and is untouched by
everything below (invariant #1 in [CLAUDE.md](../CLAUDE.md)). But vectors from two different
embedding models must never share a collection:

- **Different dimension** → `VectorStoreDimensionMismatch` at `ensure_collection` (loud, safe).
- **Same dimension, different model** (e.g. the default `all-MiniLM-L6-v2` → the 384-dim
  candidate `paraphrase-multilingual-MiniLM-L12-v2`) → without a guard this would *silently*
  mix incompatible vector spaces. The P1.4 fingerprint
  ([docs/22 § P1.4](22_rag_scaling_roadmap.md#p14-embedding-model-identity-fingerprint-adversarially-verified-new),
  shipped 2026-07-09) closes this: each collection stores its model identity, and any write
  under a different `EMBEDDING_MODEL` raises `VectorStoreModelMismatch` pointing here.

If you hit `VectorStoreModelMismatch`, you skipped the reset step — resume at step 3.

## Choosing a model

Use the verified candidate list in
[docs/22 § P1.2](22_rag_scaling_roadmap.md#p12-embedding-model-upgrade-path-multilingual)
(fastembed 0.8.0 support checked there; several obvious candidates are *not* supported).
**Symmetric models only** for now: asymmetric families (e.g. `multilingual-e5-large`) need
query/passage prefixes the `EmbeddingProvider` protocol does not support yet — see the
prefix caveat in the same section. Swapping the shipped default requires P0.4 evidence,
per the measure-first workflow; swapping your own install just requires this runbook.

## Procedure

All commands from the repo root with the venv active. `rolerag` is the same entry point as
`python -m app.cli`.

1. **Quiesce.** Stop the API server; finish or abandon in-flight turns. (Live-turn memory
   indexing is fail-open — a mismatch mid-play becomes a write-blocking turn warning, not a
   crash — but don't migrate under load.)
2. **Optional insurance:** `python -m app.cli backup` (SQLite only; vectors are rebuilt
   below, so they are deliberately not part of backups).
3. **Set the new model** in `.env`: `EMBEDDING_MODEL=<new model key>`.
4. **Drop all collections** (also clears the P1.4 fingerprints — a stale fingerprint would
   otherwise brick the rebuild):

   ```bash
   python -m app.cli reset-index --collection all --yes
   ```

5. **Re-ingest lore** for every scenario pack in use:

   ```bash
   python -m app.cli ingest-scenario-lore --content-root data/scenarios/<pack>
   ```

   (Standalone documents go through `python -m app.cli ingest <path> --visibility ...
   --source-type ...` with the same scope flags used originally. `start-session` also
   auto-ingests a pack's lore on next session start, but don't rely on that for existing
   sessions — reindex explicitly.)
6. **Rebuild memory vectors per session** — this restores both `session_memory` and the
   cross-session `persona_memory` dual-writes from authoritative SQLite:

   ```bash
   python -m app.cli list-sessions --limit 1000   # session ids
   python -m app.cli reindex-memories --session-id <id>   # for each
   ```

7. **Verify:**

   ```bash
   python -m app.cli doctor --check-qdrant
   ```

   The doctor's read-only fingerprint scan must pass for all three collections; spot-check
   retrieval with `python -m app.cli retrieve-debug` on a known query if you want more.

## Failure modes

| Symptom | Cause | Remedy |
|---------|-------|--------|
| `VectorStoreDimensionMismatch` during step 5/6 | Step 4 skipped, or a collection was recreated under the old model between steps | Rerun step 4, resume at step 5 |
| `VectorStoreModelMismatch` during step 5/6 | Same — fingerprint still carries the old model | Rerun step 4, resume at step 5 |
| `reindex-memories` fails for one session | Anything transient (Qdrant down, model download) | Fix cause, rerun that session — reindexing is idempotent (`id=memory.id` upserts) |
| Retrieval quality drops after swap | The new model is simply worse for this corpus | Swap back (same runbook) — SQLite lost nothing |

Rollback is the same procedure with the old `EMBEDDING_MODEL` value.
