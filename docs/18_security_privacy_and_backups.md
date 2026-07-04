# 18 — Security, Privacy & Backups

> Reviewed: 2026-07-04 @ 571acc8

## Purpose

This document records the security and privacy posture of a running RoleRAG instance
and the backup/restore runbook for its data. RoleRAG is a single-user, local-network
personal tool; it is not hardened for multi-tenant or public deployment. Read this before
exposing the app on any network you do not fully control, and keep it handy if you ever
need to recover a corrupted database.

The HTTP surface itself is owned by [12_api_contract.md](12_api_contract.md); config
values live in [`.env.example`](../.env.example) + [app/config.py](../app/config.py). This
doc covers only the deployment posture and the data-recovery procedure.

## Threat model & assumptions

RoleRAG is designed for **one operator running it on their own machine or a trusted LAN**.
Two consequences follow, and both are load-bearing:

- **No authentication and no CORS restriction.** The FastAPI app
  ([app/main.py](../app/main.py)) mounts the API router and the SPA with no auth
  dependency and no CORS middleware. Every endpoint — including the ones that read GM
  secrets-adjacent authoring surfaces (session memories, scene state) and the destructive
  scene/turn mutations — is reachable by anyone who can reach the port.
- **The interactive API reference is served openly.** FastAPI publishes Swagger UI at
  `/docs`, ReDoc at `/redoc`, and the machine-readable schema at `/openapi.json`. These
  are always accurate to the running build and are not gated.

**Never expose the port publicly.** The Docker image binds uvicorn to `0.0.0.0`
([Dockerfile](../Dockerfile)) so the container is reachable from the host, and
[docker-compose.yml](../docker-compose.yml) publishes `${DEV_API_PORT:-8000}:8000` (plus
Qdrant on `6333`). On a laptop behind a home router this is fine; on a public IP or a
shared host it hands the whole world your roleplay data and a mutation API. If you must
reach it remotely, put it behind a VPN or an authenticating reverse proxy — do not publish
the raw port.

## What leaves the machine

By default nothing does: with `CLOUD_MODE=off` (the compose default) every task runs
against the local model server, and no roleplay content is sent to a third party.

Cloud egress happens **only** when a session is created with the cloud provider and
`CLOUD_MODE` permits it. A cloud-bound session is immutable once created — the provider is
chosen once at `POST /sessions` and every task (actor, repair, critic, memory extraction)
for that session runs on it (see [06_local_cloud_model_strategy.md](06_local_cloud_model_strategy.md)).
When that happens, the prompts sent to the cloud provider contain the player-visible
roleplay content — scene summaries, dialogue, retrieved memories — but **not** the
authored hidden fields:

- Actor and memory-extraction prompts never contain persona `secrets`,
  `forbidden_knowledge`, `private_description`, or the scene's `gm_private_summary` on any
  provider — those fields are structurally absent from those prompts, not merely filtered.
- The critic is the only agent shown hidden facts, and only when its turn runs **locally**:
  `include_hidden` is gated to `route.provider == LOCAL`
  ([app/orchestration/stages/critique.py](../app/orchestration/stages/critique.py)), so on
  a cloud-bound session the critic prompt omits them as well.

The net effect: on a cloud session, GM-only authored fields do not leave the machine; the
player-facing narrative surface does. See
[17_content_authoring_reference.md](17_content_authoring_reference.md) for the
visible-vs-hidden field split.

The cloud API key sits in `.env` as `CLOUD_LLM_API_KEY`. Treat `.env` as a secret file:
it is git-ignored, but it is plaintext on disk, so protect it with the same care as any
other credential.

## Where data lives on disk

All authoritative state is a **plaintext SQLite database** — sessions, turns, memory
episodes, and canon facts — with no encryption at rest:

| Path | Contents | Notes |
|------|----------|-------|
| `data/rolerag.db` | Authoritative SQLite DB (`DATABASE_PATH`) | Runs in WAL mode |
| `data/rolerag.db-wal`, `data/rolerag.db-shm` | WAL sidecar files | Live alongside the DB while the server/CLI is running |
| `data/backups/` | Manual + safety DB snapshots (`rolerag-<timestamp>.db`) | See below |
| `data/qdrant/` | Qdrant vector storage (Docker volume) | Derived index, rebuildable |
| `eval-runs/` (`EVAL_RESULTS_DIR`) | Eval/bake-off run artifacts | May contain full transcripts |
| `.env` | Config including `CLOUD_LLM_API_KEY` | Plaintext secret |

The DB is opened with `PRAGMA journal_mode = WAL`
([app/persistence/sqlite.py](../app/persistence/sqlite.py)) so the CLI and the API server
can share the file without "database is locked" errors; the `-wal`/`-shm` sidecars are a
normal part of that mode. Qdrant collections (`canon_lore`, `session_memory`,
`persona_memory`) and the eval-run directory are derived data — they can be regenerated
and are not the source of truth.

## Backups

RoleRAG writes backups two ways, both via the same online-consistent SQLite copy
([`_backup_database`](../app/cli.py), which uses the SQLite backup API `source.backup(target)`):

- **Manual:** `rolerag backup [--output-dir DIR]` writes `data/backups/rolerag-<timestamp>.db`.
- **Automatic safety snapshot:** the destructive SQLite CLI commands — `delete-session`
  and `reset-db` — take a safety backup first and print its path before deleting anything.
  Note that `reset-index` drops only the Qdrant collections and does not touch or snapshot
  the SQLite DB.

Two things to know:

- **Snapshots are self-consistent.** Because the SQLite backup API copies a consistent
  view of the live database, a snapshot does not need the `-wal`/`-shm` sidecars — the DB
  file alone is complete.
- **Vectors are not backed up.** Backups deliberately exclude Qdrant; the vector index is
  rebuilt from SQLite after a restore (see the runbook). The server never takes an
  automatic backup on its own — only the destructive CLI paths and explicit `rolerag backup`
  do.
- **There is no retention policy.** Nothing prunes `data/backups/`; snapshots accumulate
  until you delete old ones yourself.

## Restore runbook

To recover from a corrupted or unwanted database state, restore a snapshot and rebuild the
derived index:

1. **Stop the server** (and any CLI process) so nothing holds the SQLite file open. This
   also lets SQLite flush and remove the WAL sidecars cleanly.
2. **Copy the snapshot over the live DB.** Replace `data/rolerag.db` with the chosen
   `data/backups/rolerag-<timestamp>.db`. Snapshots taken via the backup API are
   self-consistent, so the single `.db` file is all you need.
3. **Delete stale sidecars.** Remove any leftover `data/rolerag.db-wal` and
   `data/rolerag.db-shm` files so they do not get replayed on top of the restored DB.
4. **Rebuild the derived Qdrant index.** The backup did not include vectors, so reindex
   each restored session:

   ```
   rolerag reindex-memories --session-id <id>
   ```

   `reindex-memories` operates per session (list the ids with `rolerag list-sessions`).
   Canon lore is repopulated by re-ingesting the scenario lore
   (`rolerag ingest ...` / auto-ingest on `start-session`); see
   [17_content_authoring_reference.md](17_content_authoring_reference.md).
5. **Start the server** and verify. `rolerag doctor` checks that SQLite and Qdrant are
   reachable and consistent.

If Qdrant was reset independently (`rolerag reset-index`), the SQLite DB is untouched and
you only need step 4 to rebuild the vectors.

## See also

- [12_api_contract.md](12_api_contract.md) — the HTTP surface these assumptions apply to.
- [06_local_cloud_model_strategy.md](06_local_cloud_model_strategy.md) — provider binding
  and cloud modes.
- [17_content_authoring_reference.md](17_content_authoring_reference.md) — which authored
  fields are hidden and how containment keeps them off the wire.
