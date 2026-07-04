# 20 — Playing RoleRAG

> Reviewed: 2026-07-04 @ 571acc8

## Purpose

Every other doc in this folder addresses a developer or a coding agent. This one is for
whoever sits down to *play*: it walks the web play surface at `/app/play`, explains what the
side panels mean, describes what a "controlled failure" looks like mid-scene and what to do
about it, and ends with a symptom → cause → fix troubleshooting table.

It assumes the stack is already running. If it is not, see the root
[README](../README.md) for setup and `make dev`, and
[18_security_privacy_and_backups](18_security_privacy_and_backups.md) for the "keep it on
the local network" posture. The play surface is a **thin client** over the FastAPI backend —
the browser holds no authoritative state, so anything you see in play you can also reach over
the API ([12_api_contract](12_api_contract.md)) or the CLI.

## The play surface at a glance

Open `http://127.0.0.1:8000/app/` and go to the **Play** tab. The screen has one main column
and a side column:

| Area | What it is |
|------|------------|
| Setup picker (top) | Before a session exists: choose **World**, **Scene**, **Persona**, **Model** (Local/Cloud), and a player name, then **Start session**. Once a session is active it collapses to a scene switcher and a next-turn persona selector. |
| Transcript | The running conversation: your turns and the narrator/character replies, newest work streaming in live. |
| Composer | The text box plus **Send** and **Reroll last**. While a turn runs, a small progress line shows the current pipeline stage. |
| Memory panel (side) | The durable memories the system has written for this session (a **Refresh** button re-pulls them). |
| Canon panel (side) | The session's standing facts — author-pinned truths plus ones you add or delete by hand. |

A separate **RAG Inspector** tab (`/app/inspector`) is a read-only diagnostic view for
inspecting any past turn's retrieval ranking, timings, critic status, and warnings. It is
optional and safe to ignore during normal play.

## Starting or resuming a session

**Start fresh.** Pick a world, scene, persona, and model, then **Start session**. The
**Model** choice is bound to the session for its whole lifetime — you cannot switch a session
between Local and Cloud later; start a new session to change providers. What the **Cloud**
option does depends on the server's `CLOUD_MODE`:

- `off` — Cloud is not offered / creation is refused (`cloud_unavailable`); use Local.
- `ask` — you confirm the cloud choice **once**, at session creation (a browser confirm
  dialog). There is no per-turn confirmation.
- `auto` — Cloud binds silently.

See [06_local_cloud_model_strategy](06_local_cloud_model_strategy.md) for the full routing
rules.

**Resume.** If recent sessions exist, a **Resume session** dropdown and **Resume** button
appear under the setup form. Resuming reloads that session's transcript and panels so you can
keep going where you left off.

## Playing a turn

Type into the composer and **Send**. The reply is delivered over **buffered SSE**: the
backend runs the full turn pipeline first and streams live *stage* progress ("Retrieving
memories", "Drafting reply", "Critic reviewing", "Repairing draft", "Saving turn", …), then
delivers the finished reply text. You get progress feedback without seeing half-formed,
un-reviewed prose — the critic runs *before* any reply text is emitted.

Things you can do during a session:

- **Reroll last** — deletes your most recent turn (both your message and its reply) so you can
  retype it and get a different result. This is a real deletion: the turn's memories are
  unindexed too. There is no undo for a reroll.
- **Switch scene** — pick a different scene from the same world and **Switch scene**. Only
  scenes defined in the current world are valid.
- **Persona (next turn)** — choose a different persona to answer your *next* turn. The switch
  is validated against the world's persona list and only becomes durable if that turn is saved.

## The side panels

- **Memory.** These are the durable facts the memory curator has written from the conversation
  (a promise you made, a name you learned, a decision the group reached). They are curated in
  the background: right after a turn the panel may not yet show that turn's new memory —
  **Refresh** re-pulls the list once curation has caught up. Retrieval pulls from this store to
  keep later replies consistent. See
  [05_rag_memory_design](05_rag_memory_design.md) for how memory and retrieval work.
- **Canon.** Standing facts for the session. Some are author-pinned by the scenario; you can
  also **Add** your own or delete one with the ✕ button. Canon is fed to the model every turn
  as ground truth, independent of vector retrieval — use it to nail down details the story must
  never contradict.

## When a turn fails: controlled failures

Sometimes a turn ends without a normal reply and you see an error line in the composer instead
of new story text. Most often this is a **controlled failure**, not a crash: the actor drafted
a reply, the critic rejected it, one bounded same-provider repair pass also failed, and rather
than emit prose it does not trust, the system **fails closed** and withholds the reply. The
turn is still recorded (with `outcome = controlled_failure`); nothing corrupt is shown to you.

This is expected at a low rate. A 30-turn live run with **Cloud off** measured roughly a
**6.7% fail-closed rate** (about 2 turns in 30), concentrated on long, late-session turns.
What to do when it happens:

- **Just resend / reroll.** Failures are usually transient. **Reroll last** (or simply send a
  slightly reworded turn) frequently succeeds on the next attempt.
- **Give the model more room.** They cluster on long sessions where the recent-dialogue context
  gets clipped. Shorter, more focused turns help.
- **Allow Cloud.** These failures happen because there is no cross-provider fallback: with
  `CLOUD_MODE=ask` or `auto` and a cloud-bound session, the stronger cloud model would likely
  have carried the turn. Starting a Cloud session is the reliable way to reduce them.

A controlled failure is different from an infrastructure error (model server down, Qdrant
unreachable) — those show up as `provider_unavailable` / `provider_timeout` style errors and
are covered in the troubleshooting table below.

## Warnings

A yellow/muted warning line (top of the play area, or the **Warnings** list in the RAG
Inspector) means the turn completed but something ran in degraded mode — most commonly
retrieval or memory writing was skipped because Qdrant or embeddings were unavailable, or the
recent-dialogue context was clipped for length. Warnings are informational: play continues.
Persistent retrieval/memory warnings usually mean Qdrant needs attention (see the table).

## Troubleshooting / FAQ

Run the CLI commands from the repo root (`python -m app.cli <command>`, or `rolerag <command>`
if installed). `python -m app.cli --help` lists everything.

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Turn ends with no reply, error line in composer | Controlled failure — critic rejected the draft and the bounded repair also failed (fail-closed) | **Reroll last** and resend; try a shorter turn; or start a **Cloud** session (`CLOUD_MODE=ask`/`auto`) — see [Controlled failures](#when-a-turn-fails-controlled-failures) |
| `provider_unavailable` / `provider_timeout`, or every turn fails immediately | The bound model server (local llama-server, or the cloud provider) is unreachable or slow | For Local: confirm the model server is running and reachable, then check with `python -m app.cli doctor --check-local-provider`. Restart your `llama-server`. For Cloud: check the provider key/URL in `.env`. |
| Repeated "retrieval skipped" / "memory not written" warnings; Memory panel stays empty | Qdrant (the vector index) is down or unreachable — turns are fail-open, so play continues degraded | Confirm Qdrant is up (`docker compose up qdrant`), then verify with `python -m app.cli doctor --check-qdrant` |
| Panels/replies ignore facts you know are stored; retrieval seems stale after a restore or import | The vector index drifted from SQLite (Qdrant is a rebuildable derived index) | Rebuild it: `python -m app.cli reindex-memories --session-id <id>`. To drop and rebuild a whole collection, `python -m app.cli reset-index` |
| General "is my setup healthy?" check | — | `python -m app.cli doctor` (config + storage), add `--check-qdrant --check-local-provider` to probe live services; `python -m app.cli health` for versioned status |
| `cloud_unavailable` when starting a Cloud session | Server is running with `CLOUD_MODE=off` | Start a **Local** session, or set `CLOUD_MODE=ask`/`auto` and provide provider credentials — see [06_local_cloud_model_strategy](06_local_cloud_model_strategy.md) |
| Can't Start session / scene or persona rejected | Selection isn't valid for the chosen world | Pick a world/scene/persona from the catalog dropdowns (only in-world ids are accepted) |

For deeper live-verification and evaluation tooling see
[19_verification_and_eval_tooling](19_verification_and_eval_tooling.md); for backup and
restore see [18_security_privacy_and_backups](18_security_privacy_and_backups.md).
