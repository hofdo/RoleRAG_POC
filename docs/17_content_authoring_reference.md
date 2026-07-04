# 17 — Content Authoring Format Reference

> Reviewed: 2026-07-04 @ 571acc8

## Purpose

This document is the field-level reference for authoring scenario content: the JSON
worlds, scenes, and personas that a scenario pack ships. It records the schema exactly
as the code loads it ([app/domain/models.py](../app/domain/models.py),
[app/persistence/file_loader.py](../app/persistence/file_loader.py)), which fields are
player-visible versus GM/hidden, and how the hidden fields drive secret containment.

Config values, retrieval, and the HTTP surface are owned elsewhere — see
[05_rag_memory_design.md](05_rag_memory_design.md) for retrieval/memory and
[12_api_contract.md](12_api_contract.md) for endpoints. This doc covers only the
authored-content format.

## Content layout

A scenario pack is a content root (default `data`, or `CONTENT_ROOT`) with three
directories the catalog loader reads, plus an optional `documents/` directory for lore:

```
<content-root>/
  worlds/<world-id>.json
  scenes/<scene-id>.json          # dashes in the id become underscores in the filename
  personas/<persona-id>.json
  documents/
    manifest.json                 # optional: declares lore files + their visibility
    <lore>.md
```

The loader validates every file against the Pydantic models below; an invalid or missing
file is rejected at load time. Identifiers are restricted to ASCII alphanumerics plus `-`
and `_` (no path separators, no `..`) so an id cannot escape the content root.

The shipped example packs live under `data/scenarios/`
([bride-for-sarnhold](../data/scenarios/bride-for-sarnhold)), and are the best worked
reference alongside this schema.

## World

A world is the catalog record that ties a scene set and persona set together
(`DemoWorldRecord` in [app/persistence/file_loader.py](../app/persistence/file_loader.py)).

| Field | Type | Notes |
|-------|------|-------|
| `id` | string | Matches the filename `worlds/<id>.json`. |
| `name` | string | Display name. |
| `default_scene_id` | string | Scene a new session opens in unless overridden. |
| `persona_ids` | list of string | Personas available in this world. |
| `scene_ids` | list of string | Scenes available; a scene switch is validated against this list. |

The world file carries no hidden fields.

## Persona

A persona is a playable or non-playable character (`PersonaCard` in
[app/domain/models.py](../app/domain/models.py)). Only some fields are ever shown to the
player-facing actor; the rest are GM/hidden and exist to shape and constrain behavior
without being revealed.

| Field | Type | Visibility | Notes |
|-------|------|------------|-------|
| `id` | string | — | Matches the filename `personas/<id>.json`. |
| `name` | string | player-visible | |
| `role` | enum | player-visible | One of `narrator`, `npc`, `companion`, `antagonist`. |
| `public_description` | string | player-visible | The description the actor may reveal. |
| `speaking_style` | string | player-visible | Voice guidance for the actor. |
| `values` | list of string | player-visible | Fed to the actor prompt. |
| `goals` | list of string | player-visible | Fed to the actor prompt. |
| `fears` | list of string | GM/hidden | Not placed in the actor prompt. |
| `private_description` | string \| null | GM/hidden | Never reaches the actor; a containment fact. |
| `secrets` | list of string | GM/hidden | Containment facts (see below). |
| `forbidden_knowledge` | list of string | GM/hidden | Containment facts (see below). |
| `relationships` | map of string→string | GM/hidden | Author reference; not placed in the actor prompt. |

Fields marked player-visible are the only persona fields the actor generation prompt
receives ([app/orchestration/context_builder.py](../app/orchestration/context_builder.py)
composes it from `name`, `role`, `public_description`, `speaking_style`, `values`, and
`goals`). The GM/hidden fields never enter the actor or memory-extraction prompts on any
provider — they are structurally absent, not filtered out at emission time.

## Scene

A scene is the current situation the turn is generated against (`SceneState` in
[app/domain/models.py](../app/domain/models.py)).

| Field | Type | Visibility | Notes |
|-------|------|------------|-------|
| `id` | string | — | Matches the filename `scenes/<id>.json` (`-` → `_`). |
| `title` | string | player-visible | |
| `location` | string | player-visible | |
| `current_time` | string \| null | player-visible | Optional. |
| `active_personas` | list of string | — | Persona ids present in the scene. |
| `player_visible_summary` | string | player-visible | The summary the actor may work from. |
| `recent_events` | list of string | player-visible | Fed to the actor prompt. |
| `open_conflicts` | list of string | GM/hidden | Author reference; not placed in the actor prompt. |
| `active_quests` | list of string | GM/hidden | Author reference; not placed in the actor prompt. |
| `gm_private_summary` | string \| null | GM/hidden | The hidden truth of the scene; a containment fact. |

Only `title`, `location`, `current_time`, `player_visible_summary`, and `recent_events`
reach the actor prompt. `gm_private_summary` is the scene-level counterpart to a persona's
`secrets` — the GM-only truth the actor must never reveal.

## Player-visible vs. hidden, at a glance

| Player-visible (reaches the actor) | GM/hidden (never reaches the actor) |
|------------------------------------|-------------------------------------|
| Persona: `name`, `role`, `public_description`, `speaking_style`, `values`, `goals` | Persona: `private_description`, `fears`, `secrets`, `forbidden_knowledge`, `relationships` |
| Scene: `title`, `location`, `current_time`, `player_visible_summary`, `recent_events` | Scene: `open_conflicts`, `active_quests`, `gm_private_summary` |

## How hidden fields drive containment

The engine enforces the visible/hidden split with three independent mechanisms. Hidden
authored fields are never handed to the actor, so containment is primarily *structural*;
the remaining mechanisms are backstops against retrieved lore and against the model itself.

1. **Retrieval visibility filtering.** Actor context retrieval requests only
   `player` visibility ([app/rag/retriever.py](../app/rag/retriever.py) applies
   `RetrievalFilter.player_visible`). Lore ingested at `gm` visibility (see the manifest
   below) sits in the same `canon_lore` collection but is filtered out of the actor's
   retrieved chunks, so GM lore can inform authoring without leaking into play.
2. **`include_hidden` critic gate.** The critic is the only agent shown the hidden facts,
   and only on a local session — the actor stage never sees them. The critic receives
   `secrets`, `forbidden_knowledge`, and `gm_private_summary` *only when the session is
   bound to the local provider* (`include_hidden = provider == LOCAL`, in
   [app/orchestration/stages/critique.py](../app/orchestration/stages/critique.py)), so a
   cloud critic checks prose and consistency only and hidden authored content stays on the
   machine. See [06_local_cloud_model_strategy.md](06_local_cloud_model_strategy.md).
3. **`secret_guard` output-side scan.** A deterministic guard
   ([app/agents/secret_guard.py](../app/agents/secret_guard.py)) collects the hidden facts
   (`secrets`, `forbidden_knowledge`, `private_description`, `gm_private_summary`) and
   scans the reply: it redacts any verbatim or per-sentence echo and flags likely
   paraphrase by content-word overlap. This runs regardless of provider and catches the
   case where the model *confabulates* a secret the prompt never contained.

The takeaway for authors: write `secrets`, `forbidden_knowledge`, `private_description`,
and `gm_private_summary` freely as the GM truth. They shape the critic's judgment and the
containment guard's target set without ever being placed in front of the actor.

## Manifest lore and auto-ingest

Freeform lore (`.md` / `.txt`) is declared in `documents/manifest.json` and indexed into
the `canon_lore` vector collection ([app/rag/ingestion.py](../app/rag/ingestion.py),
[app/content/validator.py](../app/content/validator.py)). Each manifest entry sets:

| Field | Notes |
|-------|-------|
| `path` | File under `documents/`. |
| `visibility` | `player` for lore the actor may retrieve; `gm` for author-only truth. |
| `source_type` | Free label (e.g. `lore`, `history`). |
| `tags` | Retrieval/boost tags. |
| `world_id` | Scopes the chunks to a world. |

The CLI `start-session` command best-effort auto-ingests the scenario's manifest lore on
session creation ([app/cli.py](../app/cli.py)). It is **idempotent** (re-running re-indexes
the same lore without duplicating it) and **fail-open** (an unreachable vector store or
missing embedding backend only warns; the session is still created). Pass
`--skip-lore-ingest` to opt out. API/SPA session creation does **not** auto-ingest — run
the `ingest-scenario-lore` CLI command (or `ingest` for a single document) to index lore
for those sessions.

## The `bride-for-sarnhold` containment probe

`data/scenarios/bride-for-sarnhold`
([here](../data/scenarios/bride-for-sarnhold)) is the second shipped pack and the
secret-containment probe scenario. It is deliberately built so that the interesting facts
are hidden: three personas (`lady-iseult`, `chancellor-bram`, `handmaid-nessa`) with
populated `secrets`, `forbidden_knowledge`, and `private_description`; a scene whose
`gm_private_summary` states the truth the court does not know; and a GM-only
`documents/gm_truth.md` ingested at `gm` visibility. The player-visible summary presents a
"spoiled heiress" the hidden fields contradict, which is exactly the pressure the secret
probe measures. The containment-probe tooling that drives this pack is documented with the
rest of the eval harnesses.

## Fields that no longer exist

An earlier background guide
([personal_python_roleplaying_rag_implementation_guide.md](personal_python_roleplaying_rag_implementation_guide.md))
listed two fields that are **not** in the current model and must not be authored:

- `PersonaCard.allow_private_context_for_actor` — there is no such field. Whether hidden
  persona content reaches an agent is decided by the engine (see containment above), not by
  a per-persona flag.
- `SceneState.tags` — there is no scene-level `tags` field. Tagging lives on ingested lore
  chunks and on memory episodes, not on the scene record.

An author copying the schema from that older guide would write invalid-per-current-model
files that fail validation on load. This document supersedes it as the authoring reference.
