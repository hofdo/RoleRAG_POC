# Implementation plans

Dated implementation plans authored for the superpowers agentic workflow. Each plan
is a task-by-task specification an agent executes end to end; once shipped, it becomes
a historical record of *how* a batch of work landed, not a source of current state — for
that, read the living docs (see the [docs hub](../../README.md)).

| Plan | Status |
|------|--------|
| [2026-07-01-play-experience-v1.2](2026-07-01-play-experience-v1.2.md) | EXECUTED (merged to main 2026-07-01/02) |
| [2026-07-02-session-bound-provider](2026-07-02-session-bound-provider.md) | EXECUTED (merged to main 2026-07-02/03) |

## Executed-status convention

When every task in a plan has merged, add a status banner directly above the
`> **For agentic workers:**` blockquote:

```
> **Status: EXECUTED** — all N tasks merged to main <dates> (<commit range>).
```

Note any parts later superseded by a subsequent plan so an agent does not trust stale
code snippets. Without this banner an agent following the header instruction would
re-execute completed work.

## Mandatory final task

Every new plan MUST end with a **"Sweep living docs"** task so shipped behavior and its
documentation move together. That task updates the living docs and refreshes each touched
doc's `> Reviewed: YYYY-MM-DD @ <short-sha>` header. The living-docs set is enumerated
once, in the [docs hub](../../README.md)'s "Doc maintenance & freshness" section — sweep
that list rather than a copy of it here. When behavior or config changed, also update the
root `CHANGELOG.md` entry and `.env.example` comments (neither carries a `Reviewed:`
header).

The 2026-07-03 documentation overhaul added this rule after two plans shipped without a
docs sweep and left the living docs drifting.
