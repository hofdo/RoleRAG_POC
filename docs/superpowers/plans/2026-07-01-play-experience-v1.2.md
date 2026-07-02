# Play Experience v1.2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the eight highest-leverage gaps from the 2026-07-01 in-depth analysis: data durability, contradiction memory loss, per-turn latency waste, dead-air turns, session continuity, reroll, campaign stasis, and cross-session NPC memory.

**Architecture:** Every task is an independently shippable slice against the existing stage pipeline (`TurnOrchestrator` + injectable stages), the FastAPI routes, and the Angular signals store. No new services, no new dependencies, no schema-breaking changes — only additive columns-free SQLite pragmas, new repository methods, one new SSE frame type, and small SPA wiring.

**Tech Stack:** Python 3.12, FastAPI, SQLite, Qdrant (via existing `VectorStore` abstraction), fastembed, Angular 19 signals SPA, pytest, karma/jasmine (`ng test`).

## Global Constraints

- Backend gate after every task: `make check` (ruff + mypy + pytest). For tasks touching orchestration/memory/retrieval also run `.venv/bin/python -m app.evals.regression_runner` and `make smoke`.
- Frontend gate after every task touching `frontend/`: `cd frontend && npx ng build` (plus `npx ng test --watch=false` when specs change).
- Work on a branch: `git checkout -b feat/v1.2-play-experience` before Task 1.
- One commit per task minimum, Conventional Commits format.
- All new code follows existing idioms: frozen dataclass stage results, Protocol-typed repositories, warning-string degradation (never fail a turn from an ancillary path), signals store in the SPA (no NgRx).
- Deliberate shortcuts carry a `# ponytail:` comment naming the ceiling and upgrade path.
- Do not change defaults validated by live acceptance: `critic_gating`/`curator_gating` stay `"always"`, `session_memory_max_episodes` stays `0` (see docs/BACKLOG.md #29).

---

### Task 1: SQLite durability — WAL, busy_timeout, `rolerag backup`, auto-snapshot before destructive ops

**Files:**
- Modify: `app/persistence/sqlite.py:8-14`
- Modify: `app/cli.py` (add `backup` command; hook into `delete-session` ~line 788 and `reset-db` ~line 986)
- Test: `tests/unit/test_sqlite.py`, `tests/integration/test_cli_sessions.py`

**Interfaces:**
- Consumes: `connect_sqlite(database_path)`, existing `_open_repositories()` / `get_settings()` in `app/cli.py`.
- Produces: `_backup_database(output_dir: Path = Path("data/backups")) -> Path` in `app/cli.py`; CLI command `rolerag backup [--output-dir PATH]`. Later tasks rely on WAL being on (concurrent API + background writes in Task 8).

- [ ] **Step 1: Write the failing test**

In `tests/unit/test_sqlite.py`:

```python
def test_connect_sqlite_enables_wal_and_busy_timeout(tmp_path):
    connection = connect_sqlite(tmp_path / "wal.db")
    try:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    finally:
        connection.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_sqlite.py::test_connect_sqlite_enables_wal_and_busy_timeout -v`
Expected: FAIL — journal_mode is `delete`, busy_timeout is `0`.

- [ ] **Step 3: Implement pragmas**

In `app/persistence/sqlite.py`, inside `connect_sqlite` after the `PRAGMA foreign_keys` line:

```python
    connection.execute("PRAGMA foreign_keys = ON")
    # CLI and API server share this file; WAL + busy_timeout removes the
    # "database is locked" failure mode between them.
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_sqlite.py -v`
Expected: PASS (all tests in file).

- [ ] **Step 5: Write the failing CLI backup test**

In `tests/integration/test_cli_sessions.py` (mirror the file's existing CliRunner + env fixtures — it already isolates `DATABASE_PATH` per test):

```python
def test_backup_writes_timestamped_copy(tmp_path, ...existing fixtures...):
    # create one session via the existing start-session invocation pattern in this file
    result = runner.invoke(app, ["backup", "--output-dir", str(tmp_path / "backups")])
    assert result.exit_code == 0, result.output
    copies = list((tmp_path / "backups").glob("rolerag-*.db"))
    assert len(copies) == 1
    import sqlite3
    check = sqlite3.connect(copies[0])
    assert check.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
    check.close()
```

- [ ] **Step 6: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/integration/test_cli_sessions.py::test_backup_writes_timestamped_copy -v`
Expected: FAIL — `No such command 'backup'`.

- [ ] **Step 7: Implement `_backup_database` + `backup` command**

In `app/cli.py` (imports: `sqlite3`, `from datetime import UTC, datetime` — add only what isn't already imported):

```python
def _backup_database(output_dir: Path = Path("data/backups")) -> Path:
    """Online-consistent copy of the SQLite DB. Vectors are excluded on purpose:
    Qdrant collections rebuild from SQLite via reindex-memories / ingest."""
    settings = get_settings()
    source = connect_sqlite(settings.database_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    destination = output_dir / f"rolerag-{stamp}.db"
    target = sqlite3.connect(destination)
    try:
        with target:
            source.backup(target)
    finally:
        target.close()
        source.close()
    return destination


@app.command()
def backup(
    output_dir: Annotated[
        Path, typer.Option(help="Directory for backup files")
    ] = Path("data/backups"),
) -> None:
    destination = _backup_database(output_dir)
    typer.secho(f"Backup written: {destination}", fg=typer.colors.GREEN)
```

- [ ] **Step 8: Auto-snapshot before destructive ops**

In `delete_session` (app/cli.py:789) and `reset_db` (app/cli.py:987), immediately after the confirmation prompt and before any deletion:

```python
    backup_path = _backup_database()
    typer.secho(f"Safety backup: {backup_path}", fg=typer.colors.YELLOW)
```

- [ ] **Step 9: Run tests and gates**

Run: `.venv/bin/python -m pytest tests/integration/test_cli_sessions.py tests/unit/test_sqlite.py -v` then `make check`
Expected: PASS. (If a `delete-session`/`reset-db` test asserts on exact stdout, extend its expectation with the new "Safety backup:" line.)

- [ ] **Step 10: Commit**

```bash
git add app/persistence/sqlite.py app/cli.py tests/unit/test_sqlite.py tests/integration/test_cli_sessions.py
git commit -m "feat(persistence): WAL + busy_timeout, backup command, auto-snapshot before destructive ops"
```

---

### Task 2: Contradiction-dedup fix — reversal memories are never dropped as duplicates

**Files:**
- Modify: `app/memory/deterministic_extractor.py:102-116`
- Test: `tests/unit/test_deterministic_extractor.py`

**Interfaces:**
- Consumes: `content_terms(text) -> set[str]` from `app/rag/ranking.py`.
- Produces: unchanged signature `is_covered_by_summaries(candidate_summary, summaries, *, threshold=0.5) -> bool` — both call sites (`app/orchestration/stages/memory.py:166` fallback filter and `MemoryDeduplicator.drop_duplicates` in `app/orchestration/stages/memory_dedup.py`) get the fix for free.

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_deterministic_extractor.py`:

```python
def test_reversal_candidate_is_not_covered_by_the_fact_it_contradicts():
    assert not is_covered_by_summaries(
        "Mira no longer trusts the player and revoked the safehouse offer",
        ["Mira trusts the player and offered the player the safehouse"],
    )


def test_identical_summary_is_still_covered():
    assert is_covered_by_summaries(
        "The player promised to guard the ledger",
        ["The player promised to guard the ledger"],
    )


def test_reversal_present_in_both_summaries_is_still_covered():
    assert is_covered_by_summaries(
        "Mira no longer trusts the player",
        ["Mira no longer trusts the player and revoked the safehouse offer"],
    )
```

- [ ] **Step 2: Run tests to verify the first fails**

Run: `.venv/bin/python -m pytest tests/unit/test_deterministic_extractor.py -v -k reversal or identical`
Expected: `test_reversal_candidate_...` FAILS (currently dropped as duplicate — 5/7 term overlap, `not` is a stopword); the other two PASS.

- [ ] **Step 3: Implement the negation guard**

In `app/memory/deterministic_extractor.py`, add below `COVERAGE_THRESHOLD`:

```python
_REVERSAL_MARKERS = re.compile(
    r"\b(?:not|never|no\s+longer|stopped|refused?|betray(?:ed|s)?|broke|broken"
    r"|revoked?|withdrew|withdrawn|abandoned|died|dead|ended)\b",
    re.IGNORECASE,
)


def _reversal_markers(text: str) -> set[str]:
    return {" ".join(match.split()).lower() for match in _REVERSAL_MARKERS.findall(text)}
```

Replace the loop body of `is_covered_by_summaries`:

```python
    candidate_terms = content_terms(candidate_summary)
    if not candidate_terms:
        return True
    candidate_markers = _reversal_markers(candidate_summary)
    for summary in summaries:
        overlap = len(candidate_terms & content_terms(summary))
        if overlap / len(candidate_terms) < threshold:
            continue
        # A reversal marker present in the candidate but absent from the covering
        # summary means this is a state CHANGE, not a duplicate — always write it.
        if candidate_markers - _reversal_markers(summary):
            continue
        return True
    return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_deterministic_extractor.py -v`
Expected: PASS (all).

- [ ] **Step 5: Run gates including memory regressions**

Run: `make check && .venv/bin/python -m pytest tests/evals -v && .venv/bin/python -m app.evals.regression_runner`
Expected: PASS — dedup evals must not regress (a looser dedup only ever writes MORE memories).

- [ ] **Step 6: Commit**

```bash
git add app/memory/deterministic_extractor.py tests/unit/test_deterministic_extractor.py
git commit -m "fix(memory): never drop reversal/negation memories as duplicates of the fact they contradict"
```

---

### Task 3: Process-wide embedding/vector singletons — stop reloading the ONNX model every turn

**Files:**
- Modify: `app/composition.py:105-110`
- Test: `tests/unit/test_config.py` or new `tests/unit/test_composition_cache.py`

**Interfaces:**
- Consumes: `Settings.embedding_model: str`, `Settings.qdrant_url: str`.
- Produces: `build_embedding_provider(settings)` / `build_vector_store(settings)` unchanged signatures, now returning process-cached instances. `AppServices.close()` continues to close only the SQLite connection (the cached clients are shared and stay open).

- [ ] **Step 1: Write the failing test**

New file `tests/unit/test_composition_cache.py`:

```python
from app.composition import build_embedding_provider, build_vector_store
from app.config import Settings


def test_embedding_provider_is_cached_per_model():
    settings = Settings()
    assert build_embedding_provider(settings) is build_embedding_provider(settings)


def test_vector_store_is_cached_per_url():
    settings = Settings()
    assert build_vector_store(settings) is build_vector_store(settings)
```

(`FastEmbedEmbeddingProvider` loads its model lazily and `QdrantVectorStore` doesn't connect at construction, so this runs offline.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_composition_cache.py -v`
Expected: FAIL — fresh instances per call.

- [ ] **Step 3: Implement the cache**

In `app/composition.py` (add `from functools import lru_cache` to imports):

```python
@lru_cache(maxsize=4)
def _cached_embedding_provider(model_name: str) -> FastEmbedEmbeddingProvider:
    return FastEmbedEmbeddingProvider(model_name=model_name)


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    # ponytail: process-wide cache — the fastembed ONNX model was reloaded from
    # disk on every HTTP request otherwise; move to app.state lifespan if
    # per-settings isolation ever matters
    return _cached_embedding_provider(settings.embedding_model)


@lru_cache(maxsize=4)
def _cached_vector_store(url: str) -> QdrantVectorStore:
    return QdrantVectorStore(url=url)


def build_vector_store(settings: Settings) -> VectorStore:
    return _cached_vector_store(settings.qdrant_url)
```

- [ ] **Step 4: Run test and full suite**

Run: `.venv/bin/python -m pytest tests/unit/test_composition_cache.py -v && make check`
Expected: PASS. If any test relied on fresh `QdrantVectorStore` instances (search for direct `build_vector_store` use in tests), add `_cached_vector_store.cache_clear()` / `_cached_embedding_provider.cache_clear()` in that test's setup rather than weakening the cache.

- [ ] **Step 5: Commit**

```bash
git add app/composition.py tests/unit/test_composition_cache.py
git commit -m "perf(composition): cache embedding provider and vector store across requests"
```

---

### Task 4: Stage-progress SSE — the minute-long wait shows live pipeline stages

**Files:**
- Modify: `app/orchestration/turn_orchestrator.py` (run_turn signature + stage emission)
- Modify: `app/api/schemas.py` (two new payloads)
- Modify: `app/api/sse.py` (two new frame serializers)
- Modify: `app/api/routes.py:259-324` (stream_turn becomes a real streaming generator; `_run_turn` gains pass-through)
- Modify: `frontend/src/app/api.service.ts` (stage/error frame handling, onStage callback)
- Modify: `frontend/src/app/session-store.ts` (`currentStage` signal)
- Modify: `frontend/src/app/components/message-input.component.ts` (stage line)
- Test: `tests/unit/test_turn_orchestrator.py`, `tests/unit/test_sse.py`, `tests/integration/test_api_turns.py`

**Interfaces:**
- Consumes: existing `_stage_timer` boundaries in `run_turn`; `build_turn_stream_frames(result, *, text_chunk_chars)`.
- Produces: `run_turn(*, turn_input, on_stage: Callable[[str], None] | None = None)`; `serialize_stage_frame(stage: str) -> str` and `serialize_error_frame(*, code: str, message: str, status: int) -> str` in `app/api/sse.py`; SSE events `stage` (`{"stage": "generation"}`) and `error` (`{"code","message","status"}`); `ApiService.createBufferedTurn(sessionId, request, {timeoutMs?, onStage?})`; `SessionStore.currentStage: signal<string | null>`.

- [ ] **Step 1: Write the failing orchestrator test**

In `tests/unit/test_turn_orchestrator.py`, reusing the file's existing orchestrator-with-fakes fixture/helper for a successful turn:

```python
async def test_run_turn_reports_stage_progression():
    orchestrator, turn_input = ...  # same construction as the existing happy-path test
    stages: list[str] = []
    await orchestrator.run_turn(turn_input=turn_input, on_stage=stages.append)
    assert stages[:4] == ["session", "retrieval", "routing", "generation"]
    assert stages[-2:] == ["persistence", "memory"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_turn_orchestrator.py::test_run_turn_reports_stage_progression -v`
Expected: FAIL — `run_turn() got an unexpected keyword argument 'on_stage'`.

- [ ] **Step 3: Implement stage emission in the orchestrator**

In `app/orchestration/turn_orchestrator.py` (add `Callable` to the `collections.abc` import). Module-level helper next to `_stage_timer`:

```python
def _emit_stage(on_stage: Callable[[str], None] | None, stage: str) -> None:
    if on_stage is None:
        return
    try:
        on_stage(stage)
    except Exception:  # noqa: BLE001 - progress reporting must never fail a turn
        pass
```

Change the signature:

```python
    async def run_turn(
        self,
        *,
        turn_input: TurnInput,
        on_stage: Callable[[str], None] | None = None,
    ) -> TurnResult:
```

Insert `_emit_stage(on_stage, "<name>")` immediately before each existing `with _stage_timer(timings, "<name>")` block (`session`, `retrieval`, `routing`, `generation`, `validation`, `critique`, `persistence`, `memory`) and `_emit_stage(on_stage, "repair")` immediately before `self.repair_stage.resolve(...)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_turn_orchestrator.py -v`
Expected: PASS.

- [ ] **Step 5: Write failing SSE serializer test**

In `tests/unit/test_sse.py`:

```python
def test_serialize_stage_frame():
    assert serialize_stage_frame("generation") == 'event: stage\ndata: {"stage":"generation"}\n\n'


def test_serialize_error_frame():
    frame = serialize_error_frame(code="provider_timeout", message="timed out", status=504)
    assert frame.startswith("event: error\n")
    assert '"code":"provider_timeout"' in frame
```

- [ ] **Step 6: Run to verify failure, then implement**

Run: `.venv/bin/python -m pytest tests/unit/test_sse.py -v` → FAIL (ImportError).

In `app/api/schemas.py`:

```python
class StreamStagePayload(BaseModel):
    stage: str


class StreamErrorPayload(BaseModel):
    code: str
    message: str
    status: int
```

In `app/api/sse.py` (extend the schemas import):

```python
def serialize_stage_frame(stage: str) -> str:
    return _serialize_frame("stage", StreamStagePayload(stage=stage))


def serialize_error_frame(*, code: str, message: str, status: int) -> str:
    return _serialize_frame("error", StreamErrorPayload(code=code, message=message, status=status))
```

Re-run: PASS.

- [ ] **Step 7: Rewrite `stream_turn` as a real streaming generator**

In `app/api/routes.py` (add `import asyncio`, `from collections.abc import AsyncIterator`, extend the sse import with the two serializers). `_run_turn` gains a keyword pass-through:

```python
async def _run_turn(
    session_id: str,
    request: CreateTurnRequest,
    services: AppServices,
    *,
    on_stage: Callable[[str], None] | None = None,
) -> TurnResult:
    try:
        result = await services.orchestrator.run_turn(
            turn_input=TurnInput(...unchanged...),
            on_stage=on_stage,
        )
    ...unchanged except handlers...
```

(add `Callable` to the `collections.abc` import at the top of routes.py). Replace `stream_turn`:

```python
async def stream_turn(
    session_id: str,
    request: CreateTurnRequest,
    services: Annotated[AppServices, Depends(get_turn_services)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> StreamingResponse:
    queue: asyncio.Queue[str] = asyncio.Queue()

    def on_stage(stage: str) -> None:
        queue.put_nowait(serialize_stage_frame(stage))

    async def event_stream() -> AsyncIterator[str]:
        turn = asyncio.create_task(
            _run_turn(session_id, request, services, on_stage=on_stage)
        )
        try:
            while not turn.done():
                frame: asyncio.Task[str] = asyncio.create_task(queue.get())
                done, _ = await asyncio.wait({frame, turn}, return_when=asyncio.FIRST_COMPLETED)
                if frame in done:
                    yield frame.result()
                else:
                    frame.cancel()
            while not queue.empty():
                yield queue.get_nowait()
            result = await turn
        except ApiError as exc:
            # The HTTP status is already 200 once streaming starts; errors must
            # travel as a terminal frame instead.
            yield serialize_error_frame(
                code=exc.code, message=exc.message, status=exc.status_code
            )
            return
        for out in build_turn_stream_frames(
            result, text_chunk_chars=settings.sse_text_chunk_chars
        ):
            yield out

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

Verify the attribute names on `ApiError` in `app/api/errors.py` (`code`, `message`, `status_code`) and adjust if they differ.

- [ ] **Step 8: Write failing integration test**

In `tests/integration/test_api_turns.py`, following the file's existing stream-endpoint test pattern:

```python
def test_stream_turn_emits_stage_frames_before_final(...existing fixtures...):
    response = client.post(f"/sessions/{session_id}/turns/stream", json={"message": "Hello"})
    body = response.text
    assert "event: stage" in body
    assert body.index("event: stage") < body.index("event: final")
    # provider errors now arrive as an error frame on an HTTP 200 stream
```

Also update any existing test that asserted an HTTP 4xx/5xx status from `/turns/stream` (e.g. provider-unavailable cases): the stream is now HTTP 200 with a terminal `event: error` frame carrying the same code/status.

- [ ] **Step 9: Run integration tests**

Run: `.venv/bin/python -m pytest tests/integration/test_api_turns.py tests/integration/test_smoke_api.py -v`
Expected: PASS after the assertions in Step 8 are aligned.

- [ ] **Step 10: Wire the SPA — api.service**

In `frontend/src/app/api.service.ts`:
- `parseFrame(result, frame, onStage?)` and `parseEventStream(response, onStage?)` thread a callback through; `createBufferedTurn` accepts it:

```ts
  async createBufferedTurn(
    sessionId: string,
    request: CreateTurnRequest,
    { timeoutMs = STREAM_TIMEOUT_MS, onStage }: { timeoutMs?: number; onStage?: (stage: string) => void } = {},
  ): Promise<TurnResult> {
```

- In `applyEvent` (before the existing event handling):

```ts
  if (eventName === 'stage') {
    onStage?.(String((payload as { stage?: unknown }).stage ?? ''));
    return false;
  }
  if (eventName === 'error') {
    const p = payload as { code?: string; message?: string; status?: number };
    throw new ApiError(p.code ?? 'stream_error', p.message ?? 'Turn failed.', p.status ?? 502);
  }
```

(match the existing `ApiError` constructor argument order in this file).

- [ ] **Step 11: Wire the SPA — store and composer**

`frontend/src/app/session-store.ts`:

```ts
  readonly currentStage = signal<string | null>(null);
```

In `runTurn`, pass the callback and clear it:

```ts
    this.currentStage.set(null);
    try {
      const turn = await this.api.createBufferedTurn(sessionId, request, {
        onStage: (stage) => this.currentStage.set(stage),
      });
      this.applyTurn(message, turn);
    } catch (error) {
      this.turnError.set(errorMessage(error));
    } finally {
      this.currentStage.set(null);
      this.busy.set(false);
    }
```

`frontend/src/app/components/message-input.component.ts` — inside the `<section class="composer">`, above the textarea:

```html
      @if (store.busy() && store.currentStage(); as stage) {
        <p class="stage">{{ stageLabel(stage) }}…</p>
      }
```

```ts
  private static readonly STAGE_LABELS: Record<string, string> = {
    session: 'Loading session',
    retrieval: 'Retrieving memories',
    routing: 'Choosing route',
    generation: 'Drafting reply',
    validation: 'Checking draft',
    critique: 'Critic reviewing',
    repair: 'Repairing draft',
    persistence: 'Saving turn',
    memory: 'Updating memory',
  };

  stageLabel(stage: string): string {
    return MessageInputComponent.STAGE_LABELS[stage] ?? stage;
  }
```

Add `.stage { color: var(--muted); font-size: 0.8rem; margin: 0; }` to the styles.

- [ ] **Step 12: Frontend build + gates + commit**

Run: `cd frontend && npx ng build && npx ng test --watch=false` then `make check && make smoke`
Expected: PASS.

```bash
git add app/orchestration/turn_orchestrator.py app/api/schemas.py app/api/sse.py app/api/routes.py frontend/src/app tests/
git commit -m "feat(sse): live stage-progress frames during turns; errors as terminal stream frames"
```

---

### Task 5: Continuity — draft survives failed turns, resume picker with full transcript

**Files:**
- Modify: `frontend/src/app/session-store.ts` (boolean-returning turn methods, `recentSessions`, full-transcript resume)
- Modify: `frontend/src/app/api.service.ts` (`listRecentSessions`)
- Modify: `frontend/src/app/play-model.ts` (`fullTranscript`)
- Modify: `frontend/src/app/components/message-input.component.ts` (clear draft only on success)
- Modify: `frontend/src/app/components/setup-picker.component.ts` (resume select)
- Test: existing SPA specs (`frontend/src/app/session-store.spec.ts` or equivalent — match existing spec file names)

**Interfaces:**
- Consumes: `GET /sessions` (`RecentSessionsResponse`), `GET /sessions/{id}/turn-details` (`SessionTurnDetailsResponse`), existing `SessionStore.resume(sessionId)`, `formatRecentSessionOption(session)` in play-model.ts.
- Produces: `SessionStore.sendMessage/confirmCloud/forceLocal: Promise<boolean>` (true = turn accepted or confirmation requested); `SessionStore.recentSessions: signal<RecentSessionResponse[]>`; `SessionStore.loadRecentSessions(): Promise<void>`; `fullTranscript(details: SessionTurnDetailsResponse): TranscriptEntry[]`.

- [ ] **Step 1: Write failing store specs**

In the existing session-store spec file (mirror its mock-ApiService pattern):

```ts
it('keeps returning false from sendMessage when the turn fails', async () => {
  apiMock.createBufferedTurn.and.rejectWith(new ApiError('provider_timeout', 'boom', 504));
  const ok = await store.sendMessage('hello', false);
  expect(ok).toBeFalse();
  expect(store.turnError()).toContain('provider_timeout');
});

it('resume loads the full transcript from turn-details', async () => {
  apiMock.getSession.and.resolveTo(sessionDetailFixture);       // 8 recent turns
  apiMock.getSessionTurnDetails.and.resolveTo(turnDetailsFixture); // 20 turns
  await store.resume('s1');
  expect(store.transcript().length).toBe(40); // 20 player + 20 assistant entries
});
```

- [ ] **Step 2: Run specs to verify they fail**

Run: `cd frontend && npx ng test --watch=false`
Expected: FAIL — `sendMessage` returns `Promise<void>`; resume only maps the 8 recent turns.

- [ ] **Step 3: Implement store + play-model changes**

`frontend/src/app/play-model.ts`:

```ts
export function fullTranscript(details: SessionTurnDetailsResponse): TranscriptEntry[] {
  return details.turns.flatMap((turn) => [
    { role: 'player' as const, text: turn.user_message, source: 'resumed' as const },
    { role: 'assistant' as const, text: turn.assistant_message, source: 'resumed' as const },
  ]);
}
```

(align the `source` literal with what `resumeTranscript` at play-model.ts:213 already uses).

`frontend/src/app/api.service.ts`:

```ts
  listRecentSessions(): Promise<RecentSessionsResponse> {
    return requestJson('/sessions');
  }
```

`frontend/src/app/session-store.ts`:

```ts
  readonly recentSessions = signal<RecentSessionResponse[]>([]);

  async loadRecentSessions(): Promise<void> {
    try {
      this.recentSessions.set((await this.api.listRecentSessions()).sessions);
    } catch {
      this.recentSessions.set([]);
    }
  }
```

In `resume()`, replace `this.transcript.set(resumeTranscript(detail));` with:

```ts
      const details = await this.api.getSessionTurnDetails(sessionId);
      this.transcript.set(fullTranscript(details));
```

Turn methods return success:

```ts
  sendMessage(message: string, requestCloud: boolean): Promise<boolean> {
    return this.runTurn(message, buildTurnRequest(message, requestCloud));
  }
  // confirmCloud / forceLocal: same change — return this.runTurn(...) and
  // `return Promise.resolve(true)` when there is no pending confirmation.

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
```

- [ ] **Step 4: Composer clears draft only on success**

`frontend/src/app/components/message-input.component.ts`:

```ts
  async send(): Promise<void> {
    const ok = await this.store.sendMessage(this.draft().trim(), this.requestCloud());
    if (ok) this.draft.set('');
  }
```

- [ ] **Step 5: Resume select in the setup picker**

`frontend/src/app/components/setup-picker.component.ts` — constructor loads the list; template gains a resume block inside the `@else` form:

```ts
export class SetupPickerComponent {
  readonly store = inject(SessionStore);
  readonly playerName = signal('');
  readonly formatOption = formatRecentSessionOption;

  constructor() {
    void this.store.loadRecentSessions();
  }
}
```

```html
        @if (store.recentSessions().length > 0) {
          <label>
            Resume session
            <select #resumeSel>
              @for (s of store.recentSessions(); track s.session_id) {
                <option [value]="s.session_id">{{ formatOption(s) }}</option>
              }
            </select>
          </label>
          <button type="button" (click)="store.resume(resumeSel.value)" [disabled]="store.busy()">
            Resume
          </button>
        }
```

(import `formatRecentSessionOption` from `../play-model`).

- [ ] **Step 6: Run specs, build, e2e**

Run: `cd frontend && npx ng test --watch=false && npx ng build`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/app docs/BACKLOG.md
git commit -m "feat(spa): resume picker with full transcript; draft survives failed turns"
```

(Tick the "SPA session resume" follow-up in docs/BACKLOG.md in the same commit.)

---

### Task 6: Reroll — delete the last turn plus its memories, resend the message

**Files:**
- Modify: `app/persistence/repositories.py` (`delete_last_turn`, `delete_memories_since` + Protocol entries)
- Modify: `app/composition.py` (expose `memory_indexer` on `AppServices`)
- Modify: `app/api/schemas.py` (`DeleteLastTurnResponse`)
- Modify: `app/api/routes.py` (DELETE endpoint)
- Modify: `frontend/src/app/api.service.ts`, `frontend/src/app/session-store.ts`, `frontend/src/app/components/message-input.component.ts`
- Test: `tests/unit/test_repositories.py`, `tests/integration/test_api_turns.py`

**Interfaces:**
- Consumes: `MemoryIndexer.unindex(memory_ids)`, `serialize_datetime` from `app/persistence/sqlite.py`, `StoredTurn.created_at`.
- Produces: `TurnRepository.delete_last_turn(session_id) -> StoredTurn | None`; `MemoryRepository.delete_memories_since(session_id, created_at: datetime) -> list[str]`; `AppServices.memory_indexer: MemoryIndexer | None`; `DELETE /sessions/{id}/turns/last` → `DeleteLastTurnResponse {session_id, deleted_turn_index, user_message, deleted_memory_count}`; `SessionStore.rerollLast()`.

- [ ] **Step 1: Write failing repository tests**

In `tests/unit/test_repositories.py` (reuse its connection/fixture pattern):

```python
def test_delete_last_turn_removes_and_returns_it(...):
    # append two turns via the existing append_turn helper pattern
    deleted = turn_repository.delete_last_turn(session.id)
    assert deleted is not None and deleted.turn_index == 2
    remaining = turn_repository.list_all_turns(session.id)
    assert [t.turn_index for t in remaining] == [1]
    assert turn_repository.delete_last_turn("missing-session") is None


def test_delete_memories_since_removes_only_at_or_after_cutoff(...):
    # append one memory, capture a cutoff datetime AFTER it, append a second memory
    deleted_ids = memory_repository.delete_memories_since(session.id, cutoff)
    assert deleted_ids == [second_memory.id]
    assert [m.id for m in memory_repository.list_memories_for_session(session.id)] == [first_memory.id]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/unit/test_repositories.py -v -k "delete_last or delete_memories"`
Expected: FAIL — AttributeError.

- [ ] **Step 3: Implement repository methods**

`SQLiteTurnRepository`:

```python
    def delete_last_turn(self, session_id: str) -> StoredTurn | None:
        row = self._connection.execute(
            "SELECT * FROM turns WHERE session_id = ? ORDER BY turn_index DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        self._connection.execute("DELETE FROM turns WHERE id = ?", (row["id"],))
        self._connection.commit()
        return self._row_to_turn(row)
```

`SQLiteMemoryRepository` (import `serialize_datetime` alongside the module's existing sqlite helpers):

```python
    def delete_memories_since(self, session_id: str, created_at: datetime) -> list[str]:
        cutoff = serialize_datetime(created_at)
        rows = self._connection.execute(
            "SELECT id FROM memory_episodes WHERE session_id = ? AND created_at >= ?",
            (session_id, cutoff),
        ).fetchall()
        ids = [row["id"] for row in rows]
        if ids:
            placeholders = ",".join("?" for _ in ids)
            self._connection.execute(
                f"DELETE FROM memory_episodes WHERE id IN ({placeholders})", ids
            )
            self._connection.commit()
        return ids
```

Add both to their Protocols (`TurnRepository`, `MemoryRepository`) — any test fakes implementing those Protocols need the new methods stubbed.

- [ ] **Step 4: Run repo tests, then expose the indexer**

Run: `.venv/bin/python -m pytest tests/unit/test_repositories.py -v` → PASS.

In `app/composition.py` `build_services`, extract the inline indexer into a variable and pass it to both places:

```python
    memory_indexer = (
        MemoryIndexer(
            memory_store=memory_store,
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            importance_floor=settings.rag_index_importance_floor,
            session_memory_max_episodes=settings.session_memory_max_episodes,
        )
        if embedding_provider is not None and vector_store is not None
        else None
    )
```

Use `memory_indexer=memory_indexer` in the `TurnOrchestrator(...)` call and add `memory_indexer=memory_indexer` to the returned `AppServices` (new field `memory_indexer: MemoryIndexer | None = None` on the dataclass; import `MemoryIndexer` type).

- [ ] **Step 5: Write failing API test**

In `tests/integration/test_api_turns.py`:

```python
def test_delete_last_turn_reroll_flow(...existing turn fixtures...):
    # run one successful turn first (existing pattern)
    response = client.delete(f"/sessions/{session_id}/turns/last")
    assert response.status_code == 200
    body = response.json()
    assert body["deleted_turn_index"] == 1
    assert body["user_message"]  # the message to resend
    assert client.delete(f"/sessions/{session_id}/turns/last").status_code == 404
```

- [ ] **Step 6: Implement schema + route**

`app/api/schemas.py`:

```python
class DeleteLastTurnResponse(BaseModel):
    session_id: str
    deleted_turn_index: int
    user_message: str
    deleted_memory_count: int
```

`app/api/routes.py`:

```python
@router.delete(
    "/sessions/{session_id}/turns/last",
    response_model=DeleteLastTurnResponse,
    responses=ERROR_404_RESPONSE,
)
def delete_last_turn(
    session_id: str,
    services: Annotated[AppServices, Depends(get_turn_services)],
) -> DeleteLastTurnResponse:
    _require_session(services, session_id)
    if services.turn_repository is None:
        raise RuntimeError("Turn services must include a turn repository")
    turn = services.turn_repository.delete_last_turn(session_id)
    if turn is None:
        raise ApiError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="no_turns",
            message="Session has no turns to delete.",
        )
    deleted_memory_ids: list[str] = []
    if services.memory_repository is not None and turn.created_at is not None:
        # ponytail: provenance by timestamp (memories are written after the turn is
        # persisted); add a turn_id column to memory_episodes if this ever misfires
        deleted_memory_ids = services.memory_repository.delete_memories_since(
            session_id, turn.created_at
        )
    if deleted_memory_ids and services.memory_indexer is not None:
        try:
            services.memory_indexer.unindex(deleted_memory_ids)
        except Exception:  # noqa: BLE001 - index cleanup is best-effort; SQLite is authoritative
            pass
    return DeleteLastTurnResponse(
        session_id=session_id,
        deleted_turn_index=turn.turn_index,
        user_message=turn.user_message,
        deleted_memory_count=len(deleted_memory_ids),
    )
```

Run: `.venv/bin/python -m pytest tests/integration/test_api_turns.py -v` → PASS.

- [ ] **Step 7: SPA reroll button**

`frontend/src/app/api.service.ts`:

```ts
  deleteLastTurn(sessionId: string): Promise<{ user_message: string }> {
    return requestJson(`/sessions/${encodeURIComponent(sessionId)}/turns/last`, { method: 'DELETE' });
  }
```

(match `requestJson`'s existing init-argument shape — see `deleteCanonFact`).

`frontend/src/app/session-store.ts`:

```ts
  async rerollLast(): Promise<void> {
    const sessionId = this.sessionId();
    if (!sessionId || this.busy()) return;
    const lastPlayer = [...this.transcript()].reverse().find((e) => e.role === 'player');
    if (!lastPlayer) return;
    this.turnError.set(null);
    try {
      await this.api.deleteLastTurn(sessionId);
      this.transcript.update((all) => all.slice(0, -2));
    } catch (error) {
      this.turnError.set(errorMessage(error));
      return;
    }
    await this.sendMessage(lastPlayer.text, false);
    void this.refreshMemories();
  }
```

`message-input.component.ts` controls row:

```html
        <button
          type="button"
          class="ghost"
          (click)="store.rerollLast()"
          [disabled]="store.busy() || !store.sessionId()"
        >
          Reroll last
        </button>
```

- [ ] **Step 8: Gates + commit**

Run: `make check && cd frontend && npx ng test --watch=false && npx ng build`

```bash
git add app/persistence/repositories.py app/composition.py app/api frontend/src/app tests/
git commit -m "feat(turns): reroll — delete last turn with its memories and resend"
```

---

### Task 7: Campaign unfreeze — scene switching, per-turn persona switch, cross-session persona memory

**Files:**
- Modify: `app/persistence/repositories.py` (`update_active_scene`, `update_active_persona` + Protocol)
- Modify: `app/orchestration/stages/session.py:98-115` (persona override becomes a switch)
- Modify: `app/api/schemas.py` (`UpdateSceneRequest`), `app/api/routes.py` (POST /sessions/{id}/scene)
- Modify: `app/memory/indexer.py` (persona_memory dual-write)
- Modify: `frontend/src/app/api.service.ts`, `frontend/src/app/play-model.ts` (`buildTurnRequest` persona), `frontend/src/app/session-store.ts`, `frontend/src/app/components/setup-picker.component.ts`
- Test: `tests/unit/test_repositories.py`, `tests/unit/test_turn_orchestrator.py`, `tests/unit/test_memory_indexer.py`, `tests/integration/test_api_sessions.py`

**Interfaces:**
- Consumes: `SessionRepository` Protocol, `TurnSessionLoader.load`, `RagCollection.PERSONA_MEMORY`, `Visibility.PLAYER`, retrieval already searching persona_memory with `RetrievalFilter.player_visible(persona_id=...)` (retriever.py:109).
- Produces: `SessionRepository.update_active_scene(session_id, scene_id) -> None` and `update_active_persona(session_id, persona_id) -> None`; `POST /sessions/{id}/scene {scene_id}` → `CreateSessionResponse`; per-turn `active_persona_id` now switches the session persona instead of raising; `PERSONA_MEMORY_IMPORTANCE_FLOOR = 4` in `app/memory/indexer.py`.

- [ ] **Step 1: Failing repo tests**

```python
def test_update_active_scene_and_persona(...):
    session_repository.update_active_scene(session.id, "east_wing")
    session_repository.update_active_persona(session.id, "warden")
    reloaded = session_repository.get_session(session.id)
    assert reloaded.active_scene_id == "east_wing"
    assert reloaded.active_persona_id == "warden"
```

Run to FAIL, then implement in `SQLiteSessionRepository` (both identical in shape):

```python
    def update_active_scene(self, session_id: str, scene_id: str) -> None:
        self._connection.execute(
            "UPDATE sessions SET active_scene_id = ?, updated_at = ? WHERE id = ?",
            (scene_id, serialize_datetime(utc_now()), session_id),
        )
        self._connection.commit()

    def update_active_persona(self, session_id: str, persona_id: str) -> None:
        self._connection.execute(
            "UPDATE sessions SET active_persona_id = ?, updated_at = ? WHERE id = ?",
            (persona_id, serialize_datetime(utc_now()), session_id),
        )
        self._connection.commit()
```

Add both to the `SessionRepository` Protocol; stub them on any Protocol-conforming fakes in tests. Run to PASS.

- [ ] **Step 2: Persona override becomes a switch**

Failing test in `tests/unit/test_turn_orchestrator.py` (reuse the fixture; the demo world must contain a second persona — use the fixture's own catalog):

```python
async def test_persona_override_switches_the_session_persona(...):
    result = await orchestrator.run_turn(
        turn_input=TurnInput(session_id=session.id, message="Hello", active_persona_id=other_persona_id)
    )
    assert result.outcome == TurnOutcome.SUCCESS
    assert session_repository.get_session(session.id).active_persona_id == other_persona_id
```

Run to FAIL (`ValueError: Turn persona override does not match...`), then rewrite the start of `TurnSessionLoader.load` (app/orchestration/stages/session.py:98-114):

```python
    def load(self, turn_input: TurnInput) -> LoadedTurnContext:
        session = self.resume_session(turn_input.session_id)
        persona_id = session.active_persona_id
        loader = self.loader_for_content_root(session.content_root)
        world = loader.load_world(session.world_id)
        if (
            turn_input.active_persona_id is not None
            and turn_input.active_persona_id != persona_id
        ):
            if turn_input.active_persona_id not in world.persona_ids:
                raise ValueError(
                    f"Unknown persona for world {session.world_id}: "
                    f"{turn_input.active_persona_id}"
                )
            persona_id = turn_input.active_persona_id
            self.session_repository.update_active_persona(session.id, persona_id)
            session = session.model_copy(update={"active_persona_id": persona_id})
        if persona_id not in world.persona_ids:
            raise ValueError(f"Unknown persona for world {session.world_id}: {persona_id}")
        ...rest unchanged (scene check, canon, memories, return)...
```

Delete/replace the existing test asserting the override raises. Run to PASS.

- [ ] **Step 3: Scene-switch endpoint**

Failing test in `tests/integration/test_api_sessions.py`:

```python
def test_update_session_scene(...):
    response = client.post(f"/sessions/{session_id}/scene", json={"scene_id": other_scene_id})
    assert response.status_code == 200
    assert response.json()["active_scene_id"] == other_scene_id
    assert client.post(
        f"/sessions/{session_id}/scene", json={"scene_id": "nope"}
    ).status_code == 400
```

Schema:

```python
class UpdateSceneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scene_id: str = Field(min_length=1, max_length=200)
```

Route in `app/api/routes.py`:

```python
@router.post(
    "/sessions/{session_id}/scene",
    response_model=CreateSessionResponse,
    responses={**ERROR_400_RESPONSE, **ERROR_404_RESPONSE},
)
def update_session_scene(
    session_id: str,
    request: UpdateSceneRequest,
    services: Annotated[AppServices, Depends(get_read_services)],
) -> CreateSessionResponse:
    session = services.session_repository.get_session(session_id)
    if session is None:
        raise ApiError(
            status_code=status.HTTP_404_NOT_FOUND,
            code="session_not_found",
            message=f"Unknown session id: {session_id}",
        )
    loader = services.orchestrator.loader_for_session(session)
    try:
        world = loader.load_world(session.world_id)
        if request.scene_id not in world.scene_ids:
            raise ValueError(
                f"Unknown scene for world {session.world_id}: {request.scene_id}"
            )
        loader.load_scene(request.scene_id)
    except (DataFileNotFoundError, DataValidationError, ValueError) as exc:
        raise ApiError(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="invalid_scene",
            message=_safe_request_error_message(exc),
        ) from exc
    services.session_repository.update_active_scene(session_id, request.scene_id)
    return CreateSessionResponse(
        session_id=session.id,
        world_id=session.world_id,
        active_scene_id=request.scene_id,
        active_persona_id=session.active_persona_id,
    )
```

Run to PASS. (Turns already store per-turn `scene_id`, so history stays coherent across switches.)

- [ ] **Step 4: persona_memory dual-write in the indexer**

Failing test in `tests/unit/test_memory_indexer.py` (use the file's in-memory vector store fixture):

```python
def test_high_value_player_memories_are_also_indexed_per_persona(...):
    # index one PLAYER-visible importance-4 memory with actor_id="innkeeper"
    # and one GM-visible importance-5 memory
    persona_hits = vector_store.search(  # use the fixture's search/inspection helper
        RagCollection.PERSONA_MEMORY, ...
    )
    assert [hit.actor_id for hit in persona_hits] == ["innkeeper"]
```

Implement in `app/memory/indexer.py` (import `Visibility` from `app.domain`):

```python
PERSONA_MEMORY_IMPORTANCE_FLOOR = 4
```

At the end of `index_memories`, after the session upsert and before the cap loop:

```python
        self._index_persona_memories(eligible)
```

```python
    def _index_persona_memories(self, memories: Sequence[MemoryEpisode]) -> None:
        """Cross-session NPC memory: high-value PLAYER-visible episodes are also
        indexed per actor, so a later session with the same persona retrieves them
        (retrieval already searches persona_memory filtered by persona_id)."""
        lasting = [
            memory
            for memory in memories
            if memory.visibility is Visibility.PLAYER
            and memory.actor_id
            and memory.importance >= PERSONA_MEMORY_IMPORTANCE_FLOOR
        ]
        if not lasting:
            return
        chunks = [
            self._to_chunk(memory).model_copy(
                update={"source_type": RagCollection.PERSONA_MEMORY.value}
            )
            for memory in lasting
        ]
        vectors = self.embedding_provider.embed_batch([chunk.text for chunk in chunks])
        self.vector_store.ensure_collection(
            RagCollection.PERSONA_MEMORY, self.embedding_provider.dimension
        )
        self.vector_store.upsert_chunks(RagCollection.PERSONA_MEMORY, chunks, vectors)
```

Extend `unindex` so Task 6's reroll also cleans persona copies:

```python
    def unindex(self, memory_ids: Sequence[str]) -> None:
        self.vector_store.delete_points(RagCollection.SESSION_MEMORY, list(memory_ids))
        try:
            self.vector_store.delete_points(RagCollection.PERSONA_MEMORY, list(memory_ids))
        except Exception:  # noqa: BLE001 - collection may not exist yet
            pass
```

Verify the persona filter field: `RetrievalFilter.player_visible(persona_id=...)` must match the chunk's `actor_id` payload (check `app/rag/retriever.py:35-50` and the vector-store filter mapping; `_to_chunk` sets `actor_id=memory.actor_id`). If the filter matches a different payload key, set that key in the persona chunk instead. Run to PASS.

- [ ] **Step 5: SPA — persona per turn + scene switch controls**

`frontend/src/app/play-model.ts` — extend `buildTurnRequest` with an optional persona (keep existing call sites valid):

```ts
export function buildTurnRequest(
  message: string,
  requestCloud: boolean,
  options: { cloudConfirmed?: boolean; forceLocal?: boolean; personaId?: string | null } = {},
): CreateTurnRequest {
  return {
    message,
    request_cloud: requestCloud,
    cloud_confirmed: options.cloudConfirmed ?? false,
    force_local: options.forceLocal ?? false,
    active_persona_id: options.personaId ?? undefined,
  };
}
```

(align with the existing implementation shape at play-model.ts:144; add `active_persona_id?: string` to the `CreateTurnRequest` interface in models.ts).

`frontend/src/app/api.service.ts` — include it in the POST body of `createBufferedTurn`:

```ts
        active_persona_id: request.active_persona_id ?? null,
```

and add:

```ts
  updateSessionScene(sessionId: string, sceneId: string): Promise<CreateSessionResponse> {
    return requestJson(`/sessions/${encodeURIComponent(sessionId)}/scene`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ scene_id: sceneId }),
    });
  }
```

(match `requestJson`'s existing POST pattern from `addCanonFact`).

`frontend/src/app/session-store.ts`:

```ts
  readonly personaOverride = signal<string | null>(null);

  sendMessage(message: string, requestCloud: boolean): Promise<boolean> {
    return this.runTurn(
      message,
      buildTurnRequest(message, requestCloud, { personaId: this.personaOverride() }),
    );
  }

  async switchScene(sceneId: string): Promise<void> {
    const sessionId = this.sessionId();
    if (!sessionId || !sceneId) return;
    try {
      this.session.set(await this.api.updateSessionScene(sessionId, sceneId));
      this.turnError.set(null);
    } catch (error) {
      this.turnError.set(errorMessage(error));
    }
  }
```

`setup-picker.component.ts` — replace the bare active-session line with switch controls:

```html
    @if (store.session(); as session) {
      <p class="active">Session active: {{ store.sessionId() }}</p>
      <label>
        Scene
        <select #switchScene>
          @for (scene of store.catalog()?.scenes ?? []; track scene.id) {
            <option [value]="scene.id" [selected]="scene.id === session.active_scene_id">
              {{ scene.title }}
            </option>
          }
        </select>
      </label>
      <button type="button" (click)="store.switchScene(switchScene.value)" [disabled]="store.busy()">
        Switch scene
      </button>
      <label>
        Persona (next turn)
        <select #switchPersona (change)="store.personaOverride.set(switchPersona.value)">
          @for (persona of store.catalog()?.personas ?? []; track persona.id) {
            <option [value]="persona.id" [selected]="persona.id === session.active_persona_id">
              {{ persona.name }}
            </option>
          }
        </select>
      </label>
    } @else { ...existing form... }
```

- [ ] **Step 6: Gates + commit**

Run: `make check && .venv/bin/python -m app.evals.regression_runner && make smoke && cd frontend && npx ng test --watch=false && npx ng build`

```bash
git add app tests frontend/src/app
git commit -m "feat(campaign): scene switching, per-turn persona switch, cross-session persona memory"
```

---

### Task 8: Memory curation off the critical path (API turns only)

**Files:**
- Modify: `app/domain/models.py` (`DeferredMemoryJob`, `TurnResult.deferred_memory`)
- Modify: `app/orchestration/turn_orchestrator.py` (`defer_memory` param, `run_deferred_memory`)
- Modify: `app/persistence/repositories.py` (`append_memory_outcome` + Protocol)
- Modify: `app/api/routes.py` (schedule the deferred job after responding)
- Test: `tests/unit/test_turn_orchestrator.py`, `tests/unit/test_repositories.py`, `tests/integration/test_api_turns.py`

**Interfaces:**
- Consumes: Task 3's cached embedding/vector singletons (makes the background `build_services` cheap); Task 1's WAL (background writer + next request's reader coexist).
- Produces: `run_turn(*, turn_input, on_stage=None, defer_memory: bool = False)`; `TurnOrchestrator.run_deferred_memory(job: DeferredMemoryJob) -> None`; `TurnRepository.append_memory_outcome(turn_id, *, memory_written: bool, warnings: list[str]) -> None`; `DeferredMemoryJob {session_id, turn_id, user_message, assistant_message, retrieval_confidence, scene_complexity}`. CLI behavior unchanged (`defer_memory` defaults False).

- [ ] **Step 1: Domain model**

In `app/domain/models.py`, next to `TurnResult`:

```python
class DeferredMemoryJob(BaseModel):
    """Inputs the memory stage needs when it runs after the response is sent."""

    session_id: str
    turn_id: int
    user_message: str
    assistant_message: str
    retrieval_confidence: float | None
    scene_complexity: int
```

Add to `TurnResult`:

```python
    deferred_memory: "DeferredMemoryJob | None" = Field(default=None, exclude=True)
```

Export `DeferredMemoryJob` from `app/domain/__init__.py`.

- [ ] **Step 2: Failing orchestrator tests**

In `tests/unit/test_turn_orchestrator.py`:

```python
async def test_defer_memory_skips_curation_and_returns_a_job(...):
    result = await orchestrator.run_turn(turn_input=turn_input, defer_memory=True)
    assert result.memory_written is False
    assert result.deferred_memory is not None
    assert result.deferred_memory.assistant_message == result.text
    assert any("memory curation deferred" in w for w in result.warnings)
    assert fake_curator.calls == 0  # the fixture's curator was never invoked


async def test_run_deferred_memory_writes_and_updates_diagnostics(...):
    result = await orchestrator.run_turn(turn_input=turn_input, defer_memory=True)
    await orchestrator.run_deferred_memory(result.deferred_memory)
    assert fake_curator.calls == 1
    stored = turn_repository.list_all_turns(session.id)[-1]
    assert stored.diagnostics.memory_written is True
```

- [ ] **Step 3: Run to verify failure, implement orchestrator**

Run: `.venv/bin/python -m pytest tests/unit/test_turn_orchestrator.py -v -k defer` → FAIL.

In `run_turn`, signature becomes:

```python
    async def run_turn(
        self,
        *,
        turn_input: TurnInput,
        on_stage: Callable[[str], None] | None = None,
        defer_memory: bool = False,
    ) -> TurnResult:
```

Replace the block from the memory stage-timer through the final `return` (turn_orchestrator.py:402-436):

```python
        if defer_memory:
            warnings.append("memory curation deferred: runs after this response")
            self.turn_repository.update_turn_diagnostics(
                persistence.turn.id,
                TurnDiagnostics(
                    retrieval=retrieval.diagnostics,
                    stage_timings=timings,
                    critic_status=critic_status,
                    finish_reason=final_finish_reason,
                    warnings=warnings,
                    memory_written=False,
                ),
            )
            return TurnResult(
                text=final_text,
                route=final_route,
                finish_reason=final_finish_reason,
                memory_written=False,
                critic_status=critic_status,
                warnings=warnings,
                retrieval=retrieval.diagnostics,
                stage_timings=timings,
                deferred_memory=DeferredMemoryJob(
                    session_id=context.session.id,
                    turn_id=persistence.turn.id,
                    user_message=turn_input.message,
                    assistant_message=final_text,
                    retrieval_confidence=retrieval.confidence,
                    scene_complexity=routing.scene_complexity,
                ),
            )
        _emit_stage(on_stage, "memory")
        with _stage_timer(timings, "memory"):
            ...existing memory stage call, diagnostics update, and return unchanged...
```

New method (below `run_turn`):

```python
    async def run_deferred_memory(self, job: DeferredMemoryJob) -> None:
        """Memory stage for an already-persisted turn, after the response was sent."""
        session = self.session_stage.resume_session(job.session_id)
        loader = self.session_stage.loader_for_content_root(session.content_root)
        persona = loader.load_persona(session.active_persona_id)
        scene = loader.load_scene(session.active_scene_id)
        memory = await self.memory_stage.run(
            session=session,
            scene=scene,
            persona=persona,
            user_message=job.user_message,
            assistant_message=job.assistant_message,
            retrieval_confidence=job.retrieval_confidence,
            scene_complexity=job.scene_complexity,
        )
        self.turn_repository.append_memory_outcome(
            job.turn_id,
            memory_written=memory.memory_written,
            warnings=list(memory.warnings),
        )
```

Import `DeferredMemoryJob` from `app.domain`.

- [ ] **Step 4: Repository merge method**

Failing test in `tests/unit/test_repositories.py`:

```python
def test_append_memory_outcome_merges_into_diagnostics(...):
    # append a turn, then update_turn_diagnostics with memory_written=False and one warning
    turn_repository.append_memory_outcome(turn_id, memory_written=True, warnings=["memory dedup dropped 1 duplicate candidate(s)"])
    stored = turn_repository.list_all_turns(session.id)[-1]
    assert stored.diagnostics.memory_written is True
    assert len(stored.diagnostics.warnings) == 2
```

Implement in `SQLiteTurnRepository` (the module already imports `json`):

```python
    def append_memory_outcome(
        self, turn_id: int, *, memory_written: bool, warnings: list[str]
    ) -> None:
        row = self._connection.execute(
            "SELECT diagnostics_json FROM turns WHERE id = ?", (turn_id,)
        ).fetchone()
        if row is None or row["diagnostics_json"] is None:
            return
        payload = json.loads(row["diagnostics_json"])
        payload["memory_written"] = memory_written
        payload["warnings"] = [*payload.get("warnings", []), *warnings]
        self._connection.execute(
            "UPDATE turns SET diagnostics_json = ? WHERE id = ?",
            (json.dumps(payload), turn_id),
        )
        self._connection.commit()
```

Add to the `TurnRepository` Protocol (stub on fakes). Run to PASS.

- [ ] **Step 5: Schedule from the API layer**

In `app/api/routes.py` (module level, near the top):

```python
import logging

logger = logging.getLogger(__name__)
_DEFERRED_MEMORY_TASKS: set[asyncio.Task[None]] = set()


async def _run_deferred_memory_job(job: DeferredMemoryJob, settings: Settings) -> None:
    # Fresh services: the request-scoped connection closes with the response.
    # Cheap because embedding/vector clients are process-cached (Task 3).
    services = build_services(settings, enable_retrieval=True)
    try:
        await services.orchestrator.run_deferred_memory(job)
    except Exception:  # noqa: BLE001
        # ponytail: best-effort post-response; single-user, no per-session lock —
        # add one if concurrent turns per session ever become real
        logger.warning("deferred memory curation failed", exc_info=True)
    finally:
        services.close()


def _schedule_deferred_memory(result: TurnResult, settings: Settings) -> None:
    if result.deferred_memory is None:
        return
    task = asyncio.create_task(_run_deferred_memory_job(result.deferred_memory, settings))
    _DEFERRED_MEMORY_TASKS.add(task)
    task.add_done_callback(_DEFERRED_MEMORY_TASKS.discard)
```

`create_turn` gains `settings: Annotated[Settings, Depends(get_settings)]` and defers:

```python
async def create_turn(...):
    result = await _run_turn(session_id, request, services, defer_memory=True)
    _schedule_deferred_memory(result, settings)
    return _to_turn_response(result)
```

`_run_turn` passes it through: add `defer_memory: bool = False` keyword and forward to `run_turn`. In Task 4's `event_stream`, after yielding the final frames:

```python
        _schedule_deferred_memory(result, settings)
```

Note: the orchestrator no longer emits a `memory` stage frame on deferred turns — remove `memory` from any stage-order assertion written in Task 4 for API-path tests (the unit test in Task 4 uses `defer_memory=False` and stays valid).

- [ ] **Step 6: Integration test + gates**

In `tests/integration/test_api_turns.py`:

```python
def test_api_turn_defers_memory_curation(...):
    body = client.post(f"/sessions/{session_id}/turns", json={"message": "I promise to help."}).json()
    assert body["memory_written"] is False
    assert any("memory curation deferred" in w for w in body["warnings"])
```

Then await the scheduled task deterministically (TestClient runs the app's event loop): expose the pending set via `from app.api import routes` and, inside an async test or via `anyio`, `await asyncio.gather(*routes._DEFERRED_MEMORY_TASKS)` before asserting the memory landed via `GET /sessions/{id}/memories`. If the existing tests are sync-only, keep the assertion to the deferred warning + a unit-level guarantee from Step 2 (both paths are covered).

Run: `make check && .venv/bin/python -m app.evals.regression_runner && make smoke`
Expected: PASS — CLI and smoke-run still curate synchronously (`defer_memory` defaults False).

- [ ] **Step 7: Commit**

```bash
git add app/domain app/orchestration app/persistence app/api tests
git commit -m "perf(turns): run memory curation after the response on API turns"
```

---

## Self-Review Checklist (run after writing, before executing)

- Spec coverage: durability (T1), contradiction loss (T2), per-turn model reload (T3), dead-air turns (T4), lost drafts + no resume (T5), no reroll (T6), frozen campaign + amnesiac NPCs (T7), curation on the critical path (T8). Deliberately out of scope (YAGNI until a consumer exists): token streaming, hybrid sparse retrieval, consolidation-defaults tuning, in-app content editor, FTS transcript search.
- Type consistency spot-checks: `DeferredMemoryJob` fields match `run_deferred_memory` usage; `delete_memories_since(session_id, created_at)` matches the route call; `AppServices.memory_indexer` set in T6 and reused in T7's `unindex` note; `buildTurnRequest` option object matches all three store call sites; `on_stage` name identical across orchestrator, `_run_turn`, and stream route.
- Known executor verifications flagged inline: `ApiError` attribute names (T4), `requestJson` init shape (T5/T6/T7), persona-filter payload key (T7), existing spec file names (T5).
