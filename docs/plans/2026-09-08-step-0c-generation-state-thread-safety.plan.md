# Execution Plan — Step 0C: `generation_state` threading discipline

Spec: `docs/plans/2026-09-06-consolidated-backlog-priority.plan.md` Wave 0, Step 0C (lines ~131-152).
Branch: `fix/generation-state-thread-safety` (spec-named), cut from main @ `1bc081c` (the 0B merge).
Execution: superpowers:subagent-driven-development. Controller never implements.

## Global Constraints

1. The guard owns a **`threading.Lock`** — NOT asyncio. Both worker threads (batch `to_thread`, streaming daemon) and the event loop touch this state.
2. **Preserve semantics — atomicity only.** The per-item `cancelled: False` re-clear inside `begin()` stays (erasing a cancel that landed pre-lock is #237/Step 1A's race, explicitly NOT this branch's). No behavioral change beyond serialization/atomicity.
3. **`state.generation_state` REMAINS the same plain-dict storage object.** The guard late-binds it (resolves `app_state.generation_state` at call time), so the ~130 existing test references that seed/read the dict keep working. Production reads/writes route through the guard; tests may read the dict directly for assertions.
4. **`state.generation_lock` (asyncio.Lock) SURVIVES but no longer guards `generation_state`.** Its one remaining user is `app_models.py` `/update-model-config` slot-nulling (`:328`) — which guards model SLOTS, not generation_state. Leave it untouched; ledger-note it as a pre-existing oddity (backlog line).
5. Guard module is **stdlib-only** (`threading`, `typing`) — no FastAPI/torch/mlx imports; respects the lazy-import rule by construction.
6. mypy clean (guard module fully typed); ruff clean; every new test RUN-not-skip in the torchless `.venv-310` proxy; new test modules registered in `tests/run_batches.py` BATCHES (or INTENTIONALLY_UNBATCHED with reason).
7. Async tests: `IsolatedAsyncioTestCase`; event-driven interleavings, ZERO sleeps (0B patterns: event-before-acquire).
8. Repo constants: never stage `config.json`; no AI attribution; never `--amend`; commit with explicit pathspec; python/pytest/ruff/mypy via `conda run -n qwen3-tts-mlx`; CLAUDE.md ≤300 lines.
9. Line numbers in this plan are DESCRIPTIVE (post-0B tree, surveyed 2026-09-08); implementers locate by content.

## Controller pre-flight findings (ruled, carried to dispatches)

- **F1 — the plan's clobbered-cancel interleaving looks impossible as stated.** All `cancelled` KEY writes are event-loop-side (batch `:301`/begin/finally, streaming begin/finally, ws begin/finally, cancel) — a single-threaded loop orders them, so no loop-vs-loop lost-cancel exists. The acquire→begin window on the streaming path (`:858`→`:873-883`) contains NO await, so no other coroutine can interleave there. The REAL demonstrable defects: (a) worker-THREAD mutations unserialized vs the loop (`_chunk_progress` both paths, called via `progress_callback` inside `run_inference`/`run_inference_streaming`); (b) torn multi-key UNLOCKED reads (`/generation-status` `:601-608`, `/queue-status` `:618-621`, `handle_unload_model` `:226-228`, `detect_degraded_generation` `:145+`, cancel's post-lock `:866` read). **Ruling: implement the guard + full routing + atomic snapshots as planned; the RED test targets the demonstrable torn-read defect (deterministic construction in Task 4); Task 2's implementer FIRST verifies the no-await claim against source and records it in its report — if an await DOES exist in that window, the original clobber RED applies instead.** Cost if wrong: a clobber window I missed ships untested — mitigated by the implementer's verification + the final review.
- **F2 — unlocked sites are WIDER than the plan named.** Batch `finally` reset (`:700-716`) and the batch pre-loop clear (`:301`) are unlocked too. They are in scope (routing), and this plan documents them.
- **F3 — #237 boundary.** #237 ("Cancel landing in a batch item's pre-lock window is erased by the in-lock cancelled re-clear") is OPEN and is Step 1A's fix. This branch's `begin()` PRESERVES the re-clear semantics; the guard exists so 1A's fix is a semantics change at one atomic point. PR body must cross-reference #237 (exit criterion).
- **F4 — paperwork rides this branch** (the 0B ruling): 0B status-mark + the 12 follow-up items fold into the consolidated plan as a SEPARATE docs commit (0B's `12ce975` precedent). KIT extraction is NOT executed here (it's a follow-up bullet, not 0C scope).

## The Guard API (Task 1 produces this exactly)

New module `qwen3_tts/server/generation_state_guard.py` (stdlib-only, fully typed):

```python
def guard_for(app_state) -> GenerationStateGuard:
    """Return the app's guard, constructing-and-caching it on first touch.

    Production: lifespan provisions it eagerly. Fakes/tests that never
    provisioned one get a transient guard on first handler touch — late
    binding of app_state.generation_state keeps both paths correct.
    """

class GenerationStateGuard:
    def __init__(self, app_state): ...          # stores app_state + threading.Lock()
    # _state property: LATE-BINDS app_state.generation_state at each call
    def begin(self, generation_id, *, mode="", text_length=0, start_time=None,
              batch_index=0, batch_total=0) -> None
        # in-lock update: active=True, generation_id, cancelled=False (re-clear
        # semantics PRESERVED — #237 is Step 1A's), plus provided fields
    def update_progress(self, chunk_index, chunk_total) -> None   # thread-safe
    def clear_cancelled(self) -> None           # batch pre-loop clear (:301)
    def set_cancelled(self) -> None             # /cancel-generation (:862)
    def is_cancelled(self) -> bool              # replaces _should_stop_streaming's dict arg
    def reset_if_owner(self, generation_id) -> bool   # ownership-guarded full reset; True iff it reset
    def snapshot(self, keys=None) -> dict       # atomic shallow copy (default: all keys)
```

Lifespan wiring (`app_lifespan.py` init block, after `generation_state` dict creation):
`app.state.generation_state_guard = GenerationStateGuard(app.state)` (eager; `guard_for` remains the accessor so self-provisioning covers fakes).

## Task 1 — guard module + wiring (no routing yet)

- New `qwen3_tts/server/generation_state_guard.py` per the API above; docstrings carry constraints 1-3 rationale (why threading.Lock; why late binding; re-clear preserved for #237).
- Lifespan eager provisioning. `guard_for()` lives in the same module.
- New `tests/test_generation_state_guard.py`: unit tests per method (begin keyset incl. cancelled=False; reset_if_owner True/False; snapshot atomic copy is not the dict itself + reflects later mutation; update_progress); a concurrency smoke: writer threads set `chunk_index=i, chunk_total=2*i` pairs while a reader thread snapshots and asserts `chunk_total == 2*chunk_index` on every snapshot (torn-read impossibility under the lock); register module in `run_batches.py` BATCHES.
- Gates: file tests green; ruff; mypy on core/server/interface; `.venv-310` RUN-not-skip on the new module tests.

## Task 2 — batch path routing (`app_generation.py`)

Route: `:301` → `guard.clear_cancelled()`; `:360` → `guard.is_cancelled()`; begin `:457-472` → `guard.begin(batch_gen_id, mode=…, text_length=…, batch_index=i, batch_total=len(texts))` (still inside `inference_lock`, replacing the `generation_lock` acquire); `_chunk_progress` `:474-480` → `guard.update_progress(...)` (NOW actually thread-safe); `:570` chunk_count read → `guard.snapshot(["chunk_total"])` (stays inside `inference_lock` — the comment's requirement); final-item reset `:582-600` → `guard.reset_if_owner(batch_gen_id)` (replacing the generation_lock acquire); `finally` `:700-716` → `guard.reset_if_owner(batch_gen_id)`.
- ALSO: verify F1's no-await claim on the streaming acquire→begin window and RECORD the finding in the report.
- Tests: existing batch tests (test_issue214_unload_queued_window, test_voice_server, test_batch_generation_state_ownership) must stay green UNCHANGED (constraint 3 — they seed the dict). New test: batch `_chunk_progress` mutation is serialized — a fake progress callback driven from a worker thread while the loop mutates, asserting no lost chunk_index/chunk_total pairing (event-synchronized, zero sleeps).
- Report must list each routed site + its guard call.

## Task 3 — streaming + `/ws` routing

- `app_generation.py` streaming: begin `:874-883` → `guard.begin(gen_id, mode=…, text_length=…)`; `_chunk_progress` `:885-892` → `guard.update_progress(...)`; `_should_stop_streaming` signature changes from `(stop_event, generation_state)` to `(stop_event, guard)` using `guard.is_cancelled()` (update its docstring; check call sites + direct tests of the helper); finally `:981-993` → `guard.reset_if_owner(gen_id)`.
- `websocket.py`: begin `:531-541` → `guard.begin(ws_gen_id, ...)` (replacing the generation_lock acquire); finally `:599-612` → `guard.reset_if_owner(ws_gen_id)`.
- Tests: existing ws/streaming suites green unchanged; new: streaming progress writes from the inference thread serialize against loop-side cancel (event-synchronized).

## Task 4 — readers + the RED torn-read regression

- `app.py` `/generation-status` `:596-609` → single `guard.snapshot()`; build the response from the snapshot (sensitivity stripping unchanged). `/queue-status` `:618-621` → `snapshot(["active"])`. `/cancel-generation` `:859-866` → `snapshot`-based active check + `guard.set_cancelled()` + post-decision `generation_id` from a snapshot (the `:866` post-lock read was itself unlocked); `generation_lock` acquire REMOVED there (constraint 4).
- `app_models.py` `handle_unload_model` `:226-228` → `snapshot(["active", "mode"])` (direct `["active"]` indexing gone — also removes the KeyError-on-missing-key fragility).
- `app_lifespan.py` `detect_degraded_generation` `:145+` → `guard_for(app_state).snapshot()` for active/start_time/text_length; public-endpoint callers keep taking only the boolean (docstring unchanged).
- **RED FIRST (the plan's named regression, ruling F1 shape):** deterministic torn-read test on `/generation-status`: seed `generation_state = {active: True, start_time: 1000.0, ...}`; patch `app_generation`/`app` module `time.time` (the one `generation_status` calls) with a fake that, on first invocation, signals an event and waits for the test to run a full state reset (via a second thread or direct call), then returns a FIXED value. RED: current unlocked read yields `active=True` (old) + `start_time=0.0` (new) → `elapsed_sec = FIXED - 0.0` (garbage). GREEN: snapshot reads one consistent pre-reset (or post-reset) state → `elapsed_sec = FIXED - 1000.0` (or elapsed omitted). IsolatedAsyncioTestCase, zero sleeps, event-driven.
- Response-contract tests (`test_response_contracts.py`) must stay green.

## Task 5 — structural pin + gates + evidence + paperwork

- **Exit-criterion structural pin** (0B `_GUARDED_SITES` precedent): a test asserting production modules contain NO direct `generation_state` dict accesses — grep/AST over `app_generation.py`, `websocket.py`, `app_models.py`, `app.py`, `app_lifespan.py` for `.generation_state[`, `.generation_state.update(`, `.generation_state.get(`, `.generation_state =` (allow: the lifespan INIT block that creates the dict, and the guard module itself). Missing-target-is-LOUD per 0B principle.
- Gates: full server-suite + full non-e2e (`--ignore=tests/evaluations`), ruff, mypy, `.venv-310` RUN-not-skip for every new test, batch registration present.
- Evidence doc `docs/testing/generation-state-guard.tdd.md` (0B precedent): per-task RED/GREEN history, the F1 verification result, Final state block with counted totals.
- Tracked plan copy `docs/plans/2026-09-08-step-0c-generation-state-thread-safety.plan.md`.
- Consolidated plan edits (SEPARATE docs commit, `12ce975` precedent): 0B status-mark (✅ executed via PR #269, `1bc081c`) + the 12 follow-up items recorded under their steps + a 0C note (F1's finding re the clobber window, for Step 1A's benefit).
- CLAUDE.md architecture rows: one sentence on the guard if it earns it (≤300 lines).

## Verify / Exit criteria (from the spec)

- Verify: full server test suite; ruff; mypy.
- Exit: every `generation_state` read/write goes through the guard (structural pin); the previously-unguarded streaming surface fixed with a regression test; #237 cross-reference confirmed in the PR body; no duplicate of Step 1A's fix (re-clear semantics preserved).

## Task ordering & dependencies

T1 (guard exists) → T2 (batch) → T3 (streaming+ws) → T4 (readers + RED) → T5 (pin + gates + docs). T2/T3/T4 each independently committable; T5 last. Parallelism: none (single implementer at a time, per SDD).
