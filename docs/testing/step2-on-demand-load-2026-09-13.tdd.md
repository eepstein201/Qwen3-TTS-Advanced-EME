# Step 2 (Lane H): on-demand model load in `/generate` — TDD evidence

Date: 2026-09-13 · Branch: `fix/on-demand-model-load-generate` (worktree cut from main @ `230a3c81`)
Plan: `docs/plans/2026-09-06-consolidated-backlog-priority.plan.md` § Step 2 (~line 580)

## Hard gate

The plan's merge gate — "do not merge until Wave 4 (Step 4B) has either ruled out or fixed the
server's repeated-load/unload memory-pressure crash" — was **satisfied 2026-09-13 by PR #297**:
`docs/testing/step4b-load-cycle-investigation-2026-09-13.tdd.md` records a ≥5-cycle load/unload
campaign with no death, no leak, and a flat footprint at 8bit (warm reloads 0.7–2 s).

## Design decision

Both `handle_generate` (batch) and `handle_generate_stream` resolve the model slot at the SAME
point the old `model_not_loaded` 503 fired — after validation + the memory guard, BEFORE the
pre-lock cache check, the prompt load, and any lock acquisition. When the slot is empty they now
`await load_model_deduped(state, mode, request=request)` — a direct reuse of
`qwen3_tts/server/model_loading.py`'s per-load-record owner, exactly the composition
`/load-model`'s handler uses (binding risk note 1: no bespoke unlocked `await load_model(...)`).

Why this is safe with respect to the lock order:

- `load_model_deduped` runs the weight construction in `asyncio.to_thread` WITHOUT
  `inference_lock` (its documented design: minutes of download/construction must not starve
  `/generate`), and takes `inference_lock` only for the design warm-up, as a LEAF. At the call
  site the generation handler holds NO lock, so the leaf acquisition preserves the
  inference_lock-outermost order — no lock-order inversion is constructible.
- `MODEL_LOAD_LOCK` (module-scope `threading.Lock`) is only ever held across short non-awaiting
  critical sections (claim / epoch check+assign / release); it is never held across the warm-up
  lock wait, so no inference_lock→MODEL_LOAD_LOCK cycle exists.
- Concurrency safety (double-load, waiter classification, epoch supersede, `/unload-model`
  409-during-load) is inherited from the #214 item-3 machinery rather than reimplemented.
- Failure shape (binding risk note 2): `load_model_deduped`'s except paths call
  `_recover_from_failed_load` (PRF-5) and raise `_error_response(...)` — the SAME sanitized,
  classified envelope `/load-model` produces. Note the plan's wording said "sanitized 503
  shape": `/load-model`'s OWNER failure is actually a classified **500** `load_failed`
  (`recovery: restart`); only its waiter paths are 503 (retryable). The binding requirements —
  same shape as `/load-model`, recovery via `_recover_from_failed_load`, never a bare 500 —
  hold by construction; the status code matches `/load-model` exactly.

Residual `model_not_loaded` 503 (task 5): after `load_model_deduped` returns, the handler
re-reads the slot; if it is empty again (an `/unload-model` raced past the finished load —
supersede normally catches this, but the window after a successful return exists), the 503 now
says to load via `POST /load-model` or the Manage Models tab with `recovery: "load_model"`
(previously `recovery: "restart"` with no pointer). This value matches the envelope fixture
already used in `tests/test_extract_error.py`.

`/ws` is deliberately unchanged (stays opportunistic, 6R item 19).

### Accepted design consequences (documented, not fixed)

- An all-cache-hit `/generate` on an unloaded model now pays a load (0.7–2 s warm per 4B)
  because the model gate precedes the pre-lock cache check. Moving the load after the cache
  check would restructure the handler for a rare, cheap case; not done.
- Client timeouts: `TTSClient`'s `_generation_timeout` floor (600 s) covers warm and
  local-disk loads; a first-ever multi-GB HF download through `/generate` could exceed it —
  the same residual `/load-model`'s 900 s client budget has. Not changed (no speculative
  knobs).
- During the load window the request is not yet in `pending_requests`, so
  `/cancel-generation` cannot target it (same as `/load-model`, which is not cancellable).
- Rate limiting still gates `/generate` (10/min, auth required), bounding load-spam; the
  memory guard runs before the load attempt.

## RED → GREEN

New module `tests/test_generate_on_demand_load.py` (registered in `BATCHES` batch 3 ("Server Infrastructure"), beside
`test_issue214_load_model_dedup`).

RED (pre-implementation, `pytest tests/test_generate_on_demand_load.py -v`): **7 failed in
2.33s** — every case died on the old behavior, e.g.

```
AssertionError: 503 != 200 : {"detail":{"error":"model_not_loaded","detail":"The 'design'
model is not loaded. Generate voice from text description (design mode)","recovery":"restart",
"model_type":"design"}}
```

(all seven: design batch, custom batch, design stream, load-failure shape, concurrent
double-load, racing-generation interleaving, post-load recheck text — each failed with the
`model_not_loaded` 503 or, for the recheck test, an empty deduped-calls list).

GREEN (post-implementation): **7 passed in 2.32s**.

Tests:

- `TestGenerateOnDemandLoad` (TestClient, real app, engine mocked at the facade):
  `test_generate_design_loads_model_on_demand`, `test_generate_custom_loads_model_on_demand`,
  `test_generate_stream_design_loads_model_on_demand`,
  `test_generate_on_demand_load_failure_returns_sanitized_shape`.
- `TestOnDemandLoadConcurrency` (direct `handle_generate` drives, one event loop):
  `test_two_concurrent_first_requests_load_design_once`,
  `test_on_demand_load_races_inflight_generation_without_deadlock`,
  `test_post_load_unload_race_points_at_load_model`.

## Concurrency results (task 4 + coordinator addition)

- **Double-load:** two simultaneous first-requests for unloaded `design` (owner parks in
  `load_model` on an Event; 0.3 s timer release) → `load_model` called exactly once
  (`calls == ["design"]`), both requests 200 with one result each, slot holds the sentinel,
  claim slot released. Falls out of `claim_model_load`; tested explicitly through the
  `/generate` path (not just `/load-model`).
- **Racing generation (4B gap — serial-only there):** a `custom` generation parks INSIDE
  `run_inference` holding `inference_lock`; a concurrent `design` request triggers the
  on-demand load. Ordered-marker assertions prove the interleaving:
  `custom_inference_start < design_load < custom_inference_end < design_warmup <
  design_inference` — i.e. weight construction completes while the sibling generation still
  holds the lock (it never needs it), the design warm-up runs only after the lock is released
  (and observes `inference_lock.locked() == True` inside its body), and the design generation
  runs last. A lock-ordering deadlock surfaces as the `asyncio.wait_for(..., 30)` timeout
  error rather than a hang. This is the strongest interleaving constructible without a live
  GPU; a true preemption-race test is impractical in-process (the event loop serializes
  coroutine steps between awaits, which is exactly the scheduling the markers pin).

## Legacy tests updated (they pinned the old 503-without-load)

Four endpoint tests broke when the implementation landed — and demonstrated why they had to
change: unpatched, they attempted REAL multi-GB MLX loads in-process (log: "Loaded speech
tokenizer from ~/.cache/huggingface/hub/models--mlx-community--Qwen3-TTS-12Hz-1.7B-…").

- `tests/test_fastapi_endpoints.py::test_generate_model_not_loaded` and
  `::test_generate_stream_model_not_loaded` — now stub `engine.load_model` to raise and pin
  the sanitized `load_failed` 500 shape (`recovery: restart`, path redacted).
- `tests/test_fastapi_app_ext2.py::TestGenerateStream::test_stream_model_not_loaded` — same.
- `tests/test_issue214_unload_queued_window.py::test_pre_stream_model_not_loaded_body_carries_detail_and_recovery`
  — now stubs `model_loading.load_model_deduped` (returns "loaded", slot stays None) and pins
  the residual 503 body keys (`error`/`detail`/`recovery`/`model_type`).
- `tests/test_voice_server.py::test_generate_valid_speaker_accepted` — passed before only by
  luck of the `[200, 503]` accept; now stubs the load + inference and asserts the 200 (and
  resets the slot after).

E2E note (dispatcher-owned, NOT run here): `tests/test_ai_regression.py`
`test_*_generation_fails_gracefully_when_model_not_loaded` drives the LIVE :5123 server
(main's code, unchanged) — post-merge those three will need reworking for the new behavior.

## Gates

- Targeted: `pytest tests/test_voice_server.py -v -k generate` → **18 passed, 45 deselected**.
- Touched modules together (`test_generate_on_demand_load`, `test_voice_server`,
  `test_fastapi_endpoints`, `test_fastapi_app_ext2`, `test_issue214_unload_queued_window`)
  → **178 passed** after fixes (70 + module reruns; see final suite below).
- `ruff check qwen3_tts tests` → **All checks passed!**
- `mypy qwen3_tts/{core,server}` → **Success: no issues found in 37 source files**
  (annotation-unchecked notes only, pre-existing).
- `pytest tests/test_claude_md.py` → **4 passed** (CLAUDE.md 203 ≤ 300).
- Full non-E2E suite (`pytest tests/ -m "not e2e" -q`): **3420 passed, 6 skipped, 90
  deselected, 0 failed in 102.59s**.

## Docs

- CLAUDE.md: Server API `/generate` + `/generate-stream` rows, the `load_at_startup` Key
  Settings row, and the `app_generation.py` architecture row now describe the on-demand load
  (in-place edits, no line growth).
- `docs/RUNBOOK.md` "Loading Models" already said models load "on demand when a request needs
  them" — aspirational before, true now; no edit required (checked for contradicting
  troubleshooting text: none — the "Model Loading Fails" entry covers real load failures).

## Live smoke — SKIPPED (architecturally blocked, evidence captured)

Memory headroom was sufficient (32% free ≥ 25%), but the second-instance launch is blocked by
the server's single-instance guard: `_acquire_startup_lock()` flocks a FIXED-path
`.voice_server.lock` (`USER_FILES_DIR` is hardcoded, deliberately no env override — CodeQL
py/path-injection note in `paths.py::_resolve_config_path`), and the PM2 :5123 server holds it.
The attempted :5125 instance aborted in lifespan before binding the port — by design, before
touching the shared `TOKEN_FILE` — so the PM2 server was never at risk (verified after: `/health`
still `ok`, all three models resident, token untouched):

```
RuntimeError: Another TTS server instance is already running or starting
(lock held on /Users/ericepstein/Qwen3-TTS_UserFiles/.voice_server.lock)
ERROR: Application startup failed. Exiting.
```

The only workarounds (patching the path constants at launch, or stopping the PM2 server) are
forbidden by the lane rules (never entangle :5123; a patched launch would not test the real
composition anyway). Precedent: 4B's own deviation note records no :5125 instance either.
Compensating coverage: the unit tests exercise the REAL handler → `load_model_deduped` →
`claim_model_load` → `asyncio.to_thread` → slot-assign flow with real asyncio locks and the
real record table — only the engine's `load_model`/`run_inference` boundaries are stubbed.

## Deviations

- Plan wording "sanitized 503 shape" vs. `/load-model`'s actual classified-500 owner failure —
  resolved in favor of exact `/load-model` parity (see Design decision).
- Batch/E2E gates and `tests/run_batches.py` execution are dispatcher-owned per the brief;
  module registration in `BATCHES` was done, the batch run itself was not executed here.
- Live smoke skipped — blocked by the single-instance startup lock, not memory (see above).
