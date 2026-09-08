# TDD Evidence: Step 0B — stale-model rebind (`_require_model_under_lock` returns the slot)

**Source plan:** docs/plans/2026-09-07-step-0b-stale-model-rebind.plan.md
(Step 0B of docs/plans/2026-09-06-consolidated-backlog-priority.plan.md)

**Branch:** `fix/require-model-under-lock-returns-model`

## Problem

Every generation/prompt-creation path captures `state.models[mode]` into a
local BEFORE acquiring `inference_lock`, then (T5) calls
`_require_model_under_lock(state, mode)` under the lock purely for its
side-effecting 503-raise-on-None check. The return value was discarded, so
an unload→RELOAD landing in the capture→acquire window left the slot
non-`None` (the None-check passes) while the local variable the caller
actually uses for inference still pointed at the orphaned pre-unload object.
`unload_model_cleanup` is `gc.collect()` + cache flush — it never destroys
the weights — so the orphan silently keeps working, masking the bug: the
symptom is stale/wrong model behavior under interleaved
unload/reload/generate traffic, not a crash.

## Task report

| Task | Site | Command | RED | GREEN |
|---|---|---|---|---|
| 1 | Guard signature (`_require_model_under_lock`, `app_generation.py`) | `pytest tests/test_issue214_unload_queued_window.py::TestPostLockSlotReRead::test_guard_returns_the_slot_it_validates -v --tb=short` | `AssertionError: None is not <MagicMock name='design-model' id='4404528272'> : the guard must return the re-read slot so callers can rebind` | `PASSED` — `1 passed in 0.10s`; class regression `5 passed`; file regression `17 passed` |
| 2 | Batch `/generate` (`handle_generate`) | `pytest tests/test_issue214_unload_queued_window.py::TestPostLockSlotReRead::test_batch_runs_on_the_model_reloaded_while_queued -v --tb=short` | `AssertionError: <MagicMock name='clone-model' id='4391907984'> is not <MagicMock name='reloaded-clone' id='4406629648'> : inference ran on the pre-unload orphan: the under-lock re-read was checked but not rebound` — `1 failed in 0.20s` | `PASSED`; class `6 passed in 0.13s`; file `18 passed in 2.81s`; broader sweep (23 files referencing `app_generation`/`handle_generate`) `452 passed` |
| 3 | `/generate-stream` (`audio_stream_generator`, nested in `handle_generate_stream`) | `pytest tests/test_issue214_unload_queued_window.py::TestPostLockSlotReRead::test_streaming_uses_the_model_reloaded_before_body_iteration -v --tb=short` | `AssertionError: <MagicMock name='design-model' id='4605412240'> is not <MagicMock name='reloaded-design' id='4605679888'> : the stream thread ran on the pre-unload orphan — the under-lock re-read is not rebound into the thread's scope` — `1 failed in 0.21s` | `PASSED`; class `7 passed in 0.13s`; file `19 passed in 2.77s`; streaming-adjacent sweep (`-k "streaming or app_generation or generate_stream"`) `153 passed, 3108 deselected` |
| 4 | `/ws` (`_stream_generation`, `websocket.py`) | `pytest tests/test_websocket.py::TestWebSocketFreshSlotUnderLock -v --tb=short` | `AssertionError: <MagicMock name='original-clone' id='4584008912'> is not <MagicMock name='reloaded-clone' id='4584178704'> : the /ws inference thread ran on the pre-unload orphan — the under-lock re-read is not rebound` — `1 failed in 0.16s` | `PASSED`; whole file `43 passed in 0.57s`; AST enclosure pin `test_all_guarded_sites_re_read_inside_their_locks` `1 passed` |
| 5 | `/create-voice-prompt` torch path (`handle_create_voice_prompt`, `app_prompts.py`) | `pytest tests/test_issue192_create_prompt_serialization.py::TestCreatePromptSerialization::test_create_rereads_clone_slot_under_lock_after_reload -v --tb=short` | `AssertionError: <MagicMock id='4543056400'> is not <MagicMock name='reloaded-clone' id='4543061648'> : the prompt was built on the pre-unload orphan — no under-lock re-read and rebind` — `1 failed in 0.93s` | `PASSED`; three-file run (`test_issue192_create_prompt_serialization.py` + `test_issue236_mlx_create_prompt.py` + `test_issue214_prompt_create_serialization.py`) `43 passed in 3.93s` |

Every RED failure above was a genuine assertion failure at collection (not a
skip — `_HAS_DEPS`/`HAS_DEPS` were true throughout, confirmed per-task by the
reports), on exactly the line and object identity the fix was designed to
close. Source: `.superpowers/sdd/detailed-step-0b-execution-floating-glacier/task-{1,2,3,4,5}-report.md`.

## Source hunks

### 1 — guard returns the re-read slot (`qwen3_tts/server/app_generation.py`, commit `9a9e8b9`)

```diff
-def _require_model_under_lock(state, mode) -> None:
+def _require_model_under_lock(state, mode) -> Any:
     """Re-read the model slot UNDER inference_lock and bail when it is gone.
@@
+
+    Returns:
+        The object currently occupying ``state.models[mode]`` — NOT the local
+        the caller captured pre-lock. Callers MUST rebind: ``model =
+        _require_model_under_lock(state, mode)``. Returning the re-read slot
+        closes the second half of the window: an unload→RELOAD between capture
+        and acquire leaves the slot non-None, so a None check alone passes and
+        inference runs on the ORPHANED pre-unload object (``unload_model_cleanup``
+        is gc.collect + cache flush — it never destroys weights). Residual,
+        accepted: /load-model assigns its slot without inference_lock, so a
+        load finishing between this return and the inference call is not
+        observed.
     """
-    if state.models.get(mode) is None:
+    current = state.models.get(mode)  # .get(), never [mode]: partial-dict test states exist
+    if current is None:
         logger.warning(...)
         ...
         return  # explicit guard — _error_response raises, but it is
         # typed -> None, not NoReturn, so nothing structurally stops a
         # fall-through into inference with the slot gone.
+    return current
```

### 2 — batch `/generate` rebind (`qwen3_tts/server/app_generation.py`, commit `33cc447`)

```diff
                 # T5: the slot was read into a local BEFORE this acquire;
                 # re-validate it here — an unload that landed in the
                 # capture->acquire window must surface as a retryable 503,
-                # never as an orphan generation.
-                _require_model_under_lock(state, mode)
+                # never as an orphan generation. Rebind the re-read slot: an
+                # unload->RELOAD in that window leaves the slot NON-None, so
+                # checking alone would still run inference on the orphaned
+                # pre-unload object.
+                model = _require_model_under_lock(state, mode)
```

### 3 — `/generate-stream` rebind into the thread's scope (`qwen3_tts/server/app_generation.py`, commit `83664a1`)

```diff
             try:
-                _require_model_under_lock(state, mode)
+                # Rebind, don't just check: this assignment makes `model` a
+                # local of audio_stream_generator, and inference_thread
+                # (nested below) resolves it from THIS scope's cell rather
+                # than handle_generate_stream's — so thread.start() below
+                # sees the re-read slot, never the pre-lock capture.
+                model = _require_model_under_lock(state, mode)
             except HTTPException as e:
```

### 4 — `/ws` rebind before `thread.start()` (`qwen3_tts/server/websocket.py`, commit `d8a0cc0`)

```diff
         try:
-            _require_model_under_lock(app_state, mode)
+            # Rebind the re-read slot: inference_thread (nested above) reads
+            # `model` from THIS function's cell, so the rebind must land
+            # before thread.start() below — an unload->RELOAD in the
+            # capture->acquire window otherwise leaves the thread on the
+            # orphaned pre-unload object.
+            model = _require_model_under_lock(app_state, mode)
         except _HTTPException as e:
```

A fifth hunk (`qwen3_tts/server/app_prompts.py`, commit `43195be`) adds the
guard to `/create-voice-prompt`'s torch path — closing a gap where there was
no post-lock recheck at all, rather than fixing a discarded-return-value
bug at an existing check site:

```diff
         async with state.inference_lock:
+            # T5 sibling: the clone slot was captured before four awaits
+            # (decode, stage, audio load). Re-read it UNDER the lock and
+            # rebind, so an unload->RELOAD in that window builds the prompt
+            # on the CURRENT model — and an unload alone surfaces as the
+            # same retryable 503 the generation paths raise. Function-local
+            # import: same precedent as websocket.py; keeps this module free
+            # of a new module-level sibling-handler dependency.
+            from qwen3_tts.server.app_generation import _require_model_under_lock
+
+            model = _require_model_under_lock(state, "clone")
             voice_prompt = await asyncio.to_thread(
                 create_voice_prompt,
                 model,
```

## Scope hand-off rationale (streaming and WS sites)

Both the `/generate-stream` and `/ws` fixes touch the return-value discard at
a call site that lives inside a **nested closure**, not the handler's own
top-level scope:

- `/generate-stream`: `_require_model_under_lock(state, mode)` is called
  inside `audio_stream_generator`, an async generator nested inside
  `handle_generate_stream`. `handle_generate_stream` itself already has a
  pre-lock `model = state.models.get(mode)` local. Before this fix, the
  guard call inside `audio_stream_generator` used that outer `model` only
  implicitly (via closure) once inference actually ran — but the guard's own
  return value was thrown away, so nothing in `audio_stream_generator`'s own
  scope was ever updated. `model = _require_model_under_lock(state, mode)`
  creates a **new local cell in `audio_stream_generator`'s own scope** that
  shadows the outer capture from that line forward. Because Python closures
  resolve names by scope, not by call time, `inference_thread` — itself
  nested inside `audio_stream_generator`, below this line — now reads
  `model` from `audio_stream_generator`'s cell, which was just rebound.
  `grep -n "nonlocal model\|global model"` across `qwen3_tts/server` returns
  zero hits, confirming there is no `nonlocal`/`global` anywhere that would
  instead reach back and mutate `handle_generate_stream`'s outer binding —
  the shadow-a-new-local shape is exactly what the fix relies on.
- `/ws`: the same shape, one level shallower. `_stream_generation` captures
  `model` pre-lock, then (post-fix) rebinds it under
  `async with app_state.inference_lock:` **before** `inference_thread.start()`
  is called a few lines below. `inference_thread` is a nested function
  defined inside `_stream_generation` and reads `model` via closure from
  `_stream_generation`'s own scope. Ordering is load-bearing here in a way
  it structurally cannot be violated for the streaming site (an async
  generator only starts its thread once execution reaches that point) but
  is worth stating explicitly for `/ws`: the rebind assignment is placed
  strictly before the `thread.start()` call, so the thread — once it starts
  and later resolves the closure cell — sees the freshly rebound object, not
  whatever `model` held when the function began. Had the rebind been placed
  after `thread.start()`, a data race would exist between the assignment and
  the thread's first read of `model`; placing it before removes that
  race entirely (the thread cannot begin executing until `start()` returns
  control, and the rebind precedes that call).

In both cases the `except`-block error-frame construction was left
byte-identical (confirmed by diff review) — only the try line itself changed
shape, from a bare call to an assignment.

## Accepted residual (documented in Task 1's guard docstring, verbatim)

> Residual, accepted: /load-model assigns its slot without inference_lock,
> so a load finishing between this return and the inference call is not
> observed.

`/load-model`'s load body assigns `state.models[mode]` under
`MODEL_LOAD_LOCK` only (see `qwen3_tts/server/model_loading.py`), never
under `inference_lock`. That means a load that completes in the
sub-millisecond gap between `_require_model_under_lock`'s re-read/return and
the caller's actual `run_inference`/`create_voice_prompt` call is not
observed by this fix — the caller could still run on whatever was in the
slot at the instant of the re-read, one instant before a concurrent load
finishes and swaps it again. Closing that residual would require holding
`inference_lock` across the entire model-load path, which is explicitly
rejected by the #212/#214 per-load-record design (loads and inference are
deliberately allowed to interleave via `MODEL_LOAD_LOCK` + per-load-record
CAS, not serialized behind the single coarse `inference_lock` — see
`qwen3_tts/server/model_loading.py` module docstring). This fix closes the
unload→RELOAD orphan window (T5's originally-scoped concern: a stale local
persisting across an entire generation/prompt-creation call); it does not
attempt to make the guard's re-read instant atomic with the caller's use of
it, which is a strictly narrower, accepted gap.

## Gates (Steps 1–3, observed)

| Gate | Command | Result |
|---|---|---|
| Spec verify (1) | `pytest tests/test_issue214_unload_queued_window.py tests/test_voice_server.py -v --tb=short` | `82 passed in 3.96s` |
| Touched suites (2) | `pytest tests/test_websocket.py tests/test_issue192_create_prompt_serialization.py tests/test_issue236_mlx_create_prompt.py tests/test_issue214_prompt_create_serialization.py tests/test_create_voice_endpoint.py tests/test_generation_lock_scope.py tests/test_python_review_fixes.py -q --tb=short` | `111 passed in 7.76s` |
| ruff | `ruff check qwen3_tts tests` | `All checks passed!` |
| mypy | `mypy qwen3_tts/core qwen3_tts/server qwen3_tts/interface` | `Success: no issues found in 57 source files` (only pre-existing `annotation-unchecked` notes on untyped functions, unrelated to this change) |
| batch 3 (Server Infrastructure) | `python tests/run_batches.py --batch 3` | `Ran 750 tests in 26.285s` — `OK (skipped=3)`; "1/1 batches passed" |
| batch 2 (Voice & CLI) | `python tests/run_batches.py --batch 2` | `Ran 529 tests in 5.291s` — `OK`; "1/1 batches passed" |
| baseline suite | `pytest tests/ -m "not e2e" --ignore=tests/evaluations --tb=short -q` | `3151 passed, 4 skipped, 92 deselected, 3 warnings in 45.15s` — zero failures |

No failures were observed at any gate; there was accordingly no need to
compare against the `29039eb` branch point to prove pre-existence — the
"known-red baseline caveat" in the task brief did not materialize on this
run. The 4 skips and 92 deselections match the shape of the prior known-green
baseline runs recorded elsewhere in this repo's TDD evidence docs (e.g.
`docs/testing/max-chunk-chars-bounds.tdd.md`'s `3146 passed, 4 skipped, 92
deselected`) — the deselections are `e2e`-marked tests excluded by
`-m "not e2e"`, and the skip count is stable across runs.

## Known-wrong claim in the source plan (flagged, not silently edited)

`docs/plans/2026-09-07-step-0b-stale-model-rebind.plan.md` (the tracked copy
of `~/.claude/plans/detailed-step-0b-execution-floating-glacier.md`) contains
one factually incorrect attribution in its Task 4 injection-point note:

> "the WS path's pre-lock prompt seam is `qwen3_tts.core.engine.load_voice_prompt`
> (function-local import in `_stream_generation`, ...)"

`load_voice_prompt` is not function-local in `_stream_generation`
(`qwen3_tts/server/websocket.py`). It is function-local in
`qwen3_tts/server/prompt_loading.py` (`from qwen3_tts.core.engine import
VoicePromptCreateRequired, load_voice_prompt`), reached from
`_stream_generation` indirectly via `load_voice_prompt_serialized`. Verified
by direct grep of `qwen3_tts/server/prompt_loading.py`:

```
27:    from qwen3_tts.core.engine import VoicePromptCreateRequired, load_voice_prompt
30:        return await asyncio.to_thread(load_voice_prompt, prompt_file, allow_create=False)
```

The plan's conclusion — that patching
`qwen3_tts.core.engine.load_voice_prompt` correctly intercepts the call and
the side-effect swap lands in the capture→acquire window — is correct
despite the wrong "in `_stream_generation`" location claim: patching the
name at its definition site (`qwen3_tts.core.engine.load_voice_prompt`)
intercepts every caller that imports it by that name, including
`prompt_loading.py`'s function-local import, regardless of which function
textually contains the `import` statement. Per this task's constraints, the
plan copy is committed as-is (not edited); this section is the flag.

## Test file summary (from Tasks 1–5)

- `tests/test_issue214_unload_queued_window.py`: `TestPostLockSlotReRead` grew
  from 4 tests (pre-existing) to 8 (guard-return, batch-reload,
  streaming-reload, plus the two pre-existing null/AST-pin tests already in
  the class), file total 19 tests, all green.
- `tests/test_websocket.py`: new class `TestWebSocketFreshSlotUnderLock` (1
  test), file total 43 tests (was 42), all green.
- `tests/test_issue192_create_prompt_serialization.py`: new test
  `test_create_rereads_clone_slot_under_lock_after_reload`, plus a docstring/
  assertion-message rationale refresh (assertion logic unchanged) on
  `test_create_uses_captured_clone_model_reference`; module + two sibling
  files (`test_issue236_mlx_create_prompt.py`,
  `test_issue214_prompt_create_serialization.py`) total 43 tests, all green.

## Merge evidence

Commit list (in order):

- `9a9e8b9` — `refactor(server): _require_model_under_lock returns the re-read slot (callers still ignore it)`
- `33cc447` — `fix(server): batch /generate rebinds the under-lock model slot (unload->RELOAD orphan)`
- `83664a1` — `fix(server): /generate-stream rebinds the under-lock model slot into the inference thread's scope`
- `d8a0cc0` — `fix(server): /ws rebinds the under-lock model slot before the inference thread starts`
- `43195be` — `fix(server): /create-voice-prompt re-reads the clone slot under inference_lock (T5 reload half)`
- this docs commit (evidence report + tracked plan copy)

## Addendum — adversarial fix wave (seven findings)

An adversarial hunt over the five hunks above executed mutants against the
suite and demonstrated seven findings, numbered 1–5, 7 and 8 in
`.superpowers/sdd/detailed-step-0b-execution-floating-glacier/fix-wave-report.md`
(finding 6 — the pre-existing no-op `_validate_generation_request` patches —
was parked by controller ruling, not closed). All seven findings are closed:
finding 1 below, findings 2–5 and 7 in the gap table, finding 8 in its last
row.

### Finding 1 — the fifth capture path (`qwen3_tts/server/prompt_loading.py`)

`load_voice_prompt_serialized` captures `state.models["clone"]` and then waits
on `inference_lock` — and that wait is a whole queued generation, the widest
capture→acquire window in the server. The locked call is REAL torch
create inference (auto-create-from-`.wav`), and `_require_model_under_lock` was
never called there.

The guard is **provenance-split**, because `load_model()`
(`core/engine/model_loader.py`) returns the model and never writes
`state.models` — verified by reading its body: it dispatches to
`_load_model_{mlx,torch}`, optionally warms up, and returns. So on the fallback
branch the slot is legitimately still `None` under the lock, and an
unconditional guard would 503 a path that works today:

- captured from the slot → `model = _require_model_under_lock(state, "clone")`
  (an empty slot now is a real unload → the same retryable 503)
- built locally by the fallback → keep the built model, but prefer a slot that
  a concurrent `/load-model` published while we waited.

`tests/test_issue214_prompt_create_serialization.py::TestLoadVoicePromptSerializedSlotReRead`
covers all four cases and drives the REAL contended window (the test holds
`inference_lock`, the coroutine parks on the acquire, the slot is mutated while
it waits, then the lock is released). The fallback-preservation case
(`test_locally_built_model_survives_an_empty_slot_under_the_lock`) was green
before and after — it exists to fail an unconditional guard.

### Test-side gaps closed

| Gap | Mutant that survived | Pin added |
|---|---|---|
| The batch reload test swaps at the prompt-load seam, reachable only under `if mode == "clone":` | rebind only for clone, bare call otherwise — **3155/3155 other non-e2e tests green** | `test_batch_rebinds_for_{design,custom}_after_a_contended_wait`: hold the lock, park the handler on the acquire, swap, release |
| The create path was structurally unpinned (`test_all_guarded_sites_re_read_inside_their_locks`, then still named for the two sites it started from, iterated only streaming + `/ws`) | hoist the `/create-voice-prompt` re-read outside the lock — **3157/3157 other non-e2e tests green** | `_GUARDED_SITES`: one list of all five capture paths, used by three pins |
| Nothing required the guard's result to be ASSIGNED | bare `_require_model_under_lock(...)` in `/ws` | `test_every_guarded_site_assigns_the_guard_result` (`ast.Assign` whose value is the call) |
| Nothing pinned that the `/ws` rebind precedes `thread.start()` | move the rebind below `thread.start()` | `test_thread_starting_sites_rebind_before_start` (statement order inside the lock body) |
| The `/ws` reload test's error check was a dict-KEY sniff (`"error" in m`); the real failure frame is `{"status": "error", "detail": …}` | a stub that records the model then raises — every assertion passed with zero audio | positive assertions on `sent_json[-1]["status"] == "complete"` and `len(sent_bytes)` |
| The create path's null half was promised in a docstring, asserted nowhere | `model = state.models.get("clone") or model` | `test_create_bails_with_retryable_503_when_slot_nulled_in_window` |
| `_RecordingAsyncLock` was installed on the reload and contended windows without asserting `acquire_calls` (finding 8) | a handler that never acquired `inference_lock` at all — a pre-lock rebind — satisfies every identity assertion while the capture→acquire window stays open | `acquire_calls >= 1` on the batch and streaming reload tests, `>= 2` on the two contended-window tests (the test's own acquire plus the handler's) |

Partial refutation, recorded rather than dropped: the ordering finding assumed
the reordered `/ws` rebind would be *a flake, not a failure*. On this host it
failed `TestWebSocketFreshSlotUnderLock` 10 out of 10 runs — the inference
thread happened to win every time. The race is still real (nothing synchronises
the thread's first read of `model` with the rebind); the structural pin makes
the outcome deterministic instead of scheduling-dependent. The finding's other
half — that the existing AST pin still passes — held.
